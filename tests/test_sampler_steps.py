import csv
import json
from pathlib import Path

from colmap_mask.sampler.resume import (
    load_workflow_state,
    step_is_complete,
    step_key,
    step_signature,
)
from colmap_mask.sampler.workflow import PipelineSettings, build_pipeline_steps


def _settings(tmp_path: Path) -> PipelineSettings:
    return PipelineSettings(
        video_path=tmp_path / "video.mp4",
        output_dir=tmp_path / "out",
        executable=tmp_path / "run_video_slam.exe",
        vocabulary=tmp_path / "orb_vocab.fbow",
    )


def test_steps_invoke_vendored_modules(tmp_path):
    steps = build_pipeline_steps(_settings(tmp_path), python_executable=Path("PY"))
    names = [step.name for step in steps]
    assert names == ["中間動画生成", "軌跡抽出", "フレーム選定", "画像抽出"]

    joined = ["\n".join(step.command) for step in steps]
    # Every step shells out to the vendored package, never the old standalone one.
    assert all("colmap_mask.sampler" in cmd for cmd in joined)
    assert "frame_sampler." not in "".join(joined)
    assert "colmap_mask.sampler.video_cli\ncreate-proxy" in joined[0]
    assert "colmap_mask.sampler.video_cli\nrun-stella" in joined[1]
    assert "colmap_mask.sampler.cli\nsample" in joined[2]
    assert "colmap_mask.sampler.video_cli\nextract" in joined[3]

    # Progress slices are contiguous and span the full bar.
    assert steps[0].progress_start == 0.0
    assert steps[-1].progress_end == 1.0
    for earlier, later in zip(steps, steps[1:]):
        assert earlier.progress_end == later.progress_start


def test_extract_targets_output_frames_dir(tmp_path):
    settings = _settings(tmp_path)
    extract = build_pipeline_steps(settings, python_executable=Path("PY"))[3]
    expected = str((settings.output_dir / "frames").resolve())
    assert expected in extract.command


def test_fresh_output_marks_all_stages_incomplete(tmp_path):
    settings = _settings(tmp_path)
    settings.output_dir.mkdir(parents=True)
    state = load_workflow_state(settings.output_dir)
    for name in ("中間動画生成", "軌跡抽出", "フレーム選定", "画像抽出"):
        key = step_key(name)
        signature = step_signature(settings, key)
        assert step_is_complete(settings, key, state, signature) is False


def test_extract_stage_skipped_when_outputs_present(tmp_path):
    """A fully-extracted output folder makes the extract stage resume-complete.

    This is the mechanism the GUI relies on to skip already-done stages, so a
    user can re-run cheaply and (with an upstream trajectory.csv) run without the
    stella binary.
    """
    settings = _settings(tmp_path)
    output = settings.output_dir
    sampled = output / "sampled"
    frames = output / "frames"
    sampled.mkdir(parents=True)
    frames.mkdir(parents=True)

    selected = sampled / "selected_frames.csv"
    with selected.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selection_order", "frame_index", "pts", "time_base_num", "time_base_den"])
        writer.writerow([0, 0, 0, 1, 30])

    image_name = "000000000_000000_0.jpg"
    (frames / image_name).write_bytes(b"jpegbytes")
    with (frames / "extracted_frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selection_order", "frame_index", "pts", "time_base_num", "time_base_den", "export_path"])
        writer.writerow([0, 0, 0, 1, 30, image_name])
    (frames / "extraction_report.json").write_text(
        json.dumps({"requested_count": 1, "extracted_count": 1, "written_count": 1, "skipped_existing_count": 0}),
        encoding="utf-8",
    )

    state = load_workflow_state(output)
    key = step_key("画像抽出")
    signature = step_signature(settings, key)
    assert step_is_complete(settings, key, state, signature) is True
