import json

from colmap_mask.sampler import events


def _line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def test_non_json_passes_through():
    assert events.format_log_line("plain text") == "plain text"


def test_stage_lifecycle_labels():
    start = events.format_log_line(
        _line({"event": "stage_start", "stage_index": 2, "total_steps": 4, "stage_name": "軌跡抽出"})
    )
    assert start == "[開始 2/4] 軌跡抽出"
    done = events.format_log_line(
        _line({"event": "stage_complete", "stage_index": 2, "total_steps": 4, "stage_name": "軌跡抽出"})
    )
    assert done == "[完了 2/4] 軌跡抽出"
    skip = events.format_log_line(
        _line({"event": "skip", "stage_index": 1, "total_steps": 4, "stage_name": "中間動画生成"})
    )
    assert skip == "[再利用 1/4] 中間動画生成"
    error = events.format_log_line(
        _line({"event": "stage_error", "stage_index": 3, "total_steps": 4, "stage_name": "フレーム選定", "message": "boom"})
    )
    assert error == "[失敗 3/4] フレーム選定: boom"


def test_pipeline_start_and_complete():
    assert events.format_log_line(_line({"event": "pipeline_start", "total_steps": 4})) == "[実行開始] 全4工程"
    assert events.format_log_line(_line({"event": "pipeline_complete"})) == "[全工程完了]"


def test_progress_lines_are_suppressed_in_log():
    assert events.format_log_line(_line({"event": "progress", "stage": "proxy", "progress": 0.3})) is None


def test_extract_complete_summary():
    summary = events.format_log_line(
        _line(
            {
                "event": "complete",
                "written_count": 10,
                "skipped_existing_count": 2,
                "extracted_count": 12,
            }
        )
    )
    assert summary == "  [画像抽出] 出力10件 / スキップ2件（選定12件）"


def test_substage_name():
    assert events.substage_name(_line({"event": "substage", "stage_name": "ORB特徴抽出"})) == "ORB特徴抽出"
    assert events.substage_name(_line({"event": "progress"})) is None


def test_extract_progress_status_scanning_then_extracting():
    scanning = events.extract_progress_status(
        _line({"event": "progress", "stage": "extract", "done": 0, "total": 5, "decoded": 90})
    )
    assert scanning == "解析中 90フレーム"
    extracting = events.extract_progress_status(
        _line({"event": "progress", "stage": "extract", "done": 3, "total": 5, "decoded": 200})
    )
    assert extracting == "画像抽出 3/5"
    assert events.extract_progress_status(_line({"event": "progress", "stage": "proxy", "progress": 0.5})) is None


def test_local_and_overall_progress():
    assert events.local_progress(_line({"event": "progress", "progress": 0.5})) == 0.5
    assert events.local_progress(_line({"event": "progress", "progress": 5.0})) == 1.0
    assert events.local_progress("not json") is None
    # A step occupying [0.2, 0.72] at 50% local progress maps to the slice midpoint.
    assert events.overall_progress(0.2, 0.72, 0.5) == 0.46
