"""Pure parsing helpers for the sampler pipeline's JSON Lines stdout.

The pipeline subprocesses (``colmap_mask.sampler.cli`` / ``video_cli``) print one
JSON object per line. These helpers turn those lines into human-readable log text,
sub-stage labels, and an overall progress fraction. They are deliberately free of
any Qt dependency so the GUI step runner and unit tests can share them.

Ported from the standalone Tkinter GUI (frame_sampler/gui.py).
"""

from __future__ import annotations

import json


def format_log_line(line: str) -> str | None:
    """Return a human-readable Japanese log line, or None to suppress the line.

    Non-JSON lines pass through unchanged so stray subprocess output is still shown.
    """
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return line
    event = payload.get("event")
    if event == "pipeline_start":
        return f"[実行開始] 全{payload.get('total_steps', '?')}工程"
    if event == "pipeline_complete":
        return "[全工程完了]"
    if event in {"stage_start", "stage_complete", "skip", "stage_cancelled", "stage_error"}:
        position = f"{payload.get('stage_index', '?')}/{payload.get('total_steps', '?')}"
        name = payload.get("stage_name") or payload.get("stage") or "不明"
        labels = {
            "stage_start": "開始",
            "stage_complete": "完了",
            "skip": "再利用",
            "stage_cancelled": "中止",
            "stage_error": "失敗",
        }
        result = f"[{labels[event]} {position}] {name}"
        message = payload.get("message")
        if event == "stage_error" and message:
            result += f": {message}"
        return result
    if event == "substage":
        return f"  [選定] {payload.get('stage_name', '不明')}"
    if event == "error":
        return f"[エラー] {payload.get('message', line)}"
    if event == "complete" and payload.get("skipped_existing_count") is not None:
        written = payload.get("written_count")
        skipped = payload.get("skipped_existing_count")
        extracted = payload.get("extracted_count")
        summary = f"  [画像抽出] 出力{written}件 / スキップ{skipped}件"
        if extracted is not None:
            summary += f"（選定{extracted}件）"
        return summary
    if event in {"progress", "complete"}:
        return None
    return line


def extract_progress_status(line: str) -> str | None:
    """Return a short status string for image-extraction progress events."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if payload.get("event") != "progress" or payload.get("stage") != "extract":
        return None
    done = payload.get("done")
    total = payload.get("total")
    decoded = payload.get("decoded")
    if done is None or total is None:
        return None
    if done == 0 and decoded:
        # Still scanning the video toward the first selected frame.
        return f"解析中 {decoded}フレーム"
    return f"画像抽出 {done}/{total}"


def substage_name(line: str) -> str | None:
    """Return the sub-stage label for a ``substage`` event, else None."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if payload.get("event") != "substage":
        return None
    name = payload.get("stage_name")
    return str(name) if name else None


def local_progress(line: str) -> float | None:
    """Return the 0..1 progress fraction carried by a line, clamped, or None."""
    try:
        payload = json.loads(line)
        value = float(payload.get("progress"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    return min(1.0, max(0.0, value))


def overall_progress(progress_start: float, progress_end: float, local: float) -> float:
    """Map a step-local 0..1 fraction onto the step's slice of the overall bar."""
    local = min(1.0, max(0.0, local))
    return progress_start + (progress_end - progress_start) * local
