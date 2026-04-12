"""Helpers for structured analysis of YouTube video frames."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any


_DENSE_FRAME_INTERVAL_SECONDS = 1
_DENSE_WINDOW_RADIUS_SECONDS = 2
_MAX_DENSE_WINDOWS = 3
_COUNTING_VISUAL_CUES = (
    "how many",
    "highest number",
    "maximum number",
    "minimum number",
    "lowest number",
    "simultaneously",
    "at the same time",
    "on camera simultaneously",
)


def encode_frame_base64(frame_path: Path) -> str:
    return base64.b64encode(frame_path.read_bytes()).decode("ascii")


def is_counting_visual_question(question: str) -> bool:
    lowered = question.strip().lower()
    return any(cue in lowered for cue in _COUNTING_VISUAL_CUES)


def extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = [str(text or "").strip()]
    fenced = re.findall(
        r"```(?:json)?\s*(.*?)\s*```",
        str(text or ""),
        flags=re.DOTALL | re.IGNORECASE,
    )
    candidates.extend(item.strip() for item in fenced if item.strip())

    match = re.search(r"\{.*\}", str(text or ""), flags=re.DOTALL)
    if match:
        candidates.append(match.group(0).strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _parse_payload_timestamp(value: Any) -> int | None:
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(round(value)))
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return max(0, int(round(float(candidate))))
        except ValueError:
            return None
    return None


def _parse_visual_count(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _extract_frame_counts_from_payload(payload: dict[str, Any]) -> list[tuple[int, int]]:
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return []

    counts: list[tuple[int, int]] = []
    for index, item in enumerate(frames):
        if not isinstance(item, dict):
            continue
        timestamp = _parse_payload_timestamp(
            item.get("timestamp_s", item.get("timestamp", item.get("time_s")))
        )
        if timestamp is None:
            timestamp = index
        count = None
        for key in ("count", "species_count", "visible_count"):
            count = _parse_visual_count(item.get(key))
            if count is not None:
                break
        if count is None:
            species = item.get("species")
            if isinstance(species, list):
                count = len([name for name in species if str(name).strip()])
        if count is None:
            continue
        counts.append((timestamp, count))
    return counts


def extract_max_count_from_payload(payload: dict[str, Any]) -> int | None:
    frame_counts = _extract_frame_counts_from_payload(payload)
    if frame_counts:
        return max(count for _timestamp, count in frame_counts)

    for key in (
        "max_count",
        "max_species_count",
        "highest_count",
        "highest_species_count",
    ):
        value = payload.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def select_dense_timestamps_from_payload(payload: dict[str, Any]) -> list[int]:
    frame_counts = _extract_frame_counts_from_payload(payload)
    if not frame_counts:
        return []

    ranked = sorted(frame_counts, key=lambda item: (-item[1], item[0]))
    if not ranked or ranked[0][1] <= 0:
        return []

    timestamps: list[int] = []
    for timestamp, _count in ranked[:_MAX_DENSE_WINDOWS]:
        for offset in range(
            -_DENSE_WINDOW_RADIUS_SECONDS,
            _DENSE_WINDOW_RADIUS_SECONDS + 1,
            _DENSE_FRAME_INTERVAL_SECONDS,
        ):
            timestamps.append(max(0, timestamp + offset))
    return sorted(dict.fromkeys(timestamps))


def build_video_analysis_prompt(
    *,
    question: str,
    video_id: str,
    frame_count: int,
    frame_interval_seconds: int,
    counting_mode: bool,
    prompt_mode: str = "coarse",
) -> str:
    pass_description = (
        f"sampled every {frame_interval_seconds} seconds"
        if prompt_mode == "coarse"
        else f"sampled around promising moments at roughly {frame_interval_seconds}-second spacing"
    )
    if counting_mode:
        phase_instruction = (
            "This is the initial coarse scan. Identify the frames most likely to contain the peak simultaneous count."
            if prompt_mode == "coarse"
            else "This is a verification pass around previously promising moments. Re-evaluate carefully and return per-frame counts."
        )
        return (
            f"These are {frame_count} frames {pass_description} "
            f"from a YouTube video (ID: {video_id}).\n\n"
            f"Question: {question}\n\n"
            f"{phase_instruction}\n"
            "Analyze each frame independently and count only what the question asks for.\n"
            "For species questions, count biological species, not individuals, ages, or sexes.\n"
            "Do not count chicks and adults of the same species separately.\n"
            "Look carefully for small or distant subjects in the background before deciding.\n"
            "Include a short species list for each frame when you can identify the visible species.\n"
            "Return JSON only using this schema:\n"
            '{"frames":[{"timestamp_s":0,"species":["species name"],"count":0,"notes":"short note"}],"max_count":0}\n'
            "Use integer counts."
        )
    return (
        f"These are {frame_count} frames {pass_description} "
        f"from a YouTube video (ID: {video_id}).\n\n"
        f"Question: {question}\n\n"
        "Analyze the frames carefully and answer the question. "
        "Be specific and precise."
    )


def prepend_audio_transcript(prompt_text: str, audio_transcript: str) -> str:
    if not audio_transcript:
        return prompt_text
    return (
        f"[Audio transcript]\n{audio_transcript}\n\n"
        f"[Visual frames below]\n{prompt_text}"
    )


def build_video_message_content(
    *,
    prompt_text: str,
    frame_items: list[tuple[int, Path]],
    detail: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for timestamp, frame in frame_items:
        content.append({"type": "text", "text": f"[Frame at {timestamp}s]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_frame_base64(frame)}",
                    "detail": detail,
                },
            }
        )
    return content
