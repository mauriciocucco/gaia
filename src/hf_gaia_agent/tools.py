"""Tools used by the GAIA LangGraph agent."""

from __future__ import annotations

import base64
from ast import (
    Add,
    BinOp,
    Constant,
    Div,
    Expression,
    FloorDiv,
    Load,
    Mod,
    Mult,
    Name,
    Pow,
    Sub,
    UAdd,
    USub,
    UnaryOp,
    parse,
    walk,
)
import csv
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

logger = logging.getLogger(__name__)

import httpx
from bs4 import BeautifulSoup, Tag
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from pypdf import PdfReader


_HTTP_HEADERS = {
    "User-Agent": "GaiaAgent/1.0 (https://github.com/gaia-agent; educational research)",
}


def _truncate(value: str, *, max_chars: int = 12000) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 15] + "\n...[truncated]"


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )


def _read_csv(path: Path) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            rows.append(", ".join(row))
            if index >= 49:
                break
    return "\n".join(rows)


def _read_json(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return json.dumps(data, ensure_ascii=True, indent=2)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page in reader.pages[:20]:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def read_file_content(path: str) -> str:
    """Read a local task attachment and return plain text content."""
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"File not found: {candidate}")

    suffix = candidate.suffix.lower()
    if suffix in {".txt", ".md", ".log"}:
        content = candidate.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".csv":
        content = _read_csv(candidate)
    elif suffix == ".json":
        content = _read_json(candidate)
    elif suffix in {".html", ".htm"}:
        content = _html_to_text(candidate.read_text(encoding="utf-8", errors="replace"))
    elif suffix == ".pdf":
        content = _read_pdf(candidate)
    else:
        content = candidate.read_text(encoding="utf-8", errors="replace")
    return _truncate(content)


def _extract_text_section(full_text: str, start_marker: str, end_markers: list[str]) -> str:
    lower = full_text.lower()
    start_index = lower.find(start_marker.lower())
    if start_index == -1:
        return ""

    end_index = len(full_text)
    for marker in end_markers:
        candidate = lower.find(marker.lower(), start_index + len(start_marker))
        if candidate != -1:
            end_index = min(end_index, candidate)
    return full_text[start_index:end_index]


def _find_discography_table(html: str, section_name: str = "studio albums") -> Tag | None:
    """Find the first table following a section heading on a Wikipedia page."""
    soup = BeautifulSoup(html, "html.parser")
    for heading in soup.find_all(["h2", "h3", "h4"]):
        text = heading.get_text().strip().lower()
        if section_name not in text:
            continue
        table = heading.find_next("table")
        if table is not None:
            return table
    return None


def _extract_albums_from_table(
    table: Tag, start_year: int, end_year: int
) -> set[str]:
    """Extract album titles from a Wikipedia discography table within a year range."""
    albums: set[str] = set()
    rows = table.find_all("tr")
    if len(rows) < 2:
        return albums

    current_year: int | None = None
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        cell_texts = [c.get_text().strip() for c in cells]

        # Detect year from any cell
        row_year: int | None = None
        for ct in cell_texts:
            m = re.search(r"\b((?:19|20)\d{2})\b", ct)
            if m:
                row_year = int(m.group(1))
                break

        if row_year is not None:
            current_year = row_year
        else:
            row_year = current_year

        if row_year is None or not (start_year <= row_year <= end_year):
            continue

        # Title: first non-empty cell that is not just a 4-digit year
        title = ""
        for ct in cell_texts:
            cleaned = re.sub(r"\[.*?\]", "", ct).strip()
            if cleaned and not re.match(r"^\d{4}$", cleaned):
                title = cleaned
                break

        if title:
            albums.add(title.lower())

    return albums


def count_wikipedia_studio_album_count_for_artist(
    artist_name: str,
    start_year: int,
    end_year: int,
) -> int:
    artist_slug = quote(artist_name.replace(" ", "_"), safe="()_")
    urls = [
        f"https://en.wikipedia.org/wiki/{artist_slug}_discography",
        f"https://en.wikipedia.org/wiki/{artist_slug}",
    ]

    with httpx.Client(timeout=30.0, follow_redirects=True, headers=_HTTP_HEADERS) as client:
        for url in urls:
            try:
                response = client.get(url)
                response.raise_for_status()
            except httpx.HTTPError:
                continue

            table = _find_discography_table(response.text)
            if table is not None:
                albums = _extract_albums_from_table(table, start_year, end_year)
                if albums:
                    return len(albums)

    # Fallback: text-based extraction from main page
    url = f"https://en.wikipedia.org/wiki/{artist_slug}"
    with httpx.Client(timeout=30.0, follow_redirects=True, headers=_HTTP_HEADERS) as client:
        response = client.get(url)
        response.raise_for_status()

    page_text = _html_to_text(response.text)
    section_text = _extract_text_section(
        page_text,
        "Studio albums",
        ["EPs", "Live albums", "Compilation albums", "Filmography", "References"],
    )
    if not section_text:
        raise ValueError(f"Studio albums section not found for {artist_name}.")

    albums_fallback: set[tuple[int, str]] = set()
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line or line.lower() in {"studio albums", "year album details"}:
            continue
        match = re.match(r"^(?P<year>(?:19|20)\d{2})\s+(?P<title>.+)$", line)
        if not match:
            continue
        year = int(match.group("year"))
        if start_year <= year <= end_year:
            albums_fallback.add((year, match.group("title").strip().lower()))
    return len(albums_fallback)


def extract_youtube_video_id(url: str) -> str:
    """Extract a YouTube video ID from a standard URL."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/").split("/")[0]
    if "youtube.com" in host:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0]
        match = re.match(r"^/(?:embed|shorts)/([^/?#]+)", parsed.path)
        if match:
            return match.group(1)
    raise ValueError(f"Unsupported YouTube URL: {url}")


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return short snippets from the top results."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return "No results found."

    lines = []
    for index, item in enumerate(results, start=1):
        title = item.get("title") or "Untitled"
        href = item.get("href") or item.get("url") or ""
        body = item.get("body") or ""
        lines.append(f"{index}. {title}\nURL: {href}\nSnippet: {body}")
    return "\n\n".join(lines)


@tool
def fetch_url(url: str) -> str:
    """Fetch a URL and return text extracted from the response body."""
    with httpx.Client(timeout=30.0, follow_redirects=True, headers=_HTTP_HEADERS) as client:
        response = client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        text = json.dumps(response.json(), ensure_ascii=True, indent=2)
    elif "text/html" in content_type:
        text = _html_to_text(response.text)
    else:
        text = response.text
    return _truncate(text)


@tool
def get_youtube_transcript(url: str, languages_csv: str = "en,en-US") -> str:
    """Fetch a YouTube transcript for a video URL using prioritized languages."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import CouldNotRetrieveTranscript
    except ImportError as exc:
        raise RuntimeError(
            "youtube-transcript-api is not installed. Reinstall project dependencies."
        ) from exc

    video_id = extract_youtube_video_id(url)
    languages = [item.strip() for item in languages_csv.split(",") if item.strip()]
    client = YouTubeTranscriptApi()
    try:
        transcript = client.fetch(video_id, languages=languages or ["en"])
    except CouldNotRetrieveTranscript as exc:
        return f"Transcript unavailable for {video_id}: {exc}"

    lines = [
        f"[{snippet.start:.2f}s] {snippet.text}"
        for snippet in transcript
    ]
    return _truncate("\n".join(lines), max_chars=20000)


_FRAME_INTERVAL_SECONDS = 5
_MAX_FRAMES = 20


def _check_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"'{name}' not found on PATH. Install it first.")
    return path


def _download_video(url: str, output_dir: Path) -> Path:
    yt_dlp = _check_binary("yt-dlp")
    output_path = output_dir / "video.%(ext)s"
    cmd = [
        yt_dlp,
        "--no-playlist",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(output_path),
        "--quiet",
        "--no-warnings",
        url,
    ]
    subprocess.run(cmd, check=True, timeout=120)
    videos = list(output_dir.glob("video.*"))
    if not videos:
        raise FileNotFoundError("yt-dlp did not produce an output file.")
    return videos[0]


def _extract_frames(video_path: Path, output_dir: Path) -> list[Path]:
    ffmpeg = _check_binary("ffmpeg")
    pattern = output_dir / "frame_%04d.jpg"
    cmd = [
        ffmpeg,
        "-i", str(video_path),
        "-vf", f"fps=1/{_FRAME_INTERVAL_SECONDS},scale=768:-1",
        "-q:v", "2",
        "-frames:v", str(_MAX_FRAMES),
        str(pattern),
        "-y",
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True, timeout=120)
    frames = sorted(output_dir.glob("frame_*.jpg"))
    return frames


def _encode_frame_base64(frame_path: Path) -> str:
    return base64.b64encode(frame_path.read_bytes()).decode("ascii")


@tool
def analyze_youtube_video(url: str, question: str) -> str:
    """Download a YouTube video, extract frames, and analyze them with a vision model to answer the question."""
    from langchain_core.messages import HumanMessage as HM
    from langchain_openai import ChatOpenAI
    import os

    video_id = extract_youtube_video_id(url)

    with tempfile.TemporaryDirectory(prefix="gaia_video_") as tmp:
        tmp_path = Path(tmp)
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        try:
            video_file = _download_video(url, tmp_path)
        except Exception as exc:
            return f"Failed to download video {video_id}: {exc}"

        try:
            frames = _extract_frames(video_file, frames_dir)
        except Exception as exc:
            return f"Failed to extract frames from video {video_id}: {exc}"

        if not frames:
            return f"No frames extracted from video {video_id}."

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"These are {len(frames)} frames sampled every {_FRAME_INTERVAL_SECONDS} seconds "
                    f"from a YouTube video (ID: {video_id}).\n\n"
                    f"Question: {question}\n\n"
                    "Analyze the frames carefully and answer the question. "
                    "Be specific and precise."
                ),
            }
        ]
        for i, frame in enumerate(frames):
            timestamp = i * _FRAME_INTERVAL_SECONDS
            content.append({
                "type": "text",
                "text": f"[Frame at {timestamp}s]",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_encode_frame_base64(frame)}",
                    "detail": "low",
                },
            })

        provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
        model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
        kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": 0,
            "timeout": 120,
        }
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
        elif provider == "huggingface":
            token = os.getenv("HF_TOKEN")
            base_url = os.getenv("OPENAI_BASE_URL", "https://router.huggingface.co/v1")
            if token:
                kwargs["api_key"] = token
            kwargs["base_url"] = base_url

        vision_model = ChatOpenAI(**kwargs)
        response = vision_model.invoke([HM(content=content)])
        return _truncate(str(response.content), max_chars=8000)


@tool
def count_wikipedia_studio_albums(
    artist_name: str,
    start_year: int,
    end_year: int,
) -> str:
    """Count studio albums listed on the artist's English Wikipedia page between two years inclusive."""
    return str(
        count_wikipedia_studio_album_count_for_artist(
            artist_name=artist_name,
            start_year=start_year,
            end_year=end_year,
        )
    )


@tool
def read_local_file(path: str) -> str:
    """Read a local text, CSV, JSON, HTML, or PDF file."""
    return read_file_content(path)


ALLOWED_BINOPS = {
    Add: lambda left, right: left + right,
    Sub: lambda left, right: left - right,
    Mult: lambda left, right: left * right,
    Div: lambda left, right: left / right,
    FloorDiv: lambda left, right: left // right,
    Mod: lambda left, right: left % right,
    Pow: lambda left, right: left**right,
}
ALLOWED_UNARYOPS = {
    UAdd: lambda value: +value,
    USub: lambda value: -value,
}


def _safe_eval(node: Any) -> float | int:
    if isinstance(node, Expression):
        return _safe_eval(node.body)
    if isinstance(node, Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, BinOp):
        operator = ALLOWED_BINOPS.get(type(node.op))
        if not operator:
            raise ValueError("Unsupported operator.")
        return operator(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, UnaryOp):
        operator = ALLOWED_UNARYOPS.get(type(node.op))
        if not operator:
            raise ValueError("Unsupported unary operator.")
        return operator(_safe_eval(node.operand))
    if isinstance(node, Name) and node.id in {"pi", "e"}:
        constants = {"pi": 3.141592653589793, "e": 2.718281828459045}
        return constants[node.id]
    raise ValueError("Unsafe expression.")


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    tree = parse(expression, mode="eval")
    for node in walk(tree):
        if type(node) not in {
            Expression,
            BinOp,
            UnaryOp,
            Constant,
            Load,
            Add,
            Sub,
            Mult,
            Div,
            FloorDiv,
            Mod,
            Pow,
            UAdd,
            USub,
            Name,
        }:
            raise ValueError("Unsafe expression.")

    result = _safe_eval(tree)
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


def build_tools() -> list[Any]:
    return [
        web_search,
        fetch_url,
        get_youtube_transcript,
        analyze_youtube_video,
        count_wikipedia_studio_albums,
        read_local_file,
        calculate,
    ]
