from pathlib import Path
from uuid import uuid4

import httpx

import hf_gaia_agent.tools as tools_module
from hf_gaia_agent.tools import _transcribe_audio, read_file_content


def _case_dir(name: str) -> Path:
    root = Path(".test-artifacts") / f"{name}-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_read_file_content_transcribes_mp3(monkeypatch) -> None:
    case_dir = _case_dir("audio-read")
    audio_file = case_dir / "clip.mp3"
    audio_file.write_bytes(b"fake-mp3")

    monkeypatch.setattr(tools_module, "_transcribe_audio", lambda path: f"transcript for {path.name}")

    result = read_file_content(str(audio_file))

    assert result == "transcript for clip.mp3"


def test_transcribe_audio_posts_to_audio_transcriptions_endpoint(monkeypatch) -> None:
    case_dir = _case_dir("audio-transcribe")
    audio_file = case_dir / "clip.mp3"
    audio_file.write_bytes(b"fake-mp3")

    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("AUDIO_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
    monkeypatch.delenv("AUDIO_TRANSCRIPTION_LANGUAGE", raising=False)
    monkeypatch.delenv("AUDIO_TRANSCRIPTION_PROMPT", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["Authorization"] == "Bearer test-key"
        body = request.content
        assert b'name="model"' in body
        assert b"gpt-4o-mini-transcribe" in body
        assert b'name="file"; filename="clip.mp3"' in body
        return httpx.Response(200, json={"text": "transcribed audio"})

    real_client = httpx.Client

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.client = real_client(
                transport=httpx.MockTransport(handler),
                base_url="https://example.test",
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout"),
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.client.close()

        def post(self, *args, **kwargs):
            return self.client.post(*args, **kwargs)

    monkeypatch.setattr(tools_module.httpx, "Client", FakeClient)

    result = _transcribe_audio(audio_file)

    assert result == "transcribed audio"
