"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from hf_gaia_agent.api_client import Question


# ---------------------------------------------------------------------------
# Temporary directories
# ---------------------------------------------------------------------------

def make_case_dir(name: str) -> Path:
    """Create a unique temporary directory under ``.test-artifacts/``."""
    root = Path(".test-artifacts") / f"{name}-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def case_dir(request: pytest.FixtureRequest) -> Path:
    """Per-test isolated directory named after the test function."""
    return make_case_dir(request.node.name)


@pytest.fixture()
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """Project-local replacement for pytest's built-in tmp_path on Windows."""
    return make_case_dir(f"tmp-{request.node.name}")


# ---------------------------------------------------------------------------
# Question factories
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_question() -> Question:
    return Question(task_id="test-1", question="What is 2+2?")


@pytest.fixture()
def question_with_file() -> Question:
    return Question(
        task_id="test-1",
        question="Review the file.",
        file_name="attachment.txt",
    )
