"""Smoke tests for the runnable examples.

Examples are documentation that executes, which means they rot exactly like
documentation unless something runs them.  These assert the two key-free
examples still run and still demonstrate what their prose claims — not just
that they exit 0, which would pass on an example that silently stopped
finding anything.

``02_gate_live.py`` is not covered here: it costs money and needs a key.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _run(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, str(EXAMPLES / name), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def run_example(name: str, *args: str) -> str:
    result = _run(name, *args)
    assert result.returncode == 0, f"{name} exited {result.returncode}\n{result.stderr}"
    return result.stdout


def run_example_failing(name: str, *args: str) -> str:
    """Run an example expected to reject its arguments; return stderr."""
    result = _run(name, *args)
    assert result.returncode != 0, f"{name} unexpectedly succeeded\n{result.stdout}"
    return result.stderr


@pytest.fixture(scope="module")
def gate_output() -> str:
    return run_example("01_gate_offline.py")


@pytest.fixture(scope="module")
def fidelity_output() -> str:
    return run_example("03_fidelity_elaws.py", "--offline")


class TestGateOffline:
    def test_strips_the_fabricated_sentence_and_ships_the_rest(self, gate_output: str) -> None:
        assert "[STRIP ] The provider will then pay you $100,000." in gate_output
        assert "grounded = True" in gate_output

    def test_abstains_when_too_little_survives(self, gate_output: str) -> None:
        assert "(nothing — the caller must abstain)" in gate_output

    def test_citation_aware_split_beats_naive_split(self, gate_output: str) -> None:
        # The naive splitter shatters the citation; the real one must not.
        assert "· 892 s." in gate_output, "naive split no longer demonstrates the failure"
        assert "· Report the defect under Reg. 892 s. 4.4(2)." in gate_output

    def test_fails_closed_when_the_judge_raises(self, gate_output: str) -> None:
        assert "verifier failed (connection reset by peer)" in gate_output
        # Both sentences are genuinely supported; the gate suppresses them anyway.
        assert gate_output.count("[STRIP ] You must report the defect in writing") == 1


class TestFidelityOffline:
    def test_faithful_corpus_verifies_clean(self, fidelity_output: str) -> None:
        assert "flagged: 0   findings: 0" in fidelity_output

    def test_each_injected_defect_is_caught(self, fidelity_output: str) -> None:
        # One verdict per layer: L3 both directions, plus L1/L2 on the edit.
        assert "[missing_in_corpus] s.9" in fidelity_output
        assert "[extra_in_corpus] s.99" in fidelity_output
        assert "[text_mismatch]" in fidelity_output
        assert "[containment_fail]" in fidelity_output

    def test_the_edit_lands_in_a_section_body_not_the_label(self, fidelity_output: str) -> None:
        # A label-only edit would demonstrate a weaker failure than amendment
        # drift buried in the prose, and would not trip L2 containment.
        assert "changed a cross-reference in s.2 from 14 to 15" in fidelity_output


class TestFidelityUrlArgument:
    """The --url guards. Both reject before any network call is attempted."""

    def test_rejects_a_url_the_elaws_adapter_cannot_map(self) -> None:
        stderr = run_example_failing("03_fidelity_elaws.py", "--url", "https://example.com/foo")
        assert "Not a recognized e-Laws document URL" in stderr
        # The message has to show the shape that does work, or the user is stuck.
        assert "ontario.ca/laws/regulation/210242" in stderr

    def test_rejects_offline_combined_with_url(self) -> None:
        stderr = run_example_failing(
            "03_fidelity_elaws.py",
            "--url",
            "https://www.ontario.ca/laws/statute/90o31",
            "--offline",
        )
        assert "there is none for an arbitrary --url" in stderr
