"""Regression tests for NVIDIA/garak#2155.

Buff._derive_new_attempt used to hand the derived attempt the source
attempt's own notes and detector_results dicts, so every sibling attempt
ended up reporting the last sibling's detector scores, and the source
attempt was polluted with buff metadata.
"""

import pytest

from garak import _plugins, attempt
from garak.buffs.base import Buff


@pytest.fixture()
def buff():
    return _plugins.load_plugin("buffs.lowercase.Lowercase")


@pytest.fixture()
def source_attempt():
    a = attempt.Attempt(
        prompt=attempt.Message("HELLO", lang="en"), probe_classname="test.Blank"
    )
    a.notes["preexisting"] = "kept"
    a.detector_results["always.Fail"] = [1.0]
    return a


def test_derive_copies_notes_and_detector_results(buff, source_attempt):
    derived = buff._derive_new_attempt(source_attempt)
    assert derived.notes is not source_attempt.notes, (
        "derived attempt must own its notes dict"
    )
    assert derived.detector_results is not source_attempt.detector_results, (
        "derived attempt must own its detector_results dict"
    )
    # pre-existing entries are carried over by value
    assert derived.notes["preexisting"] == "kept", "notes content must be copied"
    assert derived.detector_results["always.Fail"] == [1.0], (
        "detector_results content must be copied"
    )


def test_derive_does_not_pollute_source(buff, source_attempt):
    buff._derive_new_attempt(source_attempt)
    for key in ("buff_creator", "buff_source_attempt_uuid", "buff_source_seq"):
        assert key not in source_attempt.notes, (
            f"source attempt notes must not gain {key}"
        )


def test_sibling_detector_scores_stay_independent(buff, source_attempt):
    first = buff._derive_new_attempt(source_attempt)
    second = buff._derive_new_attempt(source_attempt)
    first.detector_results["always.Fail"] = [1.0]
    second.detector_results["always.Fail"] = [0.0]
    assert first.detector_results["always.Fail"] == [1.0], (
        "writing one sibling's score must not change another sibling's"
    )
    assert second.detector_results["always.Fail"] == [0.0]
    assert source_attempt.detector_results["always.Fail"] == [1.0], (
        "writing a derived attempt's score must not change the source's"
    )


def test_derived_attempt_carries_buff_metadata(buff, source_attempt):
    derived = buff._derive_new_attempt(source_attempt)
    assert derived.notes["buff_creator"] == "Lowercase", (
        "derived attempt must record which buff created it"
    )
    assert derived.notes["buff_source_attempt_uuid"] == str(source_attempt.uuid)
    assert derived.notes["buff_source_seq"] == source_attempt.seq


def test_buff_hook_siblings_do_not_share_state():
    # end-to-end through the probe buff hook, mirroring the issue report;
    # needs no API key: lowercase buff + test.Blank probe only
    from garak import _config

    _config.load_base_config()
    _config.plugins.buffs_include_original_prompt = True
    _config.buffmanager.buffs = [_plugins.load_plugin("buffs.lowercase.Lowercase")]
    probe = _plugins.load_plugin("probes.test.Blank")
    original = attempt.Attempt(
        prompt=attempt.Message("HELLO", lang="en"), probe_classname="test.Blank"
    )
    attempts = probe._buff_hook([original])
    assert len(attempts) == 2, "expected original + buffed attempt"
    orig, buffed = attempts
    assert orig.notes is not buffed.notes
    assert orig.detector_results is not buffed.detector_results
    orig.detector_results["always.Fail"] = [1.0]
    buffed.detector_results["always.Fail"] = [0.0]
    assert orig.detector_results["always.Fail"] == [1.0], (
        "harness-style per-attempt score writes must not leak across siblings"
    )


def test_chained_derivation_grandchild_isolated(buff, source_attempt):
    # buffs stack, so derivation-of-a-derivation must isolate too
    child = buff._derive_new_attempt(source_attempt)
    grandchild = buff._derive_new_attempt(child)
    child.detector_results["always.Fail"] = [0.5]
    grandchild.detector_results["always.Fail"] = [0.0]
    assert child.detector_results["always.Fail"] == [0.5], (
        "child score must survive a grandchild write"
    )
    assert grandchild.detector_results["always.Fail"] == [0.0]
    assert source_attempt.detector_results["always.Fail"] == [1.0], (
        "source score must survive chained derivation writes"
    )
    assert "buff_creator" not in source_attempt.notes, (
        "chained derivation must not pollute the source attempt"
    )


def test_nested_notes_list_isolated_after_fresh_assign(buff, source_attempt):
    # mirrors detectors/packagehallucination.py: fresh list assigned to
    # notes before appending, so the shallow copy is enough here
    source_attempt.notes["hallucinated_python_packages"] = ["legit-pkg"]
    derived = buff._derive_new_attempt(source_attempt)
    derived.notes["hallucinated_python_packages"] = []
    derived.notes["hallucinated_python_packages"].append(["evil-pkg"])
    assert source_attempt.notes["hallucinated_python_packages"] == ["legit-pkg"], (
        "fresh-assign-then-append on a derived attempt must not touch the source list"
    )


def test_generation_outputs_do_not_leak_into_source_conversations(
    buff, source_attempt
):
    # the prompt setter deep-copies, so per-attempt output writes
    # (which append turns to conversations) stay per-attempt
    derived = buff._derive_new_attempt(source_attempt)
    source_turns_before = len(source_attempt.conversations[0].turns)
    derived.outputs = ["a generated response"]
    assert len(source_attempt.conversations[0].turns) == source_turns_before, (
        "writing outputs on a derived attempt must not append to source conversations"
    )
