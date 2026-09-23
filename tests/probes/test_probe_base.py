# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Probe.probe() with an empty prompt set.

Covers NVIDIA/garak#2026: a probe whose prompt set is empty (e.g. emptied by
translation) must return no attempts and warn, not raise IndexError.
"""

import logging

import pytest

from garak import _config
import garak.probes.base


@pytest.fixture(autouse=True)
def _config_loaded():
    _config.load_base_config()


class _EmptyPromptProbe(garak.probes.base.Probe):
    """Minimal probe subclass with an empty prompt set."""

    prompts = []
    doc_uri = ""
    primary_detector = "always.Pass"
    tags = []


class _SinglePromptProbe(garak.probes.base.Probe):
    """Control probe: one prompt, should not hit the empty-set guard."""

    prompts = ["hello"]
    doc_uri = ""
    primary_detector = "always.Pass"
    tags = []


class _BlankPromptProbe(garak.probes.base.Probe):
    """Prompts that are empty/whitespace strings: not an empty *set*."""

    prompts = ["", "   "]
    doc_uri = ""
    primary_detector = "always.Pass"
    tags = []


def test_probe_empty_prompt_set_returns_no_attempts(caplog):
    """Empty prompt set -> warning naming the probe, returns []."""
    probe = _EmptyPromptProbe()
    with caplog.at_level(logging.WARNING):
        result = probe.probe(generator=None)
    assert result == [], "probe with empty prompt set must return no attempts"
    assert any(
        "empty prompt set" in record.message and probe.probename in record.message
        for record in caplog.records
    ), "expected a warning naming the probe"


def test_probe_nonempty_prompt_set_unaffected():
    """The guard must not swallow probes that do have prompts.

    This only checks we get past the empty-set guard (IndexError would mean
    the guard misfired); the generator is a stub and _execute_attempt is not
    reached without further stubbing, so a TypeError from downstream is
    acceptable here -- the point is no IndexError from prompts[0].
    """
    probe = _SinglePromptProbe()
    try:
        probe.probe(generator=None)
    except IndexError as e:
        pytest.fail(f"non-empty prompt set wrongly treated as empty: {e}")
    except Exception:
        pass  # downstream generator stubbing is out of scope for this guard


def test_probe_blank_prompts_not_treated_as_empty_set(caplog):
    """The guard fires on set emptiness, not on blank prompt content.

    ["", "   "] is a non-empty prompt set, so it must sail past the
    empty-set guard: no warning, no early return. Execution then proceeds
    into normal prompt preparation; the stub generator failing downstream
    (AttributeError from None.generate()) is out of scope, as above.
    """
    probe = _BlankPromptProbe()
    with caplog.at_level(logging.WARNING):
        try:
            probe.probe(generator=None)
        except AttributeError:
            pass  # None.generate() in _execute_attempt; guard already passed
    assert not any(
        "empty prompt set" in record.message for record in caplog.records
    ), "blank prompts must not trigger the empty-set warning"
