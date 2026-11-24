# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""Revisit scores and readable segment metadata for a single object."""

from ..revisit_protocol import MAX_EXPECTED_REVISITS


def revisit_score_fields(scores, coverages, *, expected_revisits):
    if expected_revisits is None:
        return {}
    return {
        f"revisit_{index + 1}_{name}": float(values[index]) if index < len(values) else 0.0
        for index in range(MAX_EXPECTED_REVISITS)
        for name, values in (("score", scores), ("coverage", coverages))
    }


def official_revisit_score(scores, *, expected_revisits):
    count = len(scores) if expected_revisits is None else expected_revisits
    return float(sum(scores[:count]) / count) if count else 0.0


def revisit_metadata(phase, scores, coverages, counts, *, count_key, expected_revisits):
    count = len(scores) if expected_revisits is None else MAX_EXPECTED_REVISITS
    fields = {
        "indices": (phase.revisit_indices, ()),
        "slot": (phase.revisit_slots, ()),
        "nominal_boundary": (phase.revisit_nominal_boundaries, None),
        "search_window": (phase.revisit_search_windows, ()),
        "window_source": (phase.revisit_sources, "missing"),
        "departure_evidence": (phase.revisit_departure_evidence, False),
        "score": (scores, 0.0),
        "coverage": (coverages, 0.0),
        count_key: (counts, 0),
    }
    output = []
    for index in range(count):
        row = {"revisit_id": index + 1}
        for name, (values, default) in fields.items():
            value = values[index] if index < len(values) else default
            row[name] = list(value) if isinstance(value, tuple) else value
        indices = row["indices"]
        row["actual_window"] = [indices[0], indices[-1] + 1] if indices else []
        output.append(row)
    return output
