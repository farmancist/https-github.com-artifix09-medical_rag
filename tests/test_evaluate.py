from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate import (
    _cases_for_source,
    eval_matching,
    first_hit_rank,
    run_retrieval,
    source_manifest_entries,
    summarise,
)


def _hit(text: str) -> SimpleNamespace:
    return SimpleNamespace(chunk=SimpleNamespace(text=text))


def test_first_hit_rank_matches_expect_all_without_expect_any() -> None:
    case = {"expect_all": ["paracetamol", "240 mg"]}

    assert first_hit_rank([_hit("Paracetamol dose: 240 mg")], case) == 1
    assert eval_matching("Paracetamol dose: 240 mg", case)


def test_first_hit_rank_requires_all_terms_and_one_any_term() -> None:
    hits = [
        _hit("amoxicillin for pneumonia, given twice daily"),
        _hit("amoxicillin for pneumonia, given 3 times a day"),
    ]
    case = {
        "expect_all": ["amoxicillin", "pneumonia"],
        "expect_any": ["3 times a day", "500 mg"],
    }

    assert first_hit_rank(hits, case) == 2


def test_first_hit_rank_normalises_dash_and_nonbreaking_space() -> None:
    case = {"expect_any": ["5-10 mg"]}

    assert first_hit_rank([_hit("Dose: 5\u201310\u00a0mg")], case) == 1


def test_first_hit_rank_does_not_match_cases_without_expected_terms() -> None:
    assert first_hit_rank([_hit("any text")], {}) is None


def test_source_filter_selects_tagged_cases_and_manifest_entries(tmp_path: Path) -> None:
    cases = [
        {"id": "ohcm", "source": "OHCM", "query": "CURB-65"},
        {"id": "bnf", "source": "BNF", "query": "amoxicillin"},
        {"id": "combined", "source": "BNF / OHCM", "query": "combined evidence"},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "ohcm-id": {
                    "title": "Oxford Handbook of Clinical Medicine",
                    "indexed_points": 12,
                },
                "bnf-id": {
                    "title": "British National Formulary 85",
                    "indexed_points": 34,
                },
            }
        ),
        encoding="utf-8",
    )

    assert [case["id"] for case in _cases_for_source(cases, "OHCM")] == ["ohcm"]
    assert list(source_manifest_entries(manifest_path, "OHCM")) == ["ohcm-id"]


def test_filtered_evaluation_reports_missing_source_chunks() -> None:
    class NoSearchPipeline:
        def search(self, *args, **kwargs):
            raise AssertionError("search should not run when the source has no indexed docs")

    rows = run_retrieval(
        NoSearchPipeline(),
        [{"id": "curb65", "query": "CURB-65", "expect_any": ["CURB-65"]}],
        10,
        doc_ids=[],
        source="OHCM",
    )

    assert rows[0]["note"] == "no indexed OHCM chunks"
    assert rows[0]["no_source_chunks"] is True
    assert summarise(rows)["no_source_chunks"] == 1


def test_summary_includes_metrics_by_category() -> None:
    rows = [
        {"category": "dose", "hit": True, "rank": 2, "ms": 10, "error": ""},
        {"category": "safety", "hit": False, "rank": None, "ms": 20, "error": ""},
    ]

    summary = summarise(rows)

    assert summary["by_category"]["dose"] == {
        "questions": 1,
        "hit_rate": 1.0,
        "mrr": 0.5,
    }
    assert summary["by_category"]["safety"]["hit_rate"] == 0.0
