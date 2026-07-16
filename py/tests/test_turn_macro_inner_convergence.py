from profiling.analysis.turn_macro_inner_convergence import (
    _compare,
    _is_ordered_subsequence,
    _material_plans,
    _plans,
)


def test_plans_extracts_complete_action_sequences() -> None:
    response = {
        "_profile": {
            "root_turn_plans": [
                {"action_signatures": ["MOVE:a", "ATTACK:b", "END_TURN"]},
                {"action_signatures": ["RESEARCH:c", "END_TURN"]},
            ]
        }
    }

    assert _plans(response, plan_count=8, comparison_depth=0) == [
        ("MOVE:a", "ATTACK:b", "END_TURN"),
        ("RESEARCH:c", "END_TURN"),
    ]
    assert _plans(response, plan_count=1, comparison_depth=2) == [
        ("MOVE:a", "ATTACK:b"),
    ]


def test_compare_does_not_treat_shared_first_action_as_plan_match() -> None:
    candidate = [("MOVE:a", "ATTACK:b"), ("RESEARCH:c", "END_TURN")]
    reference = [("MOVE:a", "RECOVER:d"), ("RESEARCH:c", "END_TURN")]

    metrics = _compare(candidate, reference)

    assert metrics["ranked_plan_match_rate"] == 0.5
    assert metrics["set_recall"] == 0.5
    assert metrics["set_precision"] == 0.5
    assert metrics["set_jaccard"] == 1 / 3
    assert metrics["exact_ranked_plans"] == 0
    assert metrics["exact_plan_set"] == 0


def test_material_plans_use_unordered_continuation_diagnostic() -> None:
    response = {
        "_profile": {
            "root_turn_plans": [
                {
                    "action_signatures": ["MOVE:a", "END_TURN", "ATTACK:b"],
                }
            ]
        }
    }

    assert _material_plans(response, plan_count=8) == [
        ("MOVE:a", "ATTACK:b", "END_TURN"),
    ]


def test_compare_distinguishes_set_stability_from_rank_stability() -> None:
    candidate = [("A", "END"), ("B", "END")]
    reference = list(reversed(candidate))

    metrics = _compare(candidate, reference)

    assert metrics["set_jaccard"] == 1.0
    assert metrics["exact_plan_set"] == 1
    assert metrics["ranked_plan_match_rate"] == 0.0
    assert metrics["exact_ranked_plans"] == 0


def test_sequence_containment_alignment_is_length_invariant_but_order_sensitive() -> None:
    assert _is_ordered_subsequence(("A", "C"), ("A", "B", "C"))
    assert not _is_ordered_subsequence(("C", "A"), ("A", "B", "C"))
    assert _compare([("B",)], [("A", "B")], "exact_sequence")["containment_alignment_rate"] == 0.0

    metrics = _compare(
        [("A", "C"), ("X", "Y")],
        [("A", "B", "C"), ("X", "Z", "Y")],
        "exact_sequence",
    )

    assert metrics["set_jaccard"] == 0.0
    assert metrics["containment_alignment_rate"] == 1.0
    assert metrics["ranked_containment_rate"] == 1.0


def test_material_containment_requires_same_first_action_and_matches_one_to_one() -> None:
    candidate = [("FIRST", "A"), ("FIRST", "A")]
    reference = [("FIRST", "A", "B"), ("OTHER", "A", "B")]

    metrics = _compare(candidate, reference, "material")

    assert metrics["containment_alignment_rate"] == 0.5
    assert metrics["containment_candidate_rate"] == 0.5
    assert metrics["containment_reference_rate"] == 0.5
