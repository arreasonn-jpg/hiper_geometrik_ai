from hga.evaluation.statistics import adjust_p_values, cliffs_delta, paired_power_plan


def test_correction_methods_are_monotone_and_keep_raw_p_values():
    values = {"a": 0.01, "b": 0.04, "c": 0.2}
    for method in ("bonferroni", "holm", "fdr_bh"):
        result = adjust_p_values(values, method)
        assert set(result) == set(values)
        assert result["a"]["p_value"] == 0.01
        assert all(row["adjusted_p_value"] >= row["p_value"] for row in result.values())


def test_cliffs_delta_reports_direction_and_pairing_limit():
    result = cliffs_delta([4, 5, 6], [1, 2, 3])
    assert result["value"] == 1.0
    assert result["magnitude"] == "large"
    assert "pairing" in result["limitation"]


def test_paired_power_plan_is_explicitly_a_planning_approximation():
    plan = paired_power_plan(0.5)
    assert plan["minimum_pairs"] > 5
    assert plan["minimum_two_sided_permutation_p"] < 0.05
    assert "not post-hoc" in plan["limitation"]
