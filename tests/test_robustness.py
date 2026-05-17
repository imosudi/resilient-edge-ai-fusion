from metrics.robustness_score import robustness_score


def test_perfect_robustness():

    score = robustness_score(1.0, 1.0)

    assert score == 100


def test_partial_robustness():

    score = robustness_score(1.0, 0.5)

    assert score == 50


def test_zero_clean_confidence():

    score = robustness_score(0, 0.5)

    assert score == 0