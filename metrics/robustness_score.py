def robustness_score(
    clean_conf,
    degraded_conf
):

    """
    Measure retained confidence
    under degradation.
    """

    if clean_conf == 0:
        return 0

    score = (
        degraded_conf / clean_conf
    ) * 100

    return score


if __name__ == "__main__":

    clean_conf = 0.91
    degraded_conf = 0.52

    score = robustness_score(
        clean_conf,
        degraded_conf
    )

    print(
        f"Robustness Score: "
        f"{score:.2f}%"
    )