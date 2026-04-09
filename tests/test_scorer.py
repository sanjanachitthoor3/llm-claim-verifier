from app.modules.scoring.scorer import compute_score


def test_scorer():
    # --- Case 1: mixed ---
    verdicts1 = ["SUPPORTED", "SUPPORTED", "CONTRADICTED"]
    result1 = compute_score(verdicts1)

    print("\nTest Case 1:", result1)
    assert result1["trust_score"] == 0.6667
    assert result1["supported"] == 2
    assert result1["contradicted"] == 1

    # --- Case 2: all supported ---
    verdicts2 = ["SUPPORTED", "SUPPORTED"]
    result2 = compute_score(verdicts2)

    print("Test Case 2:", result2)
    assert result2["trust_score"] == 1.0

    # --- Case 3: none supported ---
    verdicts3 = ["CONTRADICTED", "NOT ENOUGH INFO"]
    result3 = compute_score(verdicts3)

    print("Test Case 3:", result3)
    assert result3["trust_score"] == 0.0

    # --- Case 4: empty ---
    verdicts4 = []
    result4 = compute_score(verdicts4)

    print("Test Case 4:", result4)
    assert result4["total"] == 0


if __name__ == "__main__":
    test_scorer()