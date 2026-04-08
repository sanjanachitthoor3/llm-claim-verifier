from app.modules.verification.nli_verifier import NLIVerifier, SUPPORTED, CONTRADICTED, NOT_ENOUGH_INFO


def test_nli_verifier():
    verifier = NLIVerifier()

    # --- SUPPORTED ---
    claim1 = "Albert Einstein was born in 1879"
    evidence1 = ["Albert Einstein was born on 14 March 1879 in Germany."]
    result1 = verifier.verify(claim1, evidence1)

    print("\nSUPPORTED TEST:", result1)
    assert result1 == SUPPORTED

    # --- WRONG INFO ---
    claim2 = "Albert Einstein was born in 2000"
    evidence2 = ["Albert Einstein was born in 1879."]
    result2 = verifier.verify(claim2, evidence2)

    print("CONTRADICTED TEST:", result2)
    assert result2 == CONTRADICTED

    # --- NOT ENOUGH INFO ---
    claim3 = "Einstein was a great singer"
    evidence3 = ["Einstein developed the theory of relativity."]
    result3 = verifier.verify(claim3, evidence3)

    print("NEUTRAL TEST:", result3)
    assert result3 == NOT_ENOUGH_INFO


if __name__ == "__main__":
    test_nli_verifier()