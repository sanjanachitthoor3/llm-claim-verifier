from app.modules.claim_extraction.extractor import ClaimExtractor


def test_claim_extraction():

    extractor = ClaimExtractor()

    test_cases = [
        """
Albert Einstein was born in 1879 in Germany.
He won the Nobel Prize in 1921.
""",
        """
Tesla was founded in 2003.
Its headquarters are in Austin.
The company makes electric vehicles.
""",
        """
Python was created by Guido van Rossum in 1991.
Many developers love Python.
"""
    ]

    for text in test_cases:
        claims = extractor.extract_claims(text)

        print(f"\nExtracted Claims for text:\n{text}")

        for c in claims:
            print("-", c)


if __name__ == "__main__":
    test_claim_extraction()