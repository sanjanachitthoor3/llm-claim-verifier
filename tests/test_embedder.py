from app.modules.embeddings.embedder import get_top_k_sentences

def test_get_top_k_sentences():
    claim = "Tesla was founded in 2003."
    sentences = [
        "Tesla, Inc. ( TEZ-lə or TESS-lə) is an American multinational automotive and clean energy company.",
        "Headquartered in Austin, Texas, it designs, manufactures, and sells battery electric vehicles (BEVs), stationary battery energy storage devices from home to grid-scale, solar panels and solar shingles, and related products and services.",
        "The company was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning as Tesla Motors.",
        "Its name is a tribute to the inventor and electrical engineer Nikola Tesla.",
        "In February 2004, Elon Musk led Tesla's first funding round and became the company's chairman, subsequently claiming to be a co-founder; in 2008, he was named chief executive officer."
    ]
    top_k = get_top_k_sentences(claim, sentences, k=2)

    # ---- PRINT OUTPUT (for manual inspection) ----
    print("\nCLAIM:")
    print(claim)

    print("\nTOP RESULTS:")
    for i, s in enumerate(top_k, 1):
        print(f"{i}. {s}")

    # ---- BASIC VALIDATION ----
    # At least one result should relate to founding/year
    assert any("2003" in s or "incorporated" in s for s in top_k) 

if __name__ == "__main__":
    test_get_top_k_sentences()
