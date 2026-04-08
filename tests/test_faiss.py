from app.modules.retrieval.faiss_search import FAISSRetriever


def test_faiss_retriever():
    sentences = [
        "Albert Einstein was born on 14 March 1879 in Ulm, Germany.",
        "He developed the theory of general relativity.",
        "Einstein received the Nobel Prize in Physics in 1921.",
        "Marie Curie was the first woman to win a Nobel Prize.",
        "The speed of light in a vacuum is approximately 299,792 km/s.",
    ]

    claim = "Albert Einstein was born in 1879"

    retriever = FAISSRetriever()
    retriever.build_index(sentences)

    results = retriever.search(claim, k=3)

    # ---- PRINT OUTPUT ----
    print("\nCLAIM:")
    print(claim)

    print("\nTOP RESULTS:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r}")

    # ---- BASIC CHECK ----
    assert any("1879" in r or "born" in r for r in results)
    

if __name__ == "__main__":
    test_faiss_retriever()