from app.modules.wikipedia.wiki_fetcher import WikiFetcher


def test_wikipedia_fetch():

    fetcher = WikiFetcher()

    test_claims = [
        "Albert Einstein was born in 1879",
        "Tesla was founded in 2003",
        "Python was created by Guido van Rossum in 1991"
    ]

    for claim in test_claims:

        print("\n----------------------------")
        print("Claim:", claim)

        title = fetcher.search_page(claim)
        print("Page title:", title)

        content = fetcher.fetch_page(title)

        if content:
            print("Content length:", len(content))
            print("Preview:", content[:300])
        else:
            print("No article content found.")

        # Test full pipeline method
        article = fetcher.get_article_for_claim(claim)

        if article:
            print("Pipeline OK (get_article_for_claim works)")
        else:
            print("Pipeline FAILED")


if __name__ == "__main__":
    test_wikipedia_fetch()