from app.modules.claim_extraction.extractor import ClaimExtractor
from app.modules.wikipedia.wiki_fetcher import WikiFetcher
from app.utils.text_preprocessing import split_into_sentences;

# This test simulates a manual pipeline where we take a response from an LLM, extract claims, and print them out, fetch wikipedia pages for each claim, and split the wikipedia content into sentences.


def test_manual_pipeline1():
    #llm response string
    llm_response = "Albert Einstein was a physicist. He developed the theory of relativity."
    
    extractor= ClaimExtractor()
    claims = extractor.extract_claims(llm_response)
    print(f"\nExtracted Claims for text:\n{llm_response}")
    for c in claims:
        print("-", c)
        fetcher= WikiFetcher()
        article = fetcher.get_article_for_claim(c)
        if article:
            sentences = split_into_sentences(article)
            print(f"\nWikipedia Sentences for claim:\n{c}")
            for s in sentences[:5]:  # print first 5 sentences for brevity
                print("-", s)
        else:
            print("No Wikipedia article found.")


def test_ambiguous_entity(): #gave nicola tesla article instead of tesla company -FIXED
    llm_response = "Tesla was founded in 2003."

    extractor= ClaimExtractor()
    claims = extractor.extract_claims(llm_response)
    print(f"\nExtracted Claims for text:\n{llm_response}")
    for c in claims:
        print("-", c)
        fetcher= WikiFetcher()
        article = fetcher.get_article_for_claim(c)
        if article:
            sentences = split_into_sentences(article)
            print(f"\nWikipedia Sentences for claim:\n{c}")
            for s in sentences[:5]:  # print first 5 sentences for brevity
                print("-", s)
        else:
            print("No Wikipedia article found.")

def test_multi_entity(): #only gave article for Einstein, not Tesla, and didn't split into 2 claims
    llm_response = "Einstein and Tesla were scientists. He developed relativity."

    extractor= ClaimExtractor()
    claims = extractor.extract_claims(llm_response)
    print(f"\nExtracted Claims for text:\n{llm_response}")
    for c in claims:
        print("-", c)
        fetcher= WikiFetcher()
        article = fetcher.get_article_for_claim(c)
        if article:
            sentences = split_into_sentences(article)
            print(f"\nWikipedia Sentences for claim:\n{c}")
            for s in sentences[:5]:  # print first 5 sentences for brevity
                print("-", s)
        else:
            print("No Wikipedia article found.")

def test_weak_claim(): #opinion= no article found which is correct
    llm_response = "He was a great scientist."

    extractor= ClaimExtractor()
    claims = extractor.extract_claims(llm_response)
    print(f"\nExtracted Claims for text:\n{llm_response}")
    for c in claims:
        print("-", c)
        fetcher= WikiFetcher()
        article = fetcher.get_article_for_claim(c)
        if article:
            sentences = split_into_sentences(article)
            print(f"\nWikipedia Sentences for claim:\n{c}")
            for s in sentences[:5]:  # print first 5 sentences for brevity
                print("-", s)
        else:
            print("No Wikipedia article found.")


if __name__ == "__main__":
    # test_manual_pipeline1()
    test_ambiguous_entity()
    test_multi_entity()
    # test_weak_claim()
