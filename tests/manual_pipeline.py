from app.modules.claim_extraction.extractor import ClaimExtractor
from app.modules.wikipedia.wiki_fetcher import WikiFetcher
from app.utils.text_preprocessing import split_into_sentences
from app.modules.embeddings.embedder import get_top_k_sentences
#---------------------------------------------------------------------


# module 1-2:This test simulates a manual pipeline where we take a response from an LLM, extract claims, and print them out, fetch wikipedia pages for each claim, and split the wikipedia content into sentences.
#       llm response -> claim → wiki → sentences → print 
# module 1-3: 
#       claim → wiki → sentences → embeddings → top-k → print




#----------------------------------------------------------------------
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

            #MODULE 3:


            #EDIT: --- Simple sentence filter ---
            filtered_sentences = []
            for s in sentences:
                words = s.split()

                # remove very short sentences
                if len(words) < 5:
                    continue

                # remove citation-like sentences (years in brackets etc.)
                if "(" in s and ")" in s and any(char.isdigit() for char in s):
                    continue

                # remove headings / weird wiki formatting
                if s.strip().startswith("=") or "==" in s:
                    continue

                # remove quotes / weird sentences
                if '"' in s:
                    continue

                filtered_sentences.append(s)

    
            top_k= get_top_k_sentences(c, filtered_sentences, k=3)
            print(f"\nTop-k Relevant Sentences for claim:\n{c}")
            for s in top_k: 
                print("-", s)
        else:
            print("No Wikipedia article found.")










## EXTRA TESTS FOR EDGE CASES !!!! NEEDED FOR MODULE 1-2 NO NEED TO CHECK FOR MODULE 3 EMBEDDINGS 
#AS THEY ARE INDEPENDENT OF CLAIM EXTRACTION AND WIKI FETCHING. THESE TESTS CHECK IF CLAIM EXTRACTION AND WIKI FETCHING WORK PROPERLY IN EDGE CASES.

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
            top_k= get_top_k_sentences(c, sentences, k=3)
            print(f"\nTop-k Relevant Sentences for claim:\n{c}")
            for s in top_k: 
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
            top_k= get_top_k_sentences(c, sentences, k=3)
            print(f"\nTop-k Relevant Sentences for claim:\n{c}")
            for s in top_k: 
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
            top_k= get_top_k_sentences(c, sentences, k=3)
            print(f"\nTop-k Relevant Sentences for claim:\n{c}")
            for s in top_k: 
                print("-", s)
        else:
            print("No Wikipedia article found.")









if __name__ == "__main__":
    test_manual_pipeline1()
    # test_ambiguous_entity()
    # test_multi_entity()
    # test_weak_claim()
