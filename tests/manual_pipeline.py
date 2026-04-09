from app.modules.claim_extraction.extractor import ClaimExtractor
from app.modules.wikipedia.wiki_fetcher import WikiFetcher
from app.utils.text_preprocessing import split_into_sentences
from app.modules.embeddings.embedder import get_top_k_sentences
from app.modules.verification.nli_verifier import NLIVerifier
from app.modules.scoring.scorer import compute_score
#---------------------------------------------------------------------


# module 1-2:This test simulates a manual pipeline where we take a response from an LLM, extract claims, and print them out, fetch wikipedia pages for each claim, and split the wikipedia content into sentences.
#       llm response -> claim → wiki → sentences → print 
# module 1-3: 
#       claim → wiki → sentences → embeddings → top-k → print
# module 4 [faiss not implemented in v1]
# module 5:
#       claim → wiki → sentences → top-k → nli → print final verdict per claim 
# module 6:
#       claim → wiki → sentences → top-k → nli → store aggregated verdicts per claim → compute_score(all_verdicts for overall llm hallucination risk) → print final score



#----------------------------------------------------------------------
def test_manual_pipeline1():
    #llm response string
    llm_response = "Albert Einstein was a physicist. He developed the theory of relativity."
    # llm_response = "Albert Einstein was born in 1879. He was born in 2000."

    extractor= ClaimExtractor()
    claims = extractor.extract_claims(llm_response)
    print(f"\nExtracted Claims for text:\n{llm_response}")
    all_verdicts= []
    for c in claims:
        print("-", c)
        fetcher= WikiFetcher()
        article = fetcher.get_article_for_claim(c)
        if article:
            sentences = split_into_sentences(article)

            #MODULE 3: top k evidence 


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



            #Module 5: print final verdict per claim via nli model
            verifier = NLIVerifier()
            verdict = verifier.verify(c, top_k)
            print(f"\nClaim:\n{c}")
            print(f"\nTop-k Relevant Sentences for claim:\n{c}")
            for s in top_k:
                print("-", s)
            print(f"Verification Verdict: {verdict}") #supported/contradicted/not enough info
            all_verdicts.append(verdict)

        else:
            print("No Wikipedia article found.")
    
    #module 6: compute overall hallucination risk score for the llm response based on all claim verdicts
    score_result = compute_score(all_verdicts)
    print(f"\nOverall Score for LLM response:\n{score_result}")











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




## EXTRA TESTS FOR MODULE 5
def test_additional_cases():
    extractor = ClaimExtractor()
    fetcher = WikiFetcher()
    verifier = NLIVerifier()

    test_cases = [
        "Albert Einstein was born in 2000.",
        "Albert Einstein was a singer."
    ]

    for llm_response in test_cases:
        print(f"\n\n=== TEST CASE ===")
        print(f"Input: {llm_response}")

        claims = extractor.extract_claims(llm_response)

        for c in claims:
            article = fetcher.get_article_for_claim(c)

            if not article:
                print(f"Claim: {c}")
                print("No article found → likely NOT ENOUGH INFO")
                continue

            sentences = split_into_sentences(article)

            # reuse your SAME filtering logic
            filtered_sentences = []
            for s in sentences:
                words = s.split()

                if len(words) < 5:
                    continue
                if "(" in s and ")" in s and any(char.isdigit() for char in s):
                    continue
                if s.strip().startswith("=") or "==" in s:
                    continue
                if '"' in s:
                    continue

                filtered_sentences.append(s)

            top_k = get_top_k_sentences(c, filtered_sentences, k=3)

            verdict = verifier.verify(c, top_k)

            print(f"\nClaim: {c}")
            print("Top evidence:")
            for s in top_k:
                print("-", s)
            print(f"Verdict: {verdict}")




if __name__ == "__main__":
    test_manual_pipeline1()
    # test_additional_cases()
    # test_ambiguous_entity()
    # test_multi_entity()
    # test_weak_claim()
