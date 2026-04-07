import re
from numpy.strings import title
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError


class WikiFetcher:
    """
    Fetches Wikipedia article content for a given factual claim.
    """

    def __init__(self, language: str = "en"):
        wikipedia.set_lang(language)

    # -------------------------------
    # Public API
    # -------------------------------

    def search_page(self, claim: str) -> str | None:
        """
        Identify the most relevant Wikipedia page title for a given claim.
        """

        entity = self._extract_entity(claim)
        if not entity:
            return None
        
        # --- SIMPLE QUERY IMPROVEMENT ---
        claim_lower = claim.lower()

        if any(word in claim_lower for word in ["founded", "company", "headquarters", "ceo"]):
            entity = f"{entity} company"

        results = wikipedia.search(entity, results=5)
        if not results:
            return None

        # Prefer result that closely matches entity
        for result in results:
            if entity.lower() in result.lower():
                return result

        return results[0]

    def fetch_page(self, title: str) -> str | None:
        """
        Fetch full Wikipedia article content.
        """

        if title is None:
            return None

        try:
            page = wikipedia.page(title, auto_suggest=False)
            return page.content

        except Exception:
            # retry once (handles transient API failure)
            try:
                page = wikipedia.page(title, auto_suggest=False)
                return page.content
            except Exception:
                return None

    def get_article_for_claim(self, claim: str) -> str | None:
        """
        End-to-end: claim → article text
        """

        title = self.search_page(claim)
        if not title:
            return None

        return self.fetch_page(title)

    # -------------------------------
    # Private Helpers
    # -------------------------------

    def _extract_entity(self, claim: str) -> str:
        """
        Extract likely entity from claim.
        """

        claim = claim.strip()
        if not claim:
            return ""

        match = re.match(r"^([A-Z][a-zA-Z''\-]*(?:\s+[A-Z][a-zA-Z''\-]*)*)", claim)
        if match:
            candidate = match.group(1).strip()
            if len(candidate) >= 2:
                return candidate

        words = claim.split()
        return " ".join(words[:4])

    def _retry_disambiguation(self, options: list[str]) -> str | None:
        """
        Retry fetching page from disambiguation options.
        """

        for option in options:
            try:
                page = wikipedia.page(option, auto_suggest=False)
                return page.content

            except (DisambiguationError, PageError):
                continue

            except Exception as e:
                print(f"[WikiFetcher] Error during disambiguation retry for '{option}': {e}")
                continue

        return None