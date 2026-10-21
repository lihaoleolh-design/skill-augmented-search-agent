import re
import sqlite3
from collections import defaultdict

from config import settings


DB_PATH = settings.resolve_path(settings.bm25_db_path)
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "his", "how", "i", "in", "into", "is", "it", "its", "of",
    "on", "or", "our", "she", "that", "the", "their", "them", "there",
    "these", "they", "this", "those", "to", "was", "were", "what",
    "when", "where", "which", "who", "whom", "whose", "why", "with",
}


def _quoted(value: str) -> str:
    return f'"{value.replace(chr(34), "")}"'


def _tokens(query: str) -> list[str]:
    return TOKEN_PATTERN.findall(query)


def build_legacy_query(query: str) -> str:
    """Original project query: up to 30 unfiltered tokens joined with OR."""
    return " OR ".join(_quoted(word) for word in _tokens(query)[:30])


def _entity_phrases(query: str) -> list[str]:
    """Extract useful capitalized 2-4 token phrases without an NLP dependency."""
    words = _tokens(query)
    runs: list[list[str]] = []
    current: list[str] = []
    for word in words:
        is_entity_word = word[:1].isupper() and word.casefold() not in STOPWORDS
        if is_entity_word:
            current.append(word)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = []
    if len(current) >= 2:
        runs.append(current)

    phrases: list[str] = []
    for run in runs:
        # Long title-like runs are represented by overlapping windows. This
        # avoids requiring an entire noisy question fragment to occur verbatim.
        width = min(4, len(run))
        for start in range(0, len(run) - width + 1):
            phrases.append(" ".join(run[start:start + width]))
        if len(run) > 2:
            phrases.append(" ".join(run[-3:]))
        phrases.append(" ".join(run[:2]))
    return list(dict.fromkeys(phrases))[:4]


def build_query_variants(query: str) -> list[str]:
    """Build strict and broad FTS5 queries for recall-oriented fusion."""
    raw_tokens = _tokens(query)[:30]
    focused = [word for word in raw_tokens if word.casefold() not in STOPWORDS]
    focused = list(dict.fromkeys(focused))[:20]
    if not focused:
        focused = raw_tokens
    if not focused:
        return []

    variants: list[str] = []
    phrases = _entity_phrases(query)
    if phrases:
        # Phrase matches receive their own ranking rather than being drowned
        # out by common terms in the broad query.
        variants.append(" OR ".join(_quoted(phrase) for phrase in phrases))

    # A moderately strict query rewards documents that connect several clues.
    strict_terms = focused[: min(8, len(focused))]
    if len(strict_terms) >= 2:
        variants.append(" AND ".join(_quoted(word) for word in strict_terms))

    variants.append(" OR ".join(_quoted(word) for word in focused))
    return list(dict.fromkeys(variants))


class BM25FullRetriever:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(str(db_path))

    @staticmethod
    def prepare_query(query: str) -> str:
        """Return the broad cleaned query for backward compatibility."""
        variants = build_query_variants(query)
        return variants[-1] if variants else ""

    def _search_expression(self, expression: str, limit: int) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT
                docid,
                rank AS score,
                snippet(documents, 1, '[', ']', '...', 50) AS snippet,
                url
            FROM documents
            WHERE documents MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (expression, limit),
        ).fetchall()
        return [
            {
                "rank": rank,
                "docid": row[0],
                "bm25_score": float(row[1]),
                "snippet": row[2],
                "url": row[3],
            }
            for rank, row in enumerate(rows, 1)
        ]

    def search_legacy(self, query: str, top_k: int = 5) -> list[dict]:
        """Run the pre-optimization single-query BM25 for fair ablations."""
        expression = build_legacy_query(query)
        if not expression or top_k <= 0:
            return []
        return [
            {
                "rank": item["rank"],
                "docid": item["docid"],
                "score": item["bm25_score"],
                "bm25_score": item["bm25_score"],
                "snippet": item["snippet"],
                "url": item["url"],
            }
            for item in self._search_expression(expression, top_k)
        ]

    def search(self, query: str, top_k: int = 5, candidate_k: int | None = None):
        """Retrieve broadly, fuse query variants with RRF, return a small list."""
        if top_k <= 0:
            return []
        variants = build_query_variants(query)
        if not variants:
            return []

        pool_size = max(top_k, candidate_k or settings.retrieval_candidate_k)
        fused_scores: dict[str, float] = defaultdict(float)
        best: dict[str, dict] = {}
        best_variant_rank: dict[str, int] = {}

        for expression in variants:
            for item in self._search_expression(expression, pool_size):
                docid = str(item["docid"])
                rank = int(item["rank"])
                fused_scores[docid] += 1.0 / (settings.retrieval_rrf_k + rank)
                if docid not in best or rank < best_variant_rank[docid]:
                    best[docid] = item
                    best_variant_rank[docid] = rank

        ordered = sorted(
            best,
            key=lambda docid: (-fused_scores[docid], best_variant_rank[docid], docid),
        )[:top_k]
        return [
            {
                "rank": rank,
                "docid": docid,
                "score": fused_scores[docid],
                "bm25_score": best[docid]["bm25_score"],
                "snippet": best[docid]["snippet"],
                "url": best[docid]["url"],
            }
            for rank, docid in enumerate(ordered, 1)
        ]

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def close(self):
        self.conn.close()
