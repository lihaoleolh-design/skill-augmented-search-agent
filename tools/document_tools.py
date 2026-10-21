# tools/document_tools.py

import sqlite3
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from config import settings

DB_PATH = settings.resolve_path(settings.bm25_db_path)

MAX_READ_CHARS = settings.max_read_chars
MAX_MATCHES = settings.max_matches
MAX_CONTEXT_CHARS = settings.max_context_chars


# =====================================================
# Database
# =====================================================

def get_connection() -> sqlite3.Connection:
    """
    Open SQLite database in read-only mode.
    """
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"BM25 database not found: {DB_PATH}"
        )

    conn = sqlite3.connect(
        DB_PATH.resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=15
    )

    return conn


def get_document_metadata(
    conn: sqlite3.Connection,
    docid: str
):
    """
    Get document metadata using the indexed
    docid -> rowid mapping.

    Returns:
        (rowid, docid, url, total_chars)
        or None
    """
    return conn.execute(
        """
        SELECT
            d.rowid,
            d.docid,
            d.url,
            length(d.text)
        FROM doc_locator AS l
        JOIN documents AS d
          ON d.rowid = l.document_rowid
        WHERE l.docid = ?
        LIMIT 1
        """,
        (docid,)
    ).fetchone()


def read_document_range(
    conn: sqlite3.Connection,
    rowid: int,
    offset: int,
    limit: int
) -> str:
    """
    Read a substring without returning the entire
    document to the Python application.
    """
    row = conn.execute(
        """
        SELECT substr(text, ?, ?)
        FROM documents
        WHERE rowid = ?
        """,
        (offset + 1, limit, rowid)
    ).fetchone()

    return (row[0] or "") if row else ""


# =====================================================
# Input Schemas
# =====================================================

class OpenDocumentInput(BaseModel):
    docid: str = Field(
        min_length=1,
        description="Document ID from search_documents"
    )

    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Starting character position, zero-based"
        )
    )

    limit: int = Field(
        default=2000,
        ge=1,
        le=MAX_READ_CHARS,
        description="Maximum number of characters to read"
    )


class FindDocumentInput(BaseModel):
    docid: str = Field(
        min_length=1,
        description="Document ID from search_documents"
    )

    keyword: str = Field(
        min_length=1,
        max_length=200,
        description="Exact keyword or phrase to find"
    )

    start_offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Start searching from this character position. "
            "Use next_offset from a previous result to continue."
        )
    )

    max_matches: int = Field(
        default=5,
        ge=1,
        le=MAX_MATCHES,
        description="Maximum number of matches returned"
    )

    context_chars: int = Field(
        default=150,
        ge=0,
        le=MAX_CONTEXT_CHARS,
        description="Context characters before and after a match"
    )


# =====================================================
# Tool 1: open_document
# =====================================================

@tool(args_schema=OpenDocumentInput)
def open_document(
    docid: str,
    offset: int = 0,
    limit: int = 2000
) -> dict:
    """
    Read a specific part of a document by document ID.

    Use this tool after search_documents when the agent
    needs more information than the search snippet.

    For long documents, use next_offset to continue
    reading rather than requesting the entire document.
    """

    with get_connection() as conn:
        metadata = get_document_metadata(conn, docid)

        if metadata is None:
            return {
                "error": "document_not_found",
                "docid": docid
            }

        rowid, real_docid, url, total_chars = metadata
        total_chars = total_chars or 0

        # Prevent an out-of-range offset.
        start = min(offset, total_chars)

        if start >= total_chars:
            return {
                "docid": real_docid,
                "url": url,
                "content": "",
                "start_offset": start,
                "end_offset": start,
                "total_chars": total_chars,
                "has_more": False,
                "next_offset": None
            }

        content = read_document_range(
            conn=conn,
            rowid=rowid,
            offset=start,
            limit=limit
        )

        end = start + len(content)
        has_more = end < total_chars

        return {
            "docid": real_docid,
            "url": url,
            "content": content,
            "start_offset": start,
            "end_offset": end,
            "total_chars": total_chars,
            "has_more": has_more,
            "next_offset": end if has_more else None
        }


# =====================================================
# Tool 2: find_in_document
# =====================================================

@tool(args_schema=FindDocumentInput)
def find_in_document(
    docid: str,
    keyword: str,
    start_offset: int = 0,
    max_matches: int = 5,
    context_chars: int = 150
) -> dict:
    """
    Locate an exact keyword or phrase within one document.

    Returns character offsets and surrounding evidence.

    This version supports pagination using start_offset.
    Matching is case-insensitive for ASCII English text.
    """

    keyword = keyword.strip()

    if not keyword:
        return {
            "error": "empty_keyword",
            "docid": docid
        }

    with get_connection() as conn:
        metadata = get_document_metadata(conn, docid)

        if metadata is None:
            return {
                "error": "document_not_found",
                "docid": docid
            }

        rowid, real_docid, url, total_chars = metadata
        total_chars = total_chars or 0

        current_offset = min(start_offset, total_chars)
        matches = []

        # Find at most max_matches occurrences.
        for _ in range(max_matches):

            if current_offset >= total_chars:
                break

            result = conn.execute(
                """
                SELECT instr(
                    lower(substr(text, ?)),
                    lower(?)
                )
                FROM documents
                WHERE rowid = ?
                """,
                (
                    current_offset + 1,
                    keyword,
                    rowid
                )
            ).fetchone()

            relative_pos = result[0] if result else 0

            if relative_pos == 0:
                break

            # SQLite instr() is 1-based;
            # API offsets are 0-based.
            position = current_offset + relative_pos - 1

            snippet_start = max(
                0,
                position - context_chars
            )

            snippet_end = min(
                total_chars,
                position + len(keyword) + context_chars
            )

            snippet = read_document_range(
                conn,
                rowid,
                snippet_start,
                snippet_end - snippet_start
            )

            matches.append({
                "position": position,
                "start_offset": snippet_start,
                "end_offset": snippet_end,
                "snippet": snippet
            })

            # Continue after the matched phrase.
            current_offset = position + len(keyword)

        # Check whether more matches may exist.
        # This deliberately does not run an additional
        # full-document scan.
        has_more_possible = (
            len(matches) == max_matches
            and current_offset < total_chars
        )

        return {
            "docid": real_docid,
            "url": url,
            "keyword": keyword,
            "search_start_offset": start_offset,
            "matches": matches,
            "returned_matches": len(matches),
            "has_more_possible": has_more_possible,
            "next_offset": (
                current_offset
                if has_more_possible
                else None
            )
        }


# =====================================================
# Tool Registration
# =====================================================

DOCUMENT_TOOLS = [
    open_document,
    find_in_document
]