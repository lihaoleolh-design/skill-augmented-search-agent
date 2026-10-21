import pytest

from tools.document_tools import (
    DOCUMENT_TOOLS,
    find_in_document,
    open_document,
)


TEST_DOCID = "5412"


@pytest.fixture(scope="module")
def first_page():
    return open_document.invoke({
        "docid": TEST_DOCID,
        "offset": 0,
        "limit": 500,
    })


def test_opens_document_page(first_page):
    assert first_page["docid"] == TEST_DOCID
    assert len(first_page["content"]) == 500
    assert first_page["start_offset"] == 0
    assert first_page["next_offset"] == 500


def test_continues_document_pagination(first_page):
    second_page = open_document.invoke({
        "docid": TEST_DOCID,
        "offset": first_page["next_offset"],
        "limit": 500,
    })

    assert second_page["start_offset"] == 500
    assert second_page["content"]


def test_finds_keyword_at_expected_position():
    result = find_in_document.invoke({
        "docid": TEST_DOCID,
        "keyword": "Cultural Week",
        "max_matches": 3,
        "context_chars": 100,
    })

    assert result["matches"]
    assert result["matches"][0]["position"] == 369
    assert "Cultural Week" in result["matches"][0]["snippet"]


def test_keyword_pagination_does_not_repeat_matches():
    first = find_in_document.invoke({
        "docid": TEST_DOCID,
        "keyword": "University",
        "start_offset": 0,
        "max_matches": 2,
    })

    assert first["next_offset"] is not None

    second = find_in_document.invoke({
        "docid": TEST_DOCID,
        "keyword": "University",
        "start_offset": first["next_offset"],
        "max_matches": 2,
    })

    first_positions = {
        match["position"]
        for match in first["matches"]
    }
    second_positions = {
        match["position"]
        for match in second["matches"]
    }

    assert first_positions.isdisjoint(second_positions)


def test_missing_document_returns_structured_error():
    result = open_document.invoke({
        "docid": "not_existing_doc",
    })

    assert result == {
        "error": "document_not_found",
        "docid": "not_existing_doc",
    }


def test_out_of_range_offset_returns_empty_page():
    result = open_document.invoke({
        "docid": TEST_DOCID,
        "offset": 99_999_999,
        "limit": 500,
    })

    assert result["content"] == ""
    assert result["has_more"] is False
    assert result["next_offset"] is None


def test_document_tools_are_registered_with_schemas():
    registered = {
        tool.name: tool.args
        for tool in DOCUMENT_TOOLS
    }

    assert set(registered) == {
        "open_document",
        "find_in_document",
    }
    assert "docid" in registered["open_document"]
    assert "keyword" in registered["find_in_document"]
