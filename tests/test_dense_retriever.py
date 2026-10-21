from types import SimpleNamespace

import numpy as np

from retrieval.dense_index import iter_text_chunks
from retrieval.dense_retriever import DenseRetriever, QUERY_INSTRUCTION


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        offsets = []
        position = 0
        for word in text.split():
            start = text.index(word, position)
            end = start + len(word)
            offsets.append((start, end))
            position = end
        return {"offset_mapping": offsets}


class FakeModel:
    def __init__(self):
        self.inputs = []

    def encode(self, values, **kwargs):
        self.inputs.extend(values)
        return np.array([[1.0, 0.0]])


class FakeClient:
    def collection_exists(self, name):
        return True

    def query_points(self, **kwargs):
        return SimpleNamespace(points=[
            SimpleNamespace(id="c1", score=0.9, payload={
                "docid": "d1", "text": "first", "url": "u1",
                "start_offset": 0, "end_offset": 5,
            }),
            SimpleNamespace(id="c2", score=0.8, payload={
                "docid": "d1", "text": "duplicate", "url": "u1",
            }),
            SimpleNamespace(id="c3", score=0.7, payload={
                "docid": "d2", "text": "second", "url": "u2",
            }),
        ])


def test_token_chunks_overlap_and_preserve_offsets():
    chunks = list(iter_text_chunks(
        "d", "u", "one two three four five", FakeTokenizer(),
        chunk_tokens=3, overlap_tokens=1,
    ))
    assert [chunk.text for chunk in chunks] == ["one two three", "three four five"]
    assert chunks[0].start_offset == 0
    assert chunks[1].start_offset == 8
    assert chunks[0].point_id != chunks[1].point_id


def test_dense_retriever_adds_instruction_and_deduplicates_documents():
    model = FakeModel()
    retriever = DenseRetriever(
        client=FakeClient(), model=model, collection_name="test",
    )
    results = retriever.search("find this", top_k=2)
    assert model.inputs == [QUERY_INSTRUCTION + "find this"]
    assert [item["docid"] for item in results] == ["d1", "d2"]
    assert results[0]["snippet"] == "first"
