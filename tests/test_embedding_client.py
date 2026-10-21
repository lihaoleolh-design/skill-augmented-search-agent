from retrieval.embedding_client import SiliconFlowEmbeddingClient


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]}


class FakeHttpClient:
    def __init__(self):
        self.call = None

    def post(self, url, **kwargs):
        self.call = (url, kwargs)
        return FakeResponse()


def test_siliconflow_client_sends_batch_and_restores_index_order():
    http = FakeHttpClient()
    encoder = SiliconFlowEmbeddingClient(
        api_key="secret", model_name="BAAI/bge-large-en-v1.5",
        api_url="https://example.test/embeddings", client=http,
    )
    vectors = encoder.encode(["first", "second"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    url, request = http.call
    assert url == "https://example.test/embeddings"
    assert request["headers"]["Authorization"] == "Bearer secret"
    assert request["json"] == {
        "model": "BAAI/bge-large-en-v1.5",
        "input": ["first", "second"],
        "encoding_format": "float",
    }
