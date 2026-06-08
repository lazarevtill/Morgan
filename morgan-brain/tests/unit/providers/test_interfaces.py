from morgan_brain.interfaces.llm import ChatClient
from morgan_brain.interfaces.embedding import Embedder
from morgan_brain.interfaces.rerank import Reranker


def test_protocols_are_runtime_checkable():
    class C:
        async def agenerate(self, messages, *, model, tools=None, response_format=None): ...
        def astream(self, messages, *, model, tools=None): ...

    class E:
        async def aembed(self, texts): ...

    class R:
        async def arerank(self, query, docs, *, top_k=None): ...

    assert isinstance(C(), ChatClient) and isinstance(E(), Embedder) and isinstance(R(), Reranker)
