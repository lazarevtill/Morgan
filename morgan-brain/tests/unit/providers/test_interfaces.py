from morgan_brain.interfaces.llm import ChatClient


def test_protocols_are_runtime_checkable():
    class C:
        async def agenerate(self, messages, *, model, tools=None, response_format=None): ...
        def astream(self, messages, *, model, tools=None): ...

    assert isinstance(C(), ChatClient)
