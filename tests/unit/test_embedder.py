from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder


async def test_fake_embedder_is_deterministic_and_fixed_dim():
    emb = FakeEmbedder(dim=16)
    a = await emb.embed("hello world")
    b = await emb.embed("hello world")
    c = await emb.embed("different text")
    assert len(a) == 16
    assert a == b
    assert a != c


async def test_fake_embedder_batch():
    emb = FakeEmbedder(dim=8)
    out = await emb.embed_batch(["a", "b", "c"])
    assert len(out) == 3 and all(len(v) == 8 for v in out)
