from worldfoundry.synthesis.visual_generation.bernini.inference import attention


def test_fa3_required_seqused_arguments_are_supplied(monkeypatch):
    calls = []

    def fake_flash(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return q

    monkeypatch.setattr(attention, "_BACKEND", "fa3")
    monkeypatch.setattr(attention, "_flash_varlen", fake_flash)
    monkeypatch.setattr(attention, "_fa3_has_seqused", True)

    q = object()
    result = attention.varlen_attention(q, object(), object(), "cu-q", "cu-k", 17, 19)

    assert result is q
    assert calls[0][3]["seqused_q"] is None
    assert calls[0][3]["seqused_k"] is None
    assert calls[0][3]["max_seqlen_q"] == 17
    assert calls[0][3]["max_seqlen_k"] == 19
