"""Integration gated on optional PEFT/Transformers dependencies."""
import pytest

pytest.importorskip('peft', reason='Optional PEFT dependency is not installed')
pytest.importorskip('transformers', reason='Optional Transformers dependency is not installed')


def test_tiny_real_peft_adapter_load_logits_generation(tmp_path):
    from examples.offline_peft_equivalence import run
    result = run(tmp_path)
    assert result['status'] == 'REAL_PEFT_TINY_OFFLINE_EQUIVALENCE_PASS'
    assert result['max_abs_logits_error'] < 1e-4
