from src.calculator import apply_sign


def test_apply_sign_normalizes_positive_values():
    assert apply_sign(-4, 1) == 4


def test_apply_sign_normalizes_negative_values():
    assert apply_sign(4, -1) == -4
