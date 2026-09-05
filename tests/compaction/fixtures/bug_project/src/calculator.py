def apply_sign(magnitude: int, sign: int) -> int:
    """Return ``magnitude`` normalized to the requested sign."""
    normalized = abs(magnitude)
    if sign < 0:
        normalized = abs(normalized)
    return normalized
