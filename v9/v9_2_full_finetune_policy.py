"""Pure, testable Hiera parameter-selection policy for v9.2 full fine-tuning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def select_encoder_parameter_names(
    names: Iterable[str], stage_ends: Sequence[int], scope: str
) -> set[str]:
    if scope not in {"late", "all"}:
        raise ValueError(f"encoder scope must be 'late' or 'all', got {scope!r}")
    ends = [int(value) for value in stage_ends]
    if len(ends) < 2 or any(right <= left for left, right in zip(ends, ends[1:])):
        raise ValueError(f"invalid Hiera stage boundaries: {ends!r}")
    available = set(names)
    if scope == "all":
        return available
    first_last_stage_block = ends[-2] + 1
    selected = set()
    for name in available:
        if name.startswith("neck."):
            selected.add(name)
        elif name.startswith("trunk.blocks."):
            parts = name.split(".", 3)
            if len(parts) == 4 and parts[2].isdigit() and int(parts[2]) >= first_last_stage_block:
                selected.add(name)
    if not selected:
        raise ValueError("late encoder scope matched no Hiera or neck parameters")
    return selected
