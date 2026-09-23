"""Validation-only v9.2 entrypoint with source-selective semantic text.

TotalSegmentator and AMOS retain their legacy semantic descriptions.  The
remaining six sources use the expanded metadata-derived descriptions used by
the eight-source training entrypoint.  Model, data, loss and validation logic
are otherwise delegated unchanged to the original entrypoint.
"""

import finetune_multisource_sam2_v9_2_precision as original
import finetune_totalseg_amos_magic_v9_2_precision as precision


multi = original.multi
base = original.base
main = original.main
_make_source_dataset_adaptive = original._make_source_dataset_adaptive
_LEGACY_SOURCES = frozenset({"totalseg", "amos"})


def _catalogue_row(class_id: int) -> dict:
    row = next(
        (
            item
            for item in original._CATALOGUE
            if int(item["global_class_id"]) == int(class_id)
        ),
        None,
    )
    if row is None:
        raise KeyError(f"missing class {class_id} in eight-source catalogue")
    return row


def _legacy_text(row: dict) -> str:
    raw_text = str(row["semantic_text"])
    spec = precision.build_semantic_contrast_spec(raw_text)
    if spec is None:
        return raw_text
    return precision.format_contrastive_semantic_text(
        spec.target_text, spec.parent_text, spec.parent_scale
    )


def mixed_semantic_text(class_id: int, args=None) -> str | None:
    if args is not None and not bool(getattr(args, "prompt_semantic_enabled", True)):
        return None
    row = _catalogue_row(int(class_id))
    if str(row.get("dataset")) in _LEGACY_SOURCES:
        return _legacy_text(row)
    return original.semantic_text_for_global_class(int(class_id), args)


multi.semantic_text_for_global_class = mixed_semantic_text
multi.v91.v9.v6.semantic_text_for_class = mixed_semantic_text
base.semantic_text_for_class = mixed_semantic_text


if __name__ == "__main__":
    main()
