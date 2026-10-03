"""Current-frame gradients through frozen SAM memory attention, not memory BPTT."""


def prepare_memory_features(sam, *, training, **kwargs):
    import torch
    from torch.utils.checkpoint import checkpoint

    features = kwargs['current_vision_feats']
    train_input = training and torch.is_grad_enabled() and any(
        value.requires_grad for value in features
    )
    if not train_input:
        with torch.no_grad():
            return sam._prepare_memory_conditioned_features(**kwargs)

    def condition(*current_features):
        call_kwargs = dict(kwargs, current_vision_feats=list(current_features))
        return sam._prepare_memory_conditioned_features(**call_kwargs)

    # Parameters of memory_attention remain frozen. Historical features/pointers
    # were detached at commit time; only the current image features get gradients.
    # Frozen attention can still contain dropout, so recomputation must replay RNG.
    return checkpoint(
        condition, *features, use_reentrant=False, preserve_rng_state=True,
    )
