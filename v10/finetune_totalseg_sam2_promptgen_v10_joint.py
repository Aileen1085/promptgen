from __future__ import annotations
import argparse, logging, os, random, sys, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
V10_ROOT=Path(__file__).resolve().parent
if str(V10_ROOT) not in sys.path:
    sys.path.insert(0, str(V10_ROOT))

def _gpu(default="0,1"):
    try: i=sys.argv.index("--gpu"); return sys.argv[i+1]
    except (ValueError,IndexError): return default
os.environ["CUDA_VISIBLE_DEVICES"]=_gpu()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

import torch
import torch.distributed as dist
import nibabel as nib
import numpy as np
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from v10_sam2 import SAM2MedicalAdapter
from model.cross_view_prompt_token_generator_v10 import CrossViewPromptTokenGeneratorV10
from model.sam2_decoder_3d_prompt_adapter_v10 import SAM2MaskDecoderWith3DPromptAdapterV10
from v10_data import (DEFAULT_CLASS_PRESENCE_MANIFEST, DEFAULT_DATA_ROOT,
    DEFAULT_LABEL_MAP, DEFAULT_PROMPT_CACHE_DIR, DEFAULT_PROMPT_SLICE_CACHE_DIR,
    discover_totalseg_split_pairs, load_label_map, make_dataset)
from v10_metrics import prompt_consistency_loss, segmentation_loss
from v10_sam2_memory_roi import JointTrainingForwardMemoryV10, evaluate_memory_v10
from utils.distributed_runtime import (destroy_distributed, distributed_epoch_indices,
    find_free_port, init_distributed, parse_gpu_ids)
from v10_2_sam_finetuning import (apply_staged_learning_rates,
    clear_optimizer_group_gradients, configure_sam_finetuning,
    gradient_accumulation_window, interleave_indices_by_source,
    load_sam_tuning_from_checkpoint, sam_tuning_optimizer_groups,
    sam_tuning_state_dict)
from v10_training_ema import TrainableParameterEMA
from v10_2_plateau import ValidationPlateau

ROOT=Path(__file__).resolve().parent
V9_ROOT=ROOT.parent
ARCH="totalseg_sam2_v10_prompt_conditioned_memory_global_pool_v2"
DEFAULT_V9=str(V9_ROOT/"weights/v9best.pth")
_ACTIVE_SAM_TUNING=None
_ACTIVE_EMA=None
_ACTIVE_PLATEAU=None

class SAM2MedicalAdapterV10(SAM2MedicalAdapter):
    def __init__(self,args,device):
        cfg=str(Path(args.model_cfg).resolve()); cfg="//"+cfg if cfg.startswith("/") else cfg
        tuning_mode=str(getattr(args,"sam_tuning_mode","") or "")
        lora_targets=("q_proj","k_proj","v_proj","qkv") if tuning_mode=="encoder_lora_full_decoder" else None
        super().__init__(cfg,args.checkpoint,device=device,lora_r=int(getattr(args,"sam_encoder_lora_rank",8)),
                         lora_alpha=int(getattr(args,"sam_encoder_lora_alpha",16)),
                         lora_target_modules=lora_targets)
        core=self.sam.sam_mask_decoder
        self.sam.sam_mask_decoder=SAM2MaskDecoderWith3DPromptAdapterV10(core,int(self.sam.sam_prompt_embed_dim),256,0.0)
        self.sam.sam_mask_decoder.prompt_3d_adapter.conditioning_enabled=True


def _estimate_case_cost(dataset, index: int) -> int:
    """Cheap deterministic proxy for a case's decoder work.

    The actual prompt-derived ROI depends on the sampled class, so computing
    every ROI here would duplicate dataset work. Raw canonical depth is a good
    startup-only proxy and requires only a NIfTI header or an mmap array shape.
    """
    estimator = getattr(dataset, "estimate_case_cost", None)
    if callable(estimator):
        try:
            estimated = estimator(int(index))
            if estimated is not None and np.isfinite(float(estimated)):
                return max(1, int(estimated))
        except (OSError, ValueError, KeyError):
            pass
    path = str(dataset.image_paths[int(index)])
    try:
        if path.endswith(".npy"):
            shape = np.load(path, mmap_mode="r", allow_pickle=False).shape
        else:
            shape = nib.load(path).shape
        return max(1, int(max(shape[:3])))
    except Exception:
        try:
            return max(1, int(os.path.getsize(path) // (1024 * 1024)))
        except OSError:
            return 1


def _parse_rank_cost_weights(value, world_size: int) -> list[float]:
    """Parse optional relative rank throughput weights."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return [1.0] * int(world_size)
    if isinstance(value, str):
        rows = [row.strip() for row in value.split(",") if row.strip()]
    else:
        rows = list(value)
    weights = [float(row) for row in rows]
    if len(weights) != int(world_size):
        raise ValueError(
            "DDP rank cost weights count must match world size "
            f"({len(weights)} != {world_size})"
        )
    if any(not np.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise ValueError("DDP rank cost weights must be finite and positive")
    return weights


def _partition_indices_by_weighted_cost(
    indices,
    *,
    world_size: int,
    per_rank_cases: int,
    rank_cost_weights,
    cost_of,
) -> list[list[int]]:
    """Greedily balance estimated time while keeping equal DDP step counts."""
    world_size = int(world_size)
    per_rank_cases = int(per_rank_cases)
    weights = _parse_rank_cost_weights(rank_cost_weights, world_size)
    buckets = [[] for _ in range(world_size)]
    loads = [0.0] * world_size
    ordered = sorted(indices, key=cost_of, reverse=True)
    for index in ordered:
        cost = float(cost_of(index))
        available = [
            rank for rank in range(world_size)
            if len(buckets[rank]) < per_rank_cases
        ]
        if not available:
            raise ValueError("weighted cost partition exceeds per-rank case budget")
        rank = min(
            available,
            key=lambda item: (
                (loads[item] + cost) / weights[item],
                len(buckets[item]),
                item,
            ),
        )
        buckets[rank].append(int(index))
        loads[rank] += cost
    if any(len(bucket) != per_rank_cases for bucket in buckets):
        raise ValueError("weighted cost partition did not fill every DDP rank")
    for rank, bucket in enumerate(buckets):
        ordered = sorted(bucket, key=cost_of, reverse=True)
        interleaved = []
        left, right = 0, len(ordered) - 1
        while left <= right:
            interleaved.append(ordered[left])
            left += 1
            if left <= right:
                interleaved.append(ordered[right])
                right -= 1
        buckets[rank] = interleaved
    return buckets


def _balanced_epoch_indices(dataset, args, ctx, epoch: int) -> list[int]:
    """Shard one global case budget and pair similar-cost cases across ranks."""
    global_cases = int(args.train_cases_per_epoch)
    world_size = int(ctx.world_size)
    if global_cases <= 0:
        raise ValueError("global train cases per epoch must be positive")
    if world_size <= 0:
        raise ValueError("DDP world size must be positive")
    if global_cases % world_size:
        raise ValueError(
            f"global train cases ({global_cases}) must divide evenly by "
            f"DDP world size ({world_size})"
        )
    per_rank_cases = global_cases // world_size
    selected = distributed_epoch_indices(
        len(dataset), per_rank_cases, world_size, args.seed, epoch
    )
    if world_size > 1 and selected:
        weights = _parse_rank_cost_weights(
            getattr(args, "ddp_rank_cost_weights", ""), world_size
        )
        # Stable sorting keeps the seeded random order among equal-depth cases.
        selected.sort(key=lambda idx: _estimate_case_cost(dataset, idx), reverse=True)
        if any(not np.isclose(weight, weights[0]) for weight in weights[1:]):
            buckets = _partition_indices_by_weighted_cost(
                selected,
                world_size=world_size,
                per_rank_cases=per_rank_cases,
                rank_cost_weights=weights,
                cost_of=lambda idx: _estimate_case_cost(dataset, idx),
            )
            selected = buckets[int(ctx.rank)]
        else:
            selected = selected[int(ctx.rank)::world_size]
    if (
        selected
        and bool(getattr(args, "train_source_diverse_accumulation", False))
        and hasattr(dataset, "schedule")
    ):
        selected = interleave_indices_by_source(
            selected,
            lambda index: dataset.schedule[int(index)]["dataset"],
            int(args.train_gradient_accumulation_steps),
        )
    return selected


class _ThreadPrefetchLoader:
    """Prefetch a synchronous DataLoader without forking after CUDA init."""

    def __init__(self, loader, depth: int):
        self.loader = loader
        self.depth = max(1, int(depth))

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        source = iter(self.loader)
        pending = deque()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v10-data")
        try:
            for _ in range(self.depth):
                pending.append(executor.submit(next, source))
            while pending:
                try:
                    batch = pending.popleft().result()
                except StopIteration:
                    break
                pending.append(executor.submit(next, source))
                yield batch
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)


def _make_training_loader(dataset, indices, args):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list,
    )
    prefetch = max(0, int(getattr(args, "train_prefetch_batches", 0)))
    return _ThreadPrefetchLoader(loader, prefetch) if prefetch else loader


def _seed_process_rng(seed: int, rank: int) -> None:
    process_seed = int(seed) + int(rank)
    random.seed(process_seed)
    np.random.seed(process_seed % (2**32))
    torch.manual_seed(process_seed)


class DDPTrainingStepV10(torch.nn.Module):
    """Keep full-resolution logits inside DDP; expose only a scalar loss."""
    def __init__(self,prompt,adapter):
        super().__init__()
        self.joint=JointTrainingForwardMemoryV10(prompt,adapter)

    def forward(self,batch,args,device):
        prepared,logits=self.joint(batch,args,device)
        loss=segmentation_loss(
            logits,prepared[4],args.seg_pos_weight_max,
            args.boundary_loss_weight,args.boundary_dice_loss_weight,
            args.boundary_band_width,args.boundary_loss_size,
            args.volume_tversky_loss_weight,
            args.volume_tversky_fp_weight,args.volume_tversky_fn_weight,
        )
        if float(args.prompt_consistency_loss_weight) > 0.0:
            offset=0
            consistency_rows=[]
            for frames,context in zip(prepared[5],prepared[7]):
                consistency_rows.append(prompt_consistency_loss(
                    logits[offset:offset+frames],context["aligned_prompt"],
                    context["foreground_mode"],context["background_mode"],
                    args.prompt_consistency_threshold,
                    args.prompt_consistency_foreground_weight,
                    args.prompt_consistency_background_weight,
                    args.prompt_consistency_apply_to_box,
                ))
                offset+=frames
            loss=loss+float(args.prompt_consistency_loss_weight)*torch.stack(consistency_rows).mean()
        return loss,sum(prepared[5])

def parser():
    p=argparse.ArgumentParser("Standalone TotalSeg SAM2 PromptGen v10")
    p.add_argument(
        "--gpu",default="0,1",
        help="One GPU id or a comma-separated list; multiple ids enable DDP.",
    ); p.add_argument("--model-cfg",default=str(V9_ROOT/"sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"))
    p.add_argument("--checkpoint",default=str(V9_ROOT/"sam2/checkpoints/sam2.1_hiera_large.pt"))
    p.add_argument("--prompt-generator-checkpoint",default=DEFAULT_V9)
    p.add_argument("--resume-checkpoint",default="",
                   help="Resume v10 model, optimizer and epoch in the checkpoint's run directory.")
    p.add_argument("--resume-lr-scale",type=float,default=1.0,
                   help="Multiply restored optimizer learning rates after resume; must be positive.")
    p.add_argument("--resume-new-run",action=argparse.BooleanOptionalAction,default=False,
                   help="Resume model/optimizer/epoch but write checkpoints under a new out-dir run.")
    p.add_argument("--out-dir",default=str(ROOT/"output/output_totalseg_sam2_promptgen_v10_joint"))
    p.add_argument("--data-root",default=DEFAULT_DATA_ROOT); p.add_argument("--label-map",default=DEFAULT_LABEL_MAP)
    p.add_argument("--prompt-cache-dir",default=DEFAULT_PROMPT_CACHE_DIR); p.add_argument("--prompt-slice-cache-dir",default=DEFAULT_PROMPT_SLICE_CACHE_DIR)
    p.add_argument("--class-presence-manifest",default=DEFAULT_CLASS_PRESENCE_MANIFEST)
    p.add_argument("--epochs",type=int,default=150); p.add_argument("--validate-every",type=int,default=10)
    p.add_argument(
        "--train-cases-per-epoch",type=int,default=20,
        help="Global cases per epoch across all DDP ranks; must divide by world size.",
    )
    p.add_argument(
        "--train-prefetch-batches", type=int, default=0,
        help=("Asynchronously prepare this many upcoming training batches in one "
              "CPU thread per rank. 0 keeps synchronous loading."),
    )
    p.add_argument(
        "--train-gradient-accumulation-steps", type=int, default=1,
        help="Number of case microbatches accumulated before each optimizer step.",
    )
    p.add_argument(
        "--train-source-diverse-accumulation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interleave source schedules so accumulation groups are source-diverse.",
    )
    p.add_argument(
        "--ddp-rank-cost-weights", default="",
        help=("Optional comma-separated relative rank throughput weights. "
              "The count must match the active GPU count; empty uses equal weights."),
    )
    p.add_argument("--sam2-frame-batch-size",type=int,default=16)
    p.add_argument("--sam-encoder-frame-batch-size",type=int,default=4,
                   help="Small checkpointed image-encoder chunk used only when encoder tuning is active.")
    p.add_argument("--sam-tuning-mode",default="",
                   choices=("","encoder_lora_full_decoder","last_stage_full_decoder","full_encoder_full_decoder"))
    p.add_argument("--sam-encoder-lora-rank",type=int,default=8)
    p.add_argument("--sam-encoder-lora-alpha",type=int,default=16)
    p.add_argument("--encoder-lr",type=float,default=5e-6)
    p.add_argument("--decoder-lr",type=float,default=2e-6)
    p.add_argument("--decoder-transformer-lr",type=float,default=None,
                   help="Optional LR for mask-decoder transformer parameters.")
    p.add_argument("--decoder-head-lr",type=float,default=None,
                   help="Optional LR for non-transformer mask-decoder parameters.")
    p.add_argument("--sam-encoder-unfreeze-epoch",type=int,default=1,
                   help="First epoch that updates the configured SAM image-encoder parameters.")
    p.add_argument("--lr-warmup-epochs",type=int,default=0,
                   help="Linear optimizer learning-rate warmup length; 0 disables warmup.")
    p.add_argument("--lr-schedule-epochs",type=int,default=0,
                   help="Optional original LR schedule horizon for a shorter control run; 0 uses epochs.")
    p.add_argument("--lr-cosine-min-ratio",type=float,default=1.0,
                   help="Final cosine learning-rate ratio; 1 preserves constant learning rates.")
    p.add_argument("--plateau-early-stop-patience",type=int,default=0,
                   help="Stop after this many consecutive non-improving validations; 0 disables plateau control.")
    p.add_argument("--plateau-lr-patience",type=int,default=2,
                   help="Reduce each trainable LR after this many non-improving validations.")
    p.add_argument("--plateau-lr-factor",type=float,default=0.5)
    p.add_argument("--plateau-min-lr-ratio",type=float,default=0.1,
                   help="Lower bound as a fraction of each resumed group's initial LR.")
    p.add_argument("--plateau-min-delta",type=float,default=0.001,
                   help="Minimum absolute weighted-Dice improvement that resets patience.")
    p.add_argument("--plateau-reset-on-resume",action="store_true",
                   help="Rebase plateau best to checkpoint best and clear old patience counters; preserve optimizer/LR history.")
    p.add_argument(
        "--train-memory-group-size", type=int, default=1,
        help=("Decode every frame with its own prompt, but update the SAM2 memory bank "
              "only at each group tail. 1 preserves the original per-frame propagation."),
    )
    p.add_argument(
        "--eval-memory-group-size", type=int, default=1,
        help="Memory update group used by validation/inference; keep 1 for exact full propagation.",
    )
    p.add_argument("--train-max-roi-frames",type=int,default=192,
                   help="Prompt-derived random depth window for truncated-BPTT training; 0 keeps every ROI frame.")
    p.add_argument("--prompt-depth-chunk-size",type=int,default=48,
                   help="Number of real ROI frames retained from each PromptGen depth chunk.")
    p.add_argument("--prompt-depth-halo",type=int,default=24,
                   help="Real context frames added at both sides of each PromptGen depth chunk.")
    p.add_argument("--prompt-full-depth-limit",type=int,default=96,
                   help="Use one PromptGen pass up to this depth; larger ROIs use all-frame chunks.")
    p.add_argument("--prompt-checkpoint-depth-threshold",type=int,default=96,
                   help="Checkpoint the 3D encoder only above this per-pass depth; 0 always checkpoints.")
    p.add_argument("--decoder-gradient-checkpoint",action="store_true",
                   help="Recompute each train-time SAM2 decoder call during backward to save memory.")
    p.add_argument("--decoder-checkpoint-min-frames",type=int,default=96,
                   help="Automatically checkpoint decoder calls when total ROI depth exceeds this value.")
    p.add_argument("--sam2-feature-cache-device",choices=("auto","cpu","cuda"),default="auto",
                   help="Where frozen per-case SAM2 features are retained between PromptGen and memory decode.")
    p.add_argument("--sam2-feature-cache-gpu-max-frames",type=int,default=128,
                   help="With feature-cache-device=auto, keep features on GPU up to this ROI depth.")
    p.add_argument("--sam2-feature-cache-gpu-max-eval-frames",type=int,default=384,
                   help="Evaluation-only GPU feature-cache depth limit; inference has no backward graph.")
    p.add_argument("--prompt-lr",type=float,default=2e-5); p.add_argument("--adapter-lr",type=float,default=3e-5)
    p.add_argument("--memory-adapter-lr",type=float,default=3e-5)
    p.add_argument("--weight-decay",type=float,default=1e-5); p.add_argument("--grad-clip-norm",type=float,default=1.0)
    p.add_argument("--ema-decay",type=float,default=0.0,
                   help="EMA decay for validation/best checkpoints; 0 disables EMA.")
    p.add_argument("--seg-pos-weight-max",type=float,default=6.0)
    p.add_argument("--boundary-loss-weight",type=float,default=0.2)
    p.add_argument("--boundary-dice-loss-weight",type=float,default=0.1)
    p.add_argument("--boundary-band-width",type=int,default=3)
    p.add_argument("--boundary-loss-size",type=int,default=256)
    p.add_argument("--volume-tversky-loss-weight",type=float,default=0.0)
    p.add_argument("--volume-tversky-fp-weight",type=float,default=0.6)
    p.add_argument("--volume-tversky-fn-weight",type=float,default=0.4)
    p.add_argument("--prompt-consistency-loss-weight",type=float,default=0.10)
    p.add_argument("--prompt-consistency-foreground-weight",type=float,default=1.0)
    p.add_argument("--prompt-consistency-background-weight",type=float,default=0.5)
    p.add_argument("--prompt-consistency-threshold",type=float,default=0.75)
    p.add_argument("--prompt-consistency-apply-to-box",action=argparse.BooleanOptionalAction,default=False)
    p.add_argument("--mask-threshold",type=float,default=.60)
    p.add_argument("--validation-thresholds",type=float,nargs="+",default=(.55,.60,.65))
    p.add_argument("--model-input-size",type=int,default=1024); p.add_argument("--cache-image-size",type=int,default=512)
    p.add_argument("--prompt-plane-size",type=int,default=256); p.add_argument("--prompt-work-size",type=int,default=192)
    p.add_argument("--prompt-sigma",type=float,default=8.0); p.add_argument("--scribble-num-slices",type=int,default=1)
    p.add_argument("--scribble-thickness",type=float,default=.03); p.add_argument("--roi-context-slices",type=int,default=8)
    p.add_argument("--roi-context-ratio",type=float,default=.25); p.add_argument("--roi-min-frames",type=int,default=16)
    p.add_argument("--patch-scaling",action=argparse.BooleanOptionalAction,default=False,
                   help="Legacy direct prompt-box H/W crop; disabled by default because raw 2D prompts do not bound the full 3D target.")
    p.add_argument("--patch-margin-ratio",type=float,default=.25,
                   help="Relative margin around the prompt-derived in-plane patch.")
    p.add_argument("--patch-min-size",type=int,default=128,
                   help="Minimum padded in-plane patch side before scaling.")
    p.add_argument("--prompt-hidden-dim",type=int,default=24); p.add_argument("--prompt-num-tokens",type=int,default=2)
    p.add_argument("--prompt-volume-tokens",type=int,default=1); p.add_argument("--prompt-view-feature-dim",type=int,default=12)
    p.add_argument("--prompt-image-feature-proj-dim",type=int,default=8); p.add_argument("--prompt-semantic-dim",type=int,default=16)
    p.add_argument("--prompt-semantic-clip-ckpt",default=str(V9_ROOT/"models/clip-vit-base-patch32"))
    p.add_argument("--background-point-count",type=int,default=4); p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--amp-dtype",choices=("float16","bfloat16"),default="bfloat16")
    p.add_argument("--dataset-foreground-mode",choices=("random","scribble","box"),default="random")
    p.add_argument("--val-foreground-mode",choices=("random","scribble","box"),default="random")
    p.add_argument("--val-background-mode",choices=("random","scribble","point","none"),default="random")
    p.add_argument("--validation-max-tasks",type=int,default=0,
                   help="Validation task target; automatically raised to include every class. 0 evaluates all tasks.")
    p.add_argument("--validation-seed",type=int,default=2027,
                   help="Fixed seed for validation task selection and prompt-mode assignment.")
    p.add_argument("--anchor-coarse-crop",action=argparse.BooleanOptionalAction,default=False,
                   help="Validation/inference only: opt in to an anchor coarse-mask XY crop. Disabled by default because it can truncate the target and distort prompts.")
    p.add_argument("--anchor-coarse-threshold",type=float,default=.50)
    p.add_argument("--anchor-coarse-margin-ratio",type=float,default=.50,
                   help="Large relative margin around the anchor coarse bbox.")
    p.add_argument("--anchor-coarse-min-size",type=int,default=256)
    p.add_argument("--validate-before-train",action=argparse.BooleanOptionalAction,default=False,
                   help="Evaluate the initialized checkpoint on the same validation subset before training.")
    return p

def make_prompt(args,adapter,device):
    CrossViewPromptTokenGeneratorV10.DEFAULT_BACKGROUND_POINT_COUNT=args.background_point_count
    CrossViewPromptTokenGeneratorV10.DEFAULT_EVAL_RANDOM_PROMPT_MODES=(args.val_background_mode=="random")
    if args.val_background_mode!="random":
        CrossViewPromptTokenGeneratorV10.DEFAULT_EVAL_BACKGROUND_MODE=args.val_background_mode
    prompt=CrossViewPromptTokenGeneratorV10(prompt_dim=int(adapter.sam.sam_prompt_embed_dim),image_embedding_size=int(adapter.sam.sam_image_embedding_size),
      hidden_dim=args.prompt_hidden_dim,num_tokens=args.prompt_num_tokens,dropout=.05,refinement_iters=1,work_size=args.prompt_work_size,
      prior_logit_scale=.35,prompt_input_channels=3,view_feature_dim=args.prompt_view_feature_dim,plane_spread=2.0,
      image_feature_mode="fpn",image_feature_proj_dim=args.prompt_image_feature_proj_dim,background_branch=True,background_tokens=1,
      background_gate_bias=-2.0,background_residual_scale=.5,semantic_enabled=True,semantic_dim=args.prompt_semantic_dim,
      semantic_clip_ckpt=args.prompt_semantic_clip_ckpt,semantic_prompt_template="A CT scan of the {}.",semantic_train_text_encoder=False,
      semantic_residual=True,semantic_residual_scale=1.0,prompt_3d_extra_channels=0,prompt_volume_tokens=args.prompt_volume_tokens).to(device)
    prompt.FULL_DEPTH_CHECKPOINT_THRESHOLD=max(
        0, int(args.prompt_checkpoint_depth_threshold)
    )
    return prompt

def prompt_state_for_compat_load(prompt_state):
    """Drop frozen external encoder weights and retain trainable PromptGen state."""
    return {
        key: value for key, value in prompt_state.items()
        if not key.startswith("semantic_encoder.clip_text_model.")
    }


def load_init(path,prompt,decoder):
    state=torch.load(path,map_location="cpu",weights_only=True)
    ps=prompt_state_for_compat_load(state.get("prompt_generator",state))
    incompatible=prompt.load_state_dict(ps,strict=False)
    allowed=("feedback_","foreground_sparse_residual_head.","background_sparse_residual_head.","dense_residual_head.","conflict_strength_gates.","mode_dense_projection.")
    bad=[k for k in incompatible.unexpected_keys if not k.startswith(allowed)]
    if bad: raise RuntimeError(f"Unsupported checkpoint keys: {bad[:20]}")
    missing_mode=[k for k in incompatible.missing_keys if k.startswith("mode_dense_projection.")]
    if missing_mode:
        # Early standalone v10 checkpoints omitted the v9 dense mode bias.
        # Restore only those absent fields from the trusted v9 warm start;
        # keep every already-trained v10 parameter from the resume checkpoint.
        v9_state=torch.load(DEFAULT_V9,map_location="cpu",weights_only=True)
        v9_prompt=v9_state.get("prompt_generator",v9_state)
        current=prompt.state_dict()
        restored=[]
        for key in missing_mode:
            value=v9_prompt.get(key)
            if value is not None and tuple(value.shape)==tuple(current[key].shape):
                current[key]=value
                restored.append(key)
        prompt.load_state_dict(current,strict=True)
        still_missing=sorted(set(missing_mode)-set(restored))
        if still_missing:
            raise RuntimeError(f"Cannot restore v9 prompt-mode fields: {still_missing}")
    # Standalone v10 checkpoints use ``mask_decoder`` while v9 checkpoints
    # use ``sam2_mask_decoder``. Never silently leave the trainable 3D adapter
    # randomly initialized when warm-starting from v9.
    ds=state.get("mask_decoder")
    if ds is None:
        ds=state.get("sam2_mask_decoder")
    if ds is None:
        raise RuntimeError(
            f"Checkpoint has no decoder state (expected mask_decoder or sam2_mask_decoder): {path}"
        )
    has_memory_adapter=any(key.startswith("prompt_memory_adapter.") for key in ds)
    incompatible=decoder.load_state_dict(ds,strict=has_memory_adapter)
    if not has_memory_adapter:
        bad_missing=[key for key in incompatible.missing_keys if not key.startswith("prompt_memory_adapter.")]
        if bad_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Incompatible decoder migration: missing={bad_missing} "
                f"unexpected={incompatible.unexpected_keys}"
            )
    return state

def save(path,epoch,prompt,decoder,opt,val,args,metrics=None,best_val=None,best_scribble=None):
    path=Path(path)
    temporary=path.with_name(f"{path.name}.tmp")
    state={"architecture":ARCH,"epoch":epoch,"val_dice":val,
      "best_val_dice":best_val,"best_scribble_dice":best_scribble,
      "validation_metrics":metrics,"prompt_generator":prompt.state_dict(),
      "validation_class_dice":getattr(args,"validation_class_dice",{}),
      "validation_class_difficulty":getattr(args,"validation_class_difficulty",{}),
      "mask_decoder":decoder.state_dict(),"optimizer":opt.state_dict(),"args":vars(args)}
    if _ACTIVE_SAM_TUNING is not None:
        tuning_sam,tuning_config=_ACTIVE_SAM_TUNING
        state["sam_tuning_state"]=sam_tuning_state_dict(tuning_sam,tuning_config)
    if _ACTIVE_EMA is not None:
        state["ema_state"]=_ACTIVE_EMA.state_dict()
    if _ACTIVE_PLATEAU is not None:
        state["plateau_state"]=_ACTIVE_PLATEAU.state_dict()
    # Never overwrite the last valid checkpoint with an interrupted write.
    # os.replace is atomic when the temporary file is in the same directory.
    torch.save(state,temporary)
    os.replace(temporary,path)
def scale_resumed_optimizer_learning_rates(optimizer, scale):
    scale=float(scale)
    if scale<=0:
        raise ValueError("resume LR scale must be positive")
    learning_rates=[]
    for group in optimizer.param_groups:
        group["lr"]=float(group["lr"])*scale
        learning_rates.append(group["lr"])
    return learning_rates


def worker(rank,world,port,stamp,args):
    global _ACTIVE_SAM_TUNING, _ACTIVE_EMA, _ACTIVE_PLATEAU
    _ACTIVE_PLATEAU=None
    os.environ.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE=str(world),MASTER_ADDR="127.0.0.1",MASTER_PORT=str(port))
    ctx=init_distributed(); device=torch.device(f"cuda:{ctx.local_rank}"); torch.cuda.set_device(device)
    _seed_process_rng(args.seed, rank)
    args.label_names=load_label_map(args.label_map)
    train_pairs=discover_totalseg_split_pairs(args.data_root,"train")
    val_pairs=discover_totalseg_split_pairs(args.data_root,"val",limit=5)
    adapter=SAM2MedicalAdapterV10(args,device)
    for p in adapter.sam.parameters(): p.requires_grad=False
    sam_tuning=None
    if str(getattr(args,"sam_tuning_mode","") or ""):
        sam_tuning=configure_sam_finetuning(adapter.sam,args.sam_tuning_mode)
        _ACTIVE_SAM_TUNING=(adapter.sam,sam_tuning)
    else:
        _ACTIVE_SAM_TUNING=None
    for p in adapter.sam.sam_mask_decoder.prompt_3d_adapter.parameters(): p.requires_grad=True
    for p in adapter.sam.sam_mask_decoder.prompt_memory_adapter.parameters(): p.requires_grad=True
    prompt=make_prompt(args,adapter,device)
    checkpoint_path=args.resume_checkpoint or args.prompt_generator_checkpoint
    checkpoint_state=load_init(checkpoint_path,prompt,adapter.sam.sam_mask_decoder)
    sam_tuning_loaded=False
    if sam_tuning is not None:
        sam_tuning_loaded=load_sam_tuning_from_checkpoint(
            adapter.sam,sam_tuning,checkpoint_state,
            required=bool(args.resume_checkpoint),
        )
    previous_metrics=checkpoint_state.get("validation_metrics") or {}
    args.validation_class_dice=dict(
        checkpoint_state.get("validation_class_dice")
        or (previous_metrics.get("per_class") or {}).get("dice") or {}
    )
    args.validation_class_difficulty=dict(
        checkpoint_state.get("validation_class_difficulty") or {}
    )
    if not args.validation_class_dice and not args.validation_class_difficulty:
        v9_reference=torch.load(DEFAULT_V9,map_location="cpu",weights_only=True)
        args.validation_class_difficulty=dict(v9_reference.get("class_weights") or {})
    CrossViewPromptTokenGeneratorV10.DEFAULT_EVAL_RANDOM_PROMPT_MODES=(args.val_background_mode=="random")
    if args.val_background_mode!="random":
        CrossViewPromptTokenGeneratorV10.DEFAULT_EVAL_BACKGROUND_MODE=args.val_background_mode
    train_step=DDPTrainingStepV10(prompt,adapter).to(device)
    if ctx.enabled:
        train_step=DDP(train_step,device_ids=[ctx.local_rank],find_unused_parameters=True)
    context_prompt=[
        p for name,p in prompt.named_parameters()
        if name.startswith("v10_3_context_fusion.")
    ]
    base_prompt=[
        p for name,p in prompt.named_parameters()
        if not name.startswith("mode_dense_projection.")
        and not name.startswith("v10_3_context_fusion.")
    ]
    mode_prompt=[p for name,p in prompt.named_parameters() if name.startswith("mode_dense_projection.")]
    groups=[{"params":base_prompt,"lr":args.prompt_lr,"name":"prompt_generator"},
            {"params":adapter.sam.sam_mask_decoder.prompt_3d_adapter.parameters(),"lr":args.adapter_lr,"name":"prompt_3d_adapter"},
             {"params":adapter.sam.sam_mask_decoder.prompt_memory_adapter.parameters(),"lr":args.memory_adapter_lr,"name":"prompt_memory_adapter"}]
    if context_prompt:
        groups.append({
            "params":context_prompt,
            "lr":float(getattr(args,"v10_3_context_lr",args.prompt_lr)),
            "name":"v10_3_global_context",
        })
    if sam_tuning is not None:
        groups.extend(sam_tuning_optimizer_groups(
            adapter.sam,sam_tuning,args.encoder_lr,args.decoder_lr,
            decoder_transformer_lr=args.decoder_transformer_lr,
            decoder_head_lr=args.decoder_head_lr,
        ))
        groups.append({"params":mode_prompt,"lr":args.prompt_lr,"name":"prompt_mode"})
    opt=AdamW(groups,weight_decay=args.weight_decay)
    ema=(TrainableParameterEMA({"prompt":prompt,"sam":adapter.sam},args.ema_decay)
         if float(args.ema_decay)>0.0 else None)
    _ACTIVE_EMA=ema
    start_epoch=1; best=-1.0; best_scribble=-1.0
    if args.resume_checkpoint:
        if "optimizer" not in checkpoint_state:
            raise RuntimeError(f"Resume checkpoint has no optimizer state: {args.resume_checkpoint}")
        optimizer_state=checkpoint_state["optimizer"]
        if sam_tuning is not None:
            expected=len(opt.param_groups)
            actual=len(optimizer_state.get("param_groups",()))
            if actual!=expected:
                raise RuntimeError(f"Tuned optimizer group mismatch: checkpoint={actual} runtime={expected}")
            opt.load_state_dict(optimizer_state)
        elif len(optimizer_state.get("param_groups",()))==3:
            opt.load_state_dict(optimizer_state)
            opt.add_param_group({"params":mode_prompt,"lr":args.prompt_lr,"name":"prompt_mode"})
        else:
            opt.add_param_group({"params":mode_prompt,"lr":args.prompt_lr,"name":"prompt_mode"})
            opt.load_state_dict(optimizer_state)
        resumed_learning_rates=scale_resumed_optimizer_learning_rates(
            opt,args.resume_lr_scale
        )
        if ema is not None and checkpoint_state.get("ema_state") is not None:
            ema.load_state_dict(checkpoint_state["ema_state"])
        start_epoch=int(checkpoint_state.get("epoch",0))+1
        stored_best=checkpoint_state.get("best_val_dice")
        stored_scribble=checkpoint_state.get("best_scribble_dice")
        if stored_best is not None:
            best=float(stored_best)
        if stored_scribble is not None:
            best_scribble=float(stored_scribble)
        # Older v10 last.pth files did not store best_val_dice. Recover it
        # from the sibling best checkpoint without changing model weights.
        sibling_best=Path(args.resume_checkpoint).resolve().parent/"best.pth"
        if best<0 and sibling_best.is_file():
            best_state=torch.load(sibling_best,map_location="cpu",weights_only=True)
            previous_val=best_state.get("val_dice")
            if previous_val is not None:
                best=float(previous_val)
        run=(Path(args.out_dir)/stamp if args.resume_new_run
             else Path(args.resume_checkpoint).resolve().parent)
    else:
        if sam_tuning is None:
            opt.add_param_group({"params":mode_prompt,"lr":args.prompt_lr,"name":"prompt_mode"})
        run=Path(args.out_dir)/stamp
    if ctx.is_main:
        run.mkdir(parents=True,exist_ok=True)
        formatter=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        console=logging.StreamHandler(sys.stdout); console.setFormatter(formatter)
        file_handler=logging.FileHandler(run/"finetune.log",encoding="utf-8"); file_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.INFO,handlers=[console,file_handler],force=True)
        logging.info("Run directory: %s",run)
        logging.info("Validation cases are fixed: %s",[str(pair[0]) for pair in val_pairs])
        logging.info("Validation protocol: seed=%d max_tasks=%d fg=%s bg=%s",
                     args.validation_seed,args.validation_max_tasks,args.val_foreground_mode,args.val_background_mode)
        logging.info("Validation difficult-class reference covers %d classes",
                     len(args.validation_class_difficulty or args.validation_class_dice))
        logging.info(
            "Distributed training: world_size=%d visible_gpu_ids=%s "
            "global_cases_per_epoch=%d cases_per_rank=%d",
            ctx.world_size,args.gpu,args.train_cases_per_epoch,
            int(args.train_cases_per_epoch)//int(ctx.world_size),
        )
        if sam_tuning is not None:
            logging.info(
                "SAM tuning mode=%s encoder_tensors=%d decoder_tensors=%d encoder_lr=%.3g decoder_lr=%.3g",
                sam_tuning.mode,len(sam_tuning.encoder_names),len(sam_tuning.decoder_names),
                args.encoder_lr,args.decoder_lr,
            )
            logging.info(
                "SAM tuning checkpoint state loaded=%s source=%s",
                sam_tuning_loaded, checkpoint_path,
            )
        if args.resume_checkpoint:
            logging.info("Resumed from %s | start_epoch=%d best_val_dice=%.6f",
                         args.resume_checkpoint,start_epoch,best)
            logging.info("Resume LR scale=%.6g | learning_rates=%s",
                         args.resume_lr_scale,resumed_learning_rates)
        logging.info("EMA decay=%.6g enabled=%s updates=%d",
                     args.ema_decay,ema is not None,0 if ema is None else ema.num_updates)
        logging.info(
            "Gradient accumulation steps=%d optimizer_steps_per_epoch=%d",
            args.train_gradient_accumulation_steps,
            (int(args.train_cases_per_epoch)//int(ctx.world_size)
             + int(args.train_gradient_accumulation_steps) - 1)
            // int(args.train_gradient_accumulation_steps),
        )
        context_fusion=getattr(prompt,"v10_3_context_fusion",None)
        if context_fusion is not None:
            logging.info(
                "v10.3 dual-scale context enabled=%s thumbnail=%d hidden=%d "
                "context_lr=%.3g initial_gate=%.6f",
                bool(getattr(prompt,"v10_3_global_context_enabled",False)),
                int(getattr(args,"v10_3_global_context_size",32)),
                int(getattr(args,"v10_3_context_hidden_dim",64)),
                float(getattr(args,"v10_3_context_lr",args.prompt_lr)),
                float(torch.tanh(context_fusion.residual_gate).detach().cpu()),
            )
    dataset=make_dataset(train_pairs,args,return_target_class=True)
    if args.validate_before_train:
        torch.cuda.empty_cache(); args.dataset_foreground_mode=args.val_foreground_mode
        with (ema.average_parameters() if ema is not None else nullcontext()):
            baseline,_,baseline_metrics=evaluate_memory_v10(prompt,adapter,val_pairs,list(args.label_names),args,device)
        args.dataset_foreground_mode="random"
        if ctx.is_main:
            logging.info("Baseline val_dice=%.4f metrics=%s performance=%s",baseline,baseline_metrics["mean"],baseline_metrics["performance"])
            logging.info("Baseline validation_protocol=%s",baseline_metrics["validation_protocol"])
            for threshold,values in baseline_metrics["per_threshold"].items():
                logging.info("Baseline threshold=%s | %s",threshold,values)
            for mode,values in baseline_metrics["per_prompt_mode"].items():
                logging.info("Baseline prompt_mode %s | %s",mode,values)
        best=baseline if baseline is not None else -1.0
        best_scribble=float(baseline_metrics.get("foreground_class_macro_dice",{}).get("scribble",-1.0))
    plateau=None
    if args.plateau_early_stop_patience:
        plateau_state=checkpoint_state.get("plateau_state") if args.resume_checkpoint else None
        if args.plateau_reset_on_resume and plateau_state is None:
            raise ValueError("--plateau-reset-on-resume requires a resume checkpoint with plateau_state")
        plateau=(ValidationPlateau.restart_with_current_best(
                     plateau_state,best=best,min_delta=args.plateau_min_delta)
                 if args.plateau_reset_on_resume else
                 ValidationPlateau.from_state_dict(plateau_state) if plateau_state is not None
                 else ValidationPlateau(
                     best=best,min_delta=args.plateau_min_delta,
                     lr_patience=args.plateau_lr_patience,
                     stop_patience=args.plateau_early_stop_patience,
                     factor=args.plateau_lr_factor,min_lr_ratio=args.plateau_min_lr_ratio,
                 ))
        _ACTIVE_PLATEAU=plateau
        if ctx.is_main:
            logging.info("Validation plateau control: %s",plateau.state_dict())
    for epoch in range(start_epoch,args.epochs+1):
        learning_rates=apply_staged_learning_rates(
            opt,epoch=epoch,total_epochs=int(getattr(args,"lr_schedule_epochs",0) or args.epochs),
            warmup_epochs=args.lr_warmup_epochs,
            cosine_min_ratio=args.lr_cosine_min_ratio,
            encoder_unfreeze_epoch=args.sam_encoder_unfreeze_epoch,
        )
        encoder_frozen=(
            sam_tuning is not None
            and epoch<int(args.sam_encoder_unfreeze_epoch)
        )
        if ctx.is_main:
            logging.info(
                "Epoch %d optimizer schedule | encoder_frozen=%s learning_rates=%s",
                epoch,encoder_frozen,learning_rates,
            )
        indices=_balanced_epoch_indices(dataset,args,ctx,epoch)
        loader=_make_training_loader(dataset,indices,args)
        train_step.train(); total=0.0; epoch_frames=0
        torch.cuda.reset_peak_memory_stats(device); epoch_start=time.perf_counter()
        progress=tqdm(loader,desc=f"Train v10 epoch {epoch:03d}",leave=False,
                      disable=not ctx.is_main,dynamic_ncols=True)
        accumulation_steps=max(1,int(args.train_gradient_accumulation_steps))
        total_steps=len(loader)
        opt.zero_grad(set_to_none=True)
        for step,batch in enumerate(progress,1):
            accumulation_group_size,accumulation_boundary=gradient_accumulation_window(
                step,total_steps,accumulation_steps
            )
            sync_context=(
                train_step.no_sync()
                if ctx.enabled and not accumulation_boundary
                else nullcontext()
            )
            with sync_context:
                loss,batch_frames=train_step(batch,args,device)
                if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss: {loss}")
                loss_for_backward=loss/float(accumulation_group_size)
                loss_for_backward.backward()
            if encoder_frozen:
                clear_optimizer_group_gradients(opt,"sam_image_encoder")
            if accumulation_boundary:
                torch.nn.utils.clip_grad_norm_(train_step.parameters(),args.grad_clip_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None: ema.update()
            total+=float(loss)
            epoch_frames+=int(batch_frames)
            if ctx.is_main:
                progress.set_postfix(loss=f"{total/step:.4f}",frames=epoch_frames)
        epoch_wall=time.perf_counter()-epoch_start
        local_peak=torch.cuda.max_memory_allocated(device)/1024**3
        sum_stats=torch.tensor(
            [total,float(len(loader)),float(epoch_frames)],
            dtype=torch.float64,device=device,
        )
        max_stats=torch.tensor(
            [epoch_wall,local_peak],dtype=torch.float64,device=device,
        )
        if ctx.enabled:
            dist.all_reduce(sum_stats,op=dist.ReduceOp.SUM)
            dist.all_reduce(max_stats,op=dist.ReduceOp.MAX)
        global_loss_sum,global_cases,global_frames=sum_stats.tolist()
        global_wall,max_peak=max_stats.tolist()
        validation_due=(
            (epoch%args.validate_every==0 or epoch==args.epochs)
            and not bool(getattr(args,"v10_2_poscap_skip_validation",False))
        )
        if ctx.is_main:
            logging.info(
                "Epoch %d loss=%.4f train_wall=%.2fs cases_global=%d "
                "cases_per_rank=%d frames_global=%d max_peak_alloc=%.2fGB",
                epoch,global_loss_sum/max(1.0,global_cases),global_wall,
                int(global_cases),len(loader),int(global_frames),max_peak,
            )
            context_fusion=getattr(prompt,"v10_3_context_fusion",None)
            if context_fusion is not None:
                logging.info(
                    "Epoch %d v10.3_context_gate=%.6f",
                    epoch,
                    float(torch.tanh(context_fusion.residual_gate).detach().cpu()),
                )
            if not validation_due:
                save(run/"last.pth",epoch,prompt,adapter.sam.sam_mask_decoder,opt,None,args,
                     best_val=best,best_scribble=best_scribble)
        if validation_due:
            stop_training=False
            torch.cuda.empty_cache()
            args.dataset_foreground_mode=args.val_foreground_mode
            with (ema.average_parameters() if ema is not None else nullcontext()):
                val,_,metrics=evaluate_memory_v10(prompt,adapter,val_pairs,list(args.label_names),args,device)
                if plateau is not None:
                    reduced,stop_training=plateau.observe(val,opt.param_groups)
                    if ctx.is_main:
                        logging.info(
                            "Epoch %d plateau: score=%.6f best=%.6f bad=%d/%d "
                            "lr_reduced=%s stop=%s rates=%s",
                            epoch,val,plateau.best,plateau.bad_validations,
                            plateau.stop_patience,reduced,stop_training,
                            [group["lr"] for group in opt.param_groups],
                        )
                args.dataset_foreground_mode="random"
                if ctx.is_main:
                    logging.info("Epoch %d val_dice=%.4f metrics=%s performance=%s",epoch,val,metrics["mean"],metrics["performance"])
                    logging.info("Epoch %d validation_protocol=%s",epoch,metrics["validation_protocol"])
                    for threshold,values in metrics["per_threshold"].items():
                        logging.info("Epoch %d threshold=%s | %s",epoch,threshold,values)
                    for mode,values in metrics["per_prompt_mode"].items():
                        logging.info("Epoch %d prompt_mode %s | %s",epoch,mode,values)
                    scribble=float(metrics.get("foreground_class_macro_dice",{}).get("scribble",-1.0))
                    is_best=val is not None and val>best
                    is_best_scribble=scribble>best_scribble
                    if is_best:
                        best=float(val)
                    if is_best_scribble:
                        best_scribble=scribble
                    if is_best:
                        save(run/"best.pth",epoch,prompt,adapter.sam.sam_mask_decoder,opt,val,args,
                             metrics=metrics,best_val=best,best_scribble=best_scribble)
                        named=run/f"epoch{epoch:03d}_dice{val:.5f}.pth"
                        save(named,epoch,prompt,adapter.sam.sam_mask_decoder,opt,val,args,
                             metrics=metrics,best_val=best,best_scribble=best_scribble)
                        logging.info("New best val Dice %.6f; saved %s and %s",val,run/"best.pth",named)
                    if is_best_scribble:
                        save(run/"best_scribble.pth",epoch,prompt,adapter.sam.sam_mask_decoder,opt,val,args,
                             metrics=metrics,best_val=best,best_scribble=best_scribble)
                        logging.info("New best foreground-scribble class-macro Dice %.6f; saved %s",
                                     scribble,run/"best_scribble.pth")
                    validation_checkpoint=run/f"epoch{epoch:03d}.pth"
                    save(validation_checkpoint,epoch,prompt,adapter.sam.sam_mask_decoder,opt,val,args,
                         metrics=metrics,best_val=best,best_scribble=best_scribble)
                    logging.info("Saved validation checkpoint %s",validation_checkpoint)
            args.dataset_foreground_mode="random"
            if ctx.is_main:
                # last.pth always contains live training weights; EMA best files
                # above contain the averaged weights used for their validation.
                save(run/"last.pth",epoch,prompt,adapter.sam.sam_mask_decoder,opt,val,args,
                     metrics=metrics,best_val=best,best_scribble=best_scribble)
            if plateau is not None:
                if ctx.enabled:
                    stop_flag=torch.tensor(int(stop_training if ctx.is_main else 0),device=device)
                    dist.broadcast(stop_flag,src=0)
                    stop_training=bool(stop_flag.item())
                if stop_training:
                    if ctx.is_main:
                        logging.info("Early stop after epoch %d: %d consecutive validations without improvement",epoch,plateau.bad_validations)
                    break
    destroy_distributed()

def main():
    args=parser().parse_args(); args.amp=True; args.object_score_gate=False
    ids=parse_gpu_ids(args.gpu); stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(ids)>1: mp.spawn(worker,nprocs=len(ids),args=(len(ids),find_free_port(),stamp,args),join=True)
    else: worker(0,1,find_free_port(),stamp,args)
if __name__=="__main__": main()
