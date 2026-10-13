import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    llm_path: Optional[str] = field(default=None)
    visual_encoder_path: Optional[str] = field(default=None)
    processor_path: Optional[str] = field(default=None)
    llm_backbone: str = field(
        default="auto",
        metadata={
            "help": "Text-decoder family: auto, qwen2, or qwen3. "
                    "qwen3 preserves Q/K RMSNorm when importing Qwen3 weights."
        },
    )
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    position_embedding_type: str = field(
        default="mrope",
        metadata={"help": "Multimodal position embedding name."},
    )
    position_embedding_options: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSON object overriding position embedding defaults."},
    )
    cross_modal_position_type: str = field(
        default="dipe",
        metadata={
            "help": "Cross-modal attention rule: dipe, native, universal-relation, bounded-dipe, girf, entity-aware-girf, entity-tagged-girf, tv-neutral-girf, or pi-girf."
        },
    )
    entity_aware_girf_spatial_scales: str = field(
        default="[16, 16]",
        metadata={"help": "JSON list of fixed Y/X scales for simplified Entity-aware GIRF V3."},
    )
    entity_tag_num_embeddings: int = field(
        default=16,
        metadata={"help": "Maximum visual entities for entity-tagged-girf."},
    )
    entity_tag_gate_init: float = field(
        default=0.0,
        metadata={"help": "Initial residual gate for entity-tagged-girf."},
    )
    llm_trainable_last_n_layers: int = field(
        default=-1,
        metadata={
            "help": "When tune_mm_llm is true, train only the last N decoder layers; -1 keeps all LLM parameters trainable."
        },
    )
    attention_chunk_elements: int = field(
        default=8388608,
        metadata={"help": "Maximum temporary attention score elements per query chunk."},
    )
    attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "Transformers attention backend: eager, sdpa, or flash_attention_2."
        },
    )
    girf_ordinal_enabled: bool = field(
        default=True,
        metadata={"help": "Keep the legacy GIRF ordinal entity phase enabled."},
    )
    girf_pi_mask_enabled: bool = field(
        default=False,
        metadata={"help": "Enable the strict PI pair mask. Stage-1 paired training keeps this false."},
    )
    girf_pi_mask_components: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional JSON list selecting PI mask clauses. Null keeps the historical full PI mask."
        },
    )
    girf_pi_soft_vv_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Finite penalty subtracted from cross-entity visual-to-visual attention logits."
        },
    )

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    enforce_model_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Reject multimodal samples whose expanded token sequence exceeds "
                "TrainingArguments.model_max_length. This avoids unsafe truncation "
                "through an image-token block."
            )
        },
    )
    base_interval: int = field(default=2)
    max_pixels: int = field(default=32 * 32 * 1280)
    min_pixels: int = field(default=32 * 32 * 4)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=32 * 32 * 1280)
    video_min_pixels: int = field(default=32 * 32 * 4)
    video_fps: float = 2
    niah_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Optional grouped JSONL manifest for paired NIAH training."},
    )
    niah_grid_side: int = field(
        default=12,
        metadata={"help": "Visual-token grid side after spatial merging for paired NIAH."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    entity_tag_lr: Optional[float] = None
    niah_worst_weight: float = field(default=0.5)
    niah_js_weight: float = field(default=0.05)
    niah_worst_temperature: float = field(default=0.2)
    niah_gap_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for paired correct-answer margin-gap loss."},
    )
    niah_gap_tolerance: float = field(
        default=0.0,
        metadata={"help": "Allowed absolute paired answer-margin difference before penalty."},
    )

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
