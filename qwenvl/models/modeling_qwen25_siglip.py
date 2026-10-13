# Copyright 2025 Lin Chen. All rights reserved.
# This code is based on the implementation of Qwen2.5-VL
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging

# add by Lin Chen
from .configuration_qwen25_siglip import Qwen25_SigLIPConfig
from .modeling_utils import Qwen25_SigLIPPreTrainedModel
from .modeling_llm_eagar import Qwen25_SigLIPTextModel
from .modeling_vision import Siglip2VisionTransformerPretrainedModel
from qwenvl.data.multimodal_rope import get_multimodal_position_ids


logger = logging.get_logger(__name__)


class VisualEntityTagger(nn.Module):
    """Add an optional content-space identity tag to later visual entities.

    Entity ids are one-based.  Entity 1 (the only image in the common
    single-image case) is deliberately the exact zero residual.  A scalar,
    zero-initialized gate makes a newly created tagger functionally identical
    to the checkpoint from which it is initialized; training can enable the
    extra identity signal only when it lowers the objective.
    """

    def __init__(self, hidden_size: int, num_entities: int = 16, gate_init: float = 0.0):
        super().__init__()
        if num_entities < 2:
            raise ValueError("num_entities must be at least two")
        self.num_entities = int(num_entities)
        self.embedding = nn.Embedding(self.num_entities - 1, hidden_size)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self, inputs_embeds: torch.Tensor, visual_entity_ids: torch.Tensor
    ) -> torch.Tensor:
        if visual_entity_ids.shape != inputs_embeds.shape[:2]:
            raise ValueError("visual_entity_ids must match inputs_embeds [batch, sequence]")
        later_entities = visual_entity_ids > 1
        if not bool(later_entities.any()):
            return inputs_embeds
        max_entity = int(visual_entity_ids[later_entities].max().item())
        if max_entity > self.num_entities:
            raise ValueError(
                f"visual entity id {max_entity} exceeds configured maximum {self.num_entities}"
            )
        # Entity 2 maps to row 0. Entity 1 is excluded above and receives the
        # exact zero residual rather than a learned special case.
        indices = (visual_entity_ids.clamp_min(2).long() - 2).clamp_max(
            self.num_entities - 2
        )
        tags = self.embedding(indices).to(inputs_embeds.dtype)
        residual = torch.tanh(self.gate).to(inputs_embeds.dtype) * tags
        return inputs_embeds + residual * later_entities.unsqueeze(-1).to(inputs_embeds.dtype)


def _init_visual_entity_tagger(module: nn.Module, config) -> None:
    """Initialize a newly added tagger, including low-memory checkpoint loads.

    ``from_pretrained`` may construct missing modules on the meta device. The
    default PyTorch initialization performed in ``VisualEntityTagger.__init__``
    is then discarded when the missing tensors are materialized. Providing an
    explicit Transformers initialization hook prevents an uninitialized gate
    when an old GIRF checkpoint is upgraded to entity-tagged-girf.
    """
    if not isinstance(module, VisualEntityTagger):
        return
    initializer_range = getattr(config.text_config, "initializer_range", 0.02)
    gate_init = getattr(config, "entity_tag_gate_init", 0.0)
    module.embedding.weight.data.normal_(mean=0.0, std=initializer_range)
    module.gate.data.fill_(float(gate_init))


def should_build_visual_entity_metadata(
    input_ids: Optional[torch.LongTensor],
    pixel_values: Optional[torch.Tensor],
    pixel_values_videos: Optional[torch.Tensor],
    image_grid_thw: Optional[torch.LongTensor],
    video_grid_thw: Optional[torch.LongTensor],
) -> bool:
    """Return whether this forward pass contains visual prefill inputs."""
    return input_ids is not None and (
        (pixel_values is not None and image_grid_thw is not None)
        or (pixel_values_videos is not None and video_grid_thw is not None)
    )


def build_visual_entity_metadata(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor],
    video_grid_thw: Optional[torch.LongTensor],
    image_token_id: int,
    video_token_id: int,
    spatial_merge_size: int,
    packed_seq_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map processor visual runs to media-level entity and temporal ids.

    Qwen3-VL expands one video into ``T`` distinct ``video_pad`` runs with
    timestamp text between them.  ``video_grid_thw[:, 0]`` is therefore the
    authoritative grouping metadata: the next T video runs share one entity
    and receive temporal ids 0..T-1.  Images consume one run and use Z=0.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [batch, sequence]")
    entity_ids = torch.zeros_like(input_ids, dtype=torch.float32)
    temporal_ids = torch.zeros_like(input_ids, dtype=torch.float32)
    image_grids = [] if image_grid_thw is None else image_grid_thw.tolist()
    video_grids = [] if video_grid_thw is None else video_grid_thw.tolist()
    image_cursor = video_cursor = 0

    if packed_seq_lens is not None:
        if input_ids.shape[0] != 1:
            raise ValueError("packed sequence metadata expects a flattened batch of size one")
        boundaries = [int(value) for value in packed_seq_lens.tolist()]
        sample_ranges = [[(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]]
    else:
        sample_ranges = [[(0, input_ids.shape[1])] for _ in range(input_ids.shape[0])]

    def visual_runs(tokens: torch.Tensor, start: int, end: int):
        visual = (tokens[start:end] == image_token_id) | (tokens[start:end] == video_token_id)
        indices = torch.nonzero(visual, as_tuple=False).flatten() + start
        if indices.numel() == 0:
            return []
        splits = torch.nonzero(indices[1:] != indices[:-1] + 1, as_tuple=False).flatten() + 1
        return list(torch.tensor_split(indices, splits.cpu().tolist()))

    for batch_idx, ranges in enumerate(sample_ranges):
        for sample_start, sample_end in ranges:
            runs = visual_runs(input_ids[batch_idx], sample_start, sample_end)
            run_cursor = 0
            entity = 1
            while run_cursor < len(runs):
                run = runs[run_cursor]
                token_id = int(input_ids[batch_idx, run[0]])
                if token_id == image_token_id:
                    if image_cursor >= len(image_grids):
                        raise ValueError("image token run has no matching image_grid_thw row")
                    t, h, w = (int(value) for value in image_grids[image_cursor])
                    expected = t * h * w // (spatial_merge_size**2)
                    if run.numel() != expected:
                        raise ValueError(
                            f"image entity token count {run.numel()} does not match grid expectation {expected}"
                        )
                    entity_ids[batch_idx, run] = float(entity)
                    image_cursor += 1
                    run_cursor += 1
                elif token_id == video_token_id:
                    if video_cursor >= len(video_grids):
                        raise ValueError("video token run has no matching video_grid_thw row")
                    t, h, w = (int(value) for value in video_grids[video_cursor])
                    expected_per_time = h * w // (spatial_merge_size**2)
                    if run_cursor + t > len(runs):
                        raise ValueError("video_grid_thw requests more temporal runs than input_ids contain")
                    for time_index in range(t):
                        time_run = runs[run_cursor + time_index]
                        if int(input_ids[batch_idx, time_run[0]]) != video_token_id:
                            raise ValueError("a video entity is interrupted by a non-video visual run")
                        if time_run.numel() != expected_per_time:
                            raise ValueError(
                                f"video time token count {time_run.numel()} does not match grid expectation "
                                f"{expected_per_time}"
                            )
                        entity_ids[batch_idx, time_run] = float(entity)
                        temporal_ids[batch_idx, time_run] = float(time_index)
                    video_cursor += 1
                    run_cursor += t
                else:
                    raise RuntimeError("visual run contains an unknown placeholder token")
                entity += 1

    if image_cursor != len(image_grids):
        raise ValueError(f"unused image_grid_thw rows: {len(image_grids) - image_cursor}")
    if video_cursor != len(video_grids):
        raise ValueError(f"unused video_grid_thw rows: {len(video_grids) - video_cursor}")
    return entity_ids, temporal_ids


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class Qwen25_SigLIPModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None



@auto_docstring
class Qwen25_SigLIPModel(Qwen25_SigLIPPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen25_SigLIPConfig
    _no_split_modules = ["Qwen25_SigLIPDecoderLayer", "Siglip2EncoderLayer"]

    def _init_weights(self, module):
        _init_visual_entity_tagger(module, self.config)

    def __init__(self, config):
        super().__init__(config)
        self.visual = Siglip2VisionTransformerPretrainedModel(config.vision_config)
        self.language_model = Qwen25_SigLIPTextModel(config.text_config)
        self.visual_entity_tagger = None
        if getattr(config, "cross_modal_position_type", "dipe") == "entity-tagged-girf":
            self.visual_entity_tagger = VisualEntityTagger(
                hidden_size=config.text_config.hidden_size,
                num_entities=getattr(config, "entity_tag_num_embeddings", 16),
                gate_init=getattr(config, "entity_tag_gate_init", 0.0),
            )
        self.rope_deltas = None  # cache rope_deltas here

        # add by capychen, cache for inference
        self.visual_mask = None
        self.text_mask = None
        self.valid_mask = None
        self.position_ids = None
        self.static_position_ids = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    # keep the code consistent with data/rope2d.py, this function will be used during inference
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return get_multimodal_position_ids(
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            position_embedding_type=self.config.position_embedding_type,
            position_embedding_options=self.config.position_embedding_options,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
        )


    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        static_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen25_SigLIPModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        # import pdb; pdb.set_trace()
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)        

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # from qwen3-vl to process masks
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
        elif video_mask is not None:
            video_mask = video_mask[..., 0]

        if input_ids is not None:
            B, L = input_ids.shape
            device = input_ids.device
        elif position_ids is not None:
            B, L = position_ids.shape
            device = position_ids.device
        else:
            raise ValueError("You have to specify either input_ids or position_ids")

        packed_seq_lens = None
        if attention_mask is not None and attention_mask.ndim == 1:
            # FlattenedDataCollator passes cumulative sequence lengths here.
            packed_seq_lens = attention_mask

        visual_entity_ids = None
        visual_temporal_ids = None
        if (
            (
                getattr(self.config, "cross_modal_position_type", "dipe")
                in {
                "entity-aware-girf", "entity-tagged-girf",
                "tv-neutral-girf", "pi-girf",
                }
                or bool(getattr(self.config, "girf_pi_mask_enabled", False))
            )
            # Transformers keeps grid metadata in model_kwargs during cached
            # decoding, while prepare_inputs_for_generation deliberately drops
            # pixel inputs after prefill.  Building media entities from a
            # decode-only input_ids slice would therefore see grids but no
            # visual placeholder runs.  Entity metadata is needed only on the
            # visual prefill where the corresponding pixel inputs are present.
            and should_build_visual_entity_metadata(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
        ):
            visual_entity_ids, visual_temporal_ids = build_visual_entity_metadata(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_token_id=self.config.image_token_id,
                video_token_id=self.config.video_token_id,
                spatial_merge_size=self.visual.spatial_merge_size,
                packed_seq_lens=packed_seq_lens,
            )

        if self.visual_entity_tagger is not None and visual_entity_ids is not None:
            inputs_embeds = self.visual_entity_tagger(inputs_embeds, visual_entity_ids)

        # calc visual_mask
        if image_mask is not None and video_mask is not None:
            visual_mask = image_mask | video_mask
        elif image_mask is not None:
            visual_mask = image_mask
        elif video_mask is not None:
            visual_mask = video_mask
        else:
            visual_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        
        if B == 1:
            valid_mask = torch.ones((B, L), dtype=torch.bool, device=device)
        else:
            assert input_ids is not None, "input_ids must be specified when B > 1"
            # valid_mask = input_ids != self.config.text_config.bos_token_id
            # valid_mask = input_ids != self.config.text_config.pad_token_id
            valid_mask = input_ids != 151643
        text_mask = ~visual_mask & valid_mask

        if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
        
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, static_position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask,
                )

                # save for decoding
                self.rope_deltas = rope_deltas
                self.position_ids = position_ids
                self.static_position_ids = static_position_ids
                self.visual_mask = visual_mask
                self.text_mask = text_mask
                self.valid_mask = valid_mask
                self.visual_entity_ids = visual_entity_ids
                self.visual_temporal_ids = visual_temporal_ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids = position_ids + delta.to(position_ids.device)
                position_ids = torch.cat([self.position_ids, position_ids], dim=2)

                static_position_ids = self.static_position_ids[:, :, -1:]
                static_position_ids = torch.cat([self.static_position_ids, static_position_ids], dim=2)
                visual_mask = torch.cat([self.visual_mask, visual_mask], dim=1)
                text_mask = torch.cat([self.text_mask, text_mask], dim=1)
                valid_mask = torch.cat([self.valid_mask, valid_mask], dim=1)
                if getattr(self, "visual_entity_ids", None) is not None:
                    entity_tail = torch.zeros_like(visual_mask[:, -inputs_embeds.shape[1] :], dtype=torch.float32)
                    temporal_tail = torch.zeros_like(entity_tail)
                    visual_entity_ids = torch.cat([self.visual_entity_ids, entity_tail], dim=1)
                    visual_temporal_ids = torch.cat([self.visual_temporal_ids, temporal_tail], dim=1)
                self.visual_mask = visual_mask
                self.text_mask = text_mask
                self.valid_mask = valid_mask
                self.position_ids = position_ids
                self.static_position_ids = static_position_ids
                self.visual_entity_ids = visual_entity_ids
                self.visual_temporal_ids = visual_temporal_ids
        
        # import pdb; pdb.set_trace()
        # print('mean', inputs_embeds[text_mask].mean().item(), inputs_embeds[visual_mask].mean().item())
        # print('std', inputs_embeds[text_mask].std().item(), inputs_embeds[visual_mask].std().item())
        
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,

            # args for DIPE
            static_position_ids=static_position_ids,
            visual_mask=visual_mask,
            text_mask=text_mask,
            valid_mask=valid_mask,
            packed_seq_lens=packed_seq_lens,
            visual_entity_ids=visual_entity_ids,
            visual_temporal_ids=visual_temporal_ids,

            **kwargs,
        )

        output = Qwen25_SigLIPModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Qwen25_SigLIP causal language model (or autoregressive) outputs.
    """
)
class Qwen25_SigLIPCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen25_SigLIPForConditionalGeneration(Qwen25_SigLIPPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False

    def _init_weights(self, module):
        _init_visual_entity_tagger(module, self.config)

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen25_SigLIPModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @property
    def visual_entity_tagger(self):
        """Expose the multimodal entity adapter like ``visual``/``language_model``."""
        return self.model.visual_entity_tagger

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        static_position_ids: Optional[torch.LongTensor] = None, 
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen25_SigLIPCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen25_SigLIPForConditionalGeneration

        >>> model = Qwen25_SigLIPForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            static_position_ids=static_position_ids, # capychen add
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Qwen25_SigLIPCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # adapt from Qwen3VL
        # position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = ["Qwen25_SigLIPForConditionalGeneration", "Qwen25_SigLIPModel"]
