"""PyTorch Personaplex model."""

import torch

from ...configuration_utils import PreTrainedConfig
from ..auto.configuration_auto import AutoConfig
from ..moshi.configuration_moshi import MoshiConfig, MoshiDepthConfig
from ..moshi.modeling_moshi import (
    MoshiForCausalLM,
    MoshiForConditionalGeneration,
    MoshiModel,
    MoshiUnconditionalInput,
)


# PersonaPlex prefill constants (models_lm.py.patch): per-frame constant Mimi tokens for
# digital silence / a 440 Hz sine, precomputed for the 8-codebook moshiko Mimi. The original
# tiles these instead of encoding real waveforms so the prefill is reproducible frame-for-frame.
SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
SINE_TOKENS = [430, 1268, 381, 1611, 1095, 1495, 56, 472]
ZERO_TEXT_CODE = 3


class PersonaplexUnconditionalInput(MoshiUnconditionalInput):
    pass


class PersonaplexDepthConfig(MoshiDepthConfig):
    # PersonaPlex trains its depth decoder with dep_q=16: it predicts both the agent
    # stream (slices 0..7) and the user stream (slices 8..15), while the audio streams
    # themselves keep 8 codebooks each. 17 positions = 1 text token + 16 audio tokens.
    num_codebooks: int = 16
    max_position_embeddings: int = 17


class PersonaplexConfig(MoshiConfig):
    def __post_init__(self, **kwargs):
        self.num_key_value_heads = (
            self.num_key_value_heads if self.num_key_value_heads is not None else self.num_attention_heads
        )
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads

        if isinstance(self.audio_encoder_config, dict):
            audio_encoder_model_type = self.audio_encoder_config.pop("model_type", "mimi")
            self.audio_encoder_config = AutoConfig.for_model(audio_encoder_model_type, **self.audio_encoder_config)
        elif self.audio_encoder_config is None:
            self.audio_encoder_config = AutoConfig.for_model("mimi")

        self.audio_vocab_size = (
            self.audio_encoder_config.codebook_size if self.audio_vocab_size is None else self.audio_vocab_size
        )

        if isinstance(self.depth_decoder_config, dict):
            # Unlike Moshi, `num_codebooks` is not propagated from the main config to the
            # depth decoder: the depth decoder predicts `2 * num_codebooks` slices (dep_q=16).
            self.depth_decoder_config.update(
                {
                    "audio_vocab_size": self.audio_vocab_size,
                    "input_size": self.hidden_size,
                    "vocab_size": self.vocab_size,
                }
            )
            self.depth_decoder_config = PersonaplexDepthConfig(**self.depth_decoder_config)
        elif self.depth_decoder_config is None:
            self.depth_decoder_config = PersonaplexDepthConfig()
        PreTrainedConfig.__post_init__(self, **kwargs)


class PersonaplexModel(MoshiModel):
    pass


class PersonaplexForCausalLM(MoshiForCausalLM):
    pass


class PersonaplexForConditionalGeneration(MoshiForConditionalGeneration):
    def build_persona_prompt(
        self,
        voice_input_values: torch.FloatTensor | None = None,
        persona_input_ids: torch.LongTensor | None = None,
        silence_tokens: list[int] | None = None,
        sine_tokens: list[int] | None = None,
        zero_text_code: int = ZERO_TEXT_CODE,
        pre_silence_frames: int = 6,
        post_silence_frames: int = 6,
    ):
        """
        Builds the PersonaPlex persona prefill (`voice -> silence -> persona text -> silence`) that
        conditions generation on a target voice and/or a text role prompt, faithful to the original
        `step_system_prompts`. The result can be passed directly to `generate`.

        Args:
            voice_input_values (`torch.FloatTensor` of shape `(sequence_length,)` or `(1, 1, sequence_length)`, *optional*):
                Voice reference waveform (24kHz mono, -24 LUFS recommended). Encoded with the audio
                encoder and placed on the Personaplex (agent) stream.
            persona_input_ids (`torch.LongTensor` of shape `(sequence_length,)` or `(1, sequence_length)`, *optional*):
                Tokenized persona/role text (one token per frame on the text stream).
            silence_tokens (`list[int]`, *optional*, defaults to the moshiko constants):
                Per-codebook Mimi tokens for digital silence, tiled on the agent stream after the voice.
            sine_tokens (`list[int]`, *optional*, defaults to the moshiko constants):
                Per-codebook Mimi tokens for the 440 Hz sine, tiled on the whole user stream.
            zero_text_code (`int`, *optional*, defaults to 3):
                Text padding code used on the text stream outside the persona tokens.
            pre_silence_frames (`int`, *optional*, defaults to 6):
                Silence frames between voice and text segments (original serving uses 0.5s = 6 frames).
            post_silence_frames (`int`, *optional*, defaults to 6):
                Silence frames after the text segment.

        Example:
        ```python
        >>> inputs = model.build_persona_prompt(voice_input_values=voice, persona_input_ids=ids)
        >>> out = model.generate(**inputs, max_new_tokens=125)
        ```"""
        if voice_input_values is None and persona_input_ids is None:
            raise ValueError("At least one of `voice_input_values` or `persona_input_ids` must be provided.")

        silence_tokens = SILENCE_TOKENS if silence_tokens is None else silence_tokens
        sine_tokens = SINE_TOKENS if sine_tokens is None else sine_tokens
        if len(silence_tokens) != self.num_codebooks or len(sine_tokens) != self.num_codebooks:
            raise ValueError(
                f"`silence_tokens`/`sine_tokens` must have `num_codebooks={self.num_codebooks}` entries, "
                f"got {len(silence_tokens)} and {len(sine_tokens)}."
            )

        device = self.device
        if voice_input_values is not None:
            voice_input_values = voice_input_values.view(1, 1, -1).to(device=device, dtype=self.dtype)
            frame_size = int(self.config.sampling_rate / self.config.audio_encoder_config.frame_rate)
            remainder = voice_input_values.shape[-1] % frame_size
            if remainder:
                voice_input_values = torch.nn.functional.pad(voice_input_values, (0, frame_size - remainder))
            with torch.no_grad():
                voice_codes = self.audio_encoder.encode(voice_input_values, num_quantizers=self.num_codebooks)[0]
            num_voice_frames = voice_codes.shape[2]
        else:
            num_voice_frames = 0

        if persona_input_ids is not None:
            persona_input_ids = persona_input_ids.view(1, -1).to(device)
        num_text_frames = 0 if persona_input_ids is None else persona_input_ids.shape[1]

        num_frames = num_voice_frames + pre_silence_frames + num_text_frames + post_silence_frames

        silence_frame = torch.tensor(silence_tokens, dtype=torch.long, device=device).view(1, -1, 1)
        sine_frame = torch.tensor(sine_tokens, dtype=torch.long, device=device).view(1, -1, 1)

        personaplex_audio_codes = silence_frame.repeat(1, 1, num_frames - num_voice_frames)
        if num_voice_frames:
            personaplex_audio_codes = torch.cat([voice_codes, personaplex_audio_codes], dim=2)
        user_audio_codes = sine_frame.repeat(1, 1, num_frames)

        input_ids = torch.full((1, num_frames), zero_text_code, dtype=torch.long, device=device)
        if num_text_frames:
            text_start = num_voice_frames + pre_silence_frames
            input_ids[:, text_start : text_start + num_text_frames] = persona_input_ids
        attention_mask = torch.ones((1, num_frames), dtype=torch.long, device=device)

        return PersonaplexUnconditionalInput(
            input_ids=input_ids,
            user_audio_codes=user_audio_codes,
            personaplex_audio_codes=personaplex_audio_codes,
            attention_mask=attention_mask,
        )


__all__ = [
    "PersonaplexConfig",
    "PersonaplexDepthConfig",
    "PersonaplexModel",
    "PersonaplexForCausalLM",
    "PersonaplexForConditionalGeneration",
    "PersonaplexPreTrainedModel",  # noqa: F822
]
