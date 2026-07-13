"""PyTorch Personaplex model."""

from ..moshi.configuration_moshi import MoshiConfig, MoshiDepthConfig
from ..moshi.modeling_moshi import (
    MoshiForCausalLM,
    MoshiForConditionalGeneration,
    MoshiModel,
)


class PersonaplexDepthConfig(MoshiDepthConfig):
    pass


class PersonaplexConfig(MoshiConfig):
    pass


class PersonaplexModel(MoshiModel):
    pass


class PersonaplexForCausalLM(MoshiForCausalLM):
    pass


class PersonaplexForConditionalGeneration(MoshiForConditionalGeneration):
    pass


__all__ = [
    "PersonaplexConfig",
    "PersonaplexDepthConfig",
    "PersonaplexModel",
    "PersonaplexForCausalLM",
    "PersonaplexForConditionalGeneration",
    "PersonaplexPreTrainedModel",  # noqa: F822
]
