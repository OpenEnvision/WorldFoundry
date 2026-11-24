# ruff: noqa: F722 - jaxtyping dimension strings in annotations.
from flax import struct

from ...openpi import array_typing as at
from ...openpi.modeling.model import ArrayT
from ...openpi.modeling.model import Observation as _Observation
from ...openpi.modeling.model import preprocess_observation as _preprocess_observation

#  b: batch size
#  t: time step
#  v: view
#  h: height
#  w: width
#  c: channel
#  l: seq len


@at.typecheck
@struct.dataclass
class HistAugObservation(_Observation):
    # for perceptual memory
    static_image_emb: at.Float[at.Array, "b l1 d1"] | None = None
    static_mask: at.Bool[at.Array, "b l1"] | None = None
    static_pos_emb: at.Float[at.Array, "b l1 d2"] | None = None
    static_state_emb: at.Float[at.Array, "b l1 d3"] | None = None

    # for recurrent memory
    recur_image_emb: at.Float[at.Array, "b t v p d1"] | None = None  # left padded
    recur_mask: at.Bool[at.Array, "b t"] | None = None  # left padded
    recur_pos_emb: at.Float[at.Array, "b t v p d2"] | None = None  # left padded
    recur_state_emb: at.Float[at.Array, "b t d3"] | None = None  # left padded

    # for symbolic memory
    symbolic_tokenized_prompt: at.Int[at.Array, "b l2"] | None = None
    symbolic_tokenized_prompt_mask: at.Bool[at.Array, "b l2"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "HistAugObservation":
        parent_obs = super().from_dict(data)
        return cls(
            # Base observation fields
            images=parent_obs.images,
            image_masks=parent_obs.image_masks,
            state=parent_obs.state,
            tokenized_prompt=parent_obs.tokenized_prompt,
            tokenized_prompt_mask=parent_obs.tokenized_prompt_mask,
            token_ar_mask=parent_obs.token_ar_mask,
            # MMEVLA fields
            static_image_emb=data.get("static_image_emb", None),
            static_mask=data.get("static_mask", None),
            static_pos_emb=data.get("static_pos_emb", None),
            static_state_emb=data.get("static_state_emb", None),
            recur_image_emb=data.get("recur_image_emb", None),
            recur_mask=data.get("recur_mask", None),
            recur_pos_emb=data.get("recur_pos_emb", None),
            recur_state_emb=data.get("recur_state_emb", None),
            symbolic_tokenized_prompt=data.get("symbolic_tokenized_prompt", None),
            symbolic_tokenized_prompt_mask=data.get("symbolic_tokenized_prompt_mask", None),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        result = super().to_dict()
        result["static_image_emb"] = self.static_image_emb
        result["static_mask"] = self.static_mask
        result["static_pos_emb"] = self.static_pos_emb
        result["static_state_emb"] = self.static_state_emb
        result["recur_image_emb"] = self.recur_image_emb
        result["recur_mask"] = self.recur_mask
        result["recur_pos_emb"] = self.recur_pos_emb
        result["recur_state_emb"] = self.recur_state_emb
        result["symbolic_tokenized_prompt"] = self.symbolic_tokenized_prompt  #  subgoal
        result["symbolic_tokenized_prompt_mask"] = self.symbolic_tokenized_prompt_mask
        return result

    def to_base_obs(self) -> _Observation:
        return _Observation(
            images=self.images,
            image_masks=self.image_masks,
            state=self.state,
            tokenized_prompt=self.tokenized_prompt,
            tokenized_prompt_mask=self.tokenized_prompt_mask,
            token_ar_mask=self.token_ar_mask,
        )

    @classmethod
    def from_base_obs(
        cls,
        base_obs: _Observation,
        static_image_emb: at.Float[ArrayT, "*b l d1"] | None = None,
        static_mask: at.Bool[ArrayT, "*b l"] | None = None,
        static_pos_emb: at.Float[ArrayT, "*b l d2"] | None = None,
        static_state_emb: at.Float[ArrayT, "*b l d3"] | None = None,
        recur_image_emb: at.Float[ArrayT, "*b t v p d1"] | None = None,
        recur_mask: at.Bool[ArrayT, "*b t"] | None = None,
        recur_pos_emb: at.Float[ArrayT, "*b t v p d2"] | None = None,
        recur_state_emb: at.Float[ArrayT, "*b t d3"] | None = None,
        symbolic_tokenized_prompt: at.Int[ArrayT, "*b l d5"] | None = None,
        symbolic_tokenized_prompt_mask: at.Bool[ArrayT, "*b l d6"] | None = None,
    ) -> "HistAugObservation":
        return HistAugObservation(
            images=base_obs.images,
            image_masks=base_obs.image_masks,
            state=base_obs.state,
            tokenized_prompt=base_obs.tokenized_prompt,
            tokenized_prompt_mask=base_obs.tokenized_prompt_mask,
            token_ar_mask=base_obs.token_ar_mask,
            static_image_emb=static_image_emb,
            static_mask=static_mask,
            static_pos_emb=static_pos_emb,
            static_state_emb=static_state_emb,
            recur_image_emb=recur_image_emb,
            recur_mask=recur_mask,
            recur_pos_emb=recur_pos_emb,
            recur_state_emb=recur_state_emb,
            symbolic_tokenized_prompt=symbolic_tokenized_prompt,
            symbolic_tokenized_prompt_mask=symbolic_tokenized_prompt_mask,
        )


def preprocess_observation(
    observation: HistAugObservation,
) -> HistAugObservation:
    base_obs: _Observation = _preprocess_observation(
        None,
        observation.to_base_obs(),
        image_keys=tuple(observation.images),
    )
    return HistAugObservation.from_base_obs(
        base_obs,
        static_image_emb=observation.static_image_emb,
        static_mask=observation.static_mask,
        static_pos_emb=observation.static_pos_emb,
        static_state_emb=observation.static_state_emb,
        recur_image_emb=observation.recur_image_emb,
        recur_mask=observation.recur_mask,
        recur_pos_emb=observation.recur_pos_emb,
        recur_state_emb=observation.recur_state_emb,
        symbolic_tokenized_prompt=observation.symbolic_tokenized_prompt,
        symbolic_tokenized_prompt_mask=observation.symbolic_tokenized_prompt_mask,
    )
