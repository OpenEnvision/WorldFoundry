import functools

import jax
from absl import logging
from flax import nnx

import worldfoundry.synthesis.visual_generation.solaris.solaris_runtime.src.utils.sharding as sharding_utils
from worldfoundry.synthesis.visual_generation.solaris.solaris_runtime.src.runners.base_mp_runner import BaseMPRunner
from worldfoundry.synthesis.visual_generation.solaris.solaris_runtime.src.runners.base_runner import (
    restore_nnx_checkpoint,
)
from worldfoundry.synthesis.visual_generation.solaris.solaris_runtime.src.utils.config import instantiate_from_config


class Inference(BaseMPRunner):
    def __init__(
        self,
        model_weights_path,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_weights_path = model_weights_path

        def get_model(rngs):
            nnx_rngs = nnx.Rngs(params=rngs)
            model = instantiate_from_config(self.network_config, rngs=nnx_rngs)
            return nnx.split(model)

        _, state_shape = jax.eval_shape(functools.partial(get_model, rngs=self.rngs))
        self.model_sharding = sharding_utils.apply_sharding(state_shape, self.mesh)
        self.model_graph, model_state = jax.jit(get_model, out_shardings=(self.repl_sharding, self.model_sharding))(
            self.rngs
        )
        self.model_state = restore_nnx_checkpoint(self.pretrained_checkpointer, model_weights_path, model_state)
        self.multiplayer_method = self.network_config.params.multiplayer_method

    def _evaluate(
        self,
        model_state,
        model_graph,
        vae_state,
        vae_graph,
        clip_state,
        clip_graph,
        video,
        mouse_actions,
        keyboard_actions,
        real_lengths,
        eval_dir,
        mesh,
        left_action_padding,
        num_denoising_steps=None,
    ):
        return self.evaluate_mp(
            bidirectional=False,
            model_state=model_state,
            model_graph=model_graph,
            vae_state=vae_state,
            vae_graph=vae_graph,
            clip_state=clip_state,
            clip_graph=clip_graph,
            video=video,
            mouse_actions=mouse_actions,
            keyboard_actions=keyboard_actions,
            real_lengths=real_lengths,
            eval_dir=eval_dir,
            mesh=mesh,
            left_action_padding=left_action_padding,
            num_denoising_steps=num_denoising_steps,
        )

    def run(self):
        with self.mesh:
            self.run_evals()

    def run_evals(self):
        for eval_dataset_name, eval_dataloader_info in self.eval_dataloaders.items():
            model = nnx.merge(self.model_graph, self.model_state)
            logging.info(f"Running eval on {eval_dataset_name}")
            self.run_eval(
                model=model,
                num_denoising_steps=None,
                eval_dataloader_info=eval_dataloader_info,
                eval_dir_name=eval_dataset_name,
            )
