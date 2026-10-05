# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from worldfoundry.core.utils.python.lazy_module import _LazyModule

if TYPE_CHECKING:
    # Recommend using `xxx_main`
    from .infer import (VllmEngine, RequestConfig, LmdeployEngine, PtEngine, InferEngine, infer_main, deploy_main,
                        InferClient, run_deploy, AdapterRequest, prepare_model_template, BaseInferEngine)
    from .argument import InferArguments, DeployArguments, BaseArguments
    from .template import (TEMPLATE_MAPPING, Template, Word, get_template, TemplateType, register_template,
                           TemplateInputs, TemplateMeta, get_template_meta, InferRequest, load_image, MaxLengthError,
                           load_file, draw_bbox)
    from .model import (register_model, MODEL_MAPPING, ModelType, get_model_tokenizer, safe_snapshot_download,
                        HfConfigFactory, ModelInfo, ModelMeta, ModelKeys, register_model_arch, MultiModelKeys,
                        ModelArch, get_model_arch, MODEL_ARCH_MAPPING, get_model_info_meta, get_model_name, ModelGroup,
                        Model, get_model_tokenizer_with_flash_attn, get_model_tokenizer_multimodal, load_by_unsloth,
                        git_clone_github, get_matched_model_meta, get_llm_model)
    from .dataset import (AlpacaPreprocessor, ResponsePreprocessor, MessagesPreprocessor, AutoPreprocessor,
                          DATASET_MAPPING, MediaResource, register_dataset, register_dataset_info, EncodePreprocessor,
                          LazyLLMDataset, load_dataset, DATASET_TYPE, sample_dataset, RowPreprocessor, DatasetMeta,
                          HfDataset, SubsetDataset)
    from .utils import (deep_getattr, to_float_dtype, to_device, History, Messages, history_to_messages,
                        messages_to_history, Processor, save_checkpoint, ProcessorMixin,
                        get_temporary_cache_files_directory, get_cache_dir, is_moe_model,
                        dynamic_gradient_checkpointing)
    from .base import SwiftPipeline
    from .data_loader import DataLoaderDispatcher, DataLoaderShard, BatchSamplerShard
else:
    _import_structure = {
        'infer': [
            'deploy_main', 'VllmEngine', 'LmdeployEngine', 'PtEngine', 'infer_main', 'InferClient',
            'run_deploy', 'InferEngine', 'AdapterRequest', 'prepare_model_template', 'BaseInferEngine'
        ],
        'infer.request_config': ['RequestConfig'],
        'argument': ['InferArguments', 'DeployArguments', 'BaseArguments'],
        'template': [
            'TEMPLATE_MAPPING', 'Template', 'Word', 'get_template', 'TemplateType', 'register_template',
            'TemplateInputs', 'TemplateMeta', 'get_template_meta', 'InferRequest', 'load_image', 'MaxLengthError',
            'load_file', 'draw_bbox'
        ],
        'model': [
            'MODEL_MAPPING', 'ModelType', 'get_model_tokenizer', 'safe_snapshot_download', 'HfConfigFactory',
            'ModelInfo', 'ModelMeta', 'ModelKeys', 'register_model_arch', 'MultiModelKeys', 'ModelArch',
            'MODEL_ARCH_MAPPING', 'get_model_arch', 'get_model_info_meta', 'get_model_name', 'register_model',
            'ModelGroup', 'Model', 'get_model_tokenizer_with_flash_attn', 'get_model_tokenizer_multimodal',
            'load_by_unsloth', 'git_clone_github', 'get_matched_model_meta', 'get_llm_model'
        ],
        'dataset': [
            'AlpacaPreprocessor', 'MessagesPreprocessor', 'AutoPreprocessor', 'DATASET_MAPPING', 'MediaResource',
            'register_dataset', 'register_dataset_info', 'EncodePreprocessor', 'LazyLLMDataset', 'load_dataset',
            'DATASET_TYPE', 'sample_dataset', 'RowPreprocessor', 'ResponsePreprocessor', 'DatasetMeta', 'HfDataset',
            'SubsetDataset'
        ],
        'utils': [
            'deep_getattr', 'to_device', 'to_float_dtype', 'History', 'Messages', 'history_to_messages',
            'messages_to_history', 'Processor', 'save_checkpoint', 'ProcessorMixin',
            'get_temporary_cache_files_directory', 'get_cache_dir', 'is_moe_model', 'dynamic_gradient_checkpointing'
        ],
        'base': ['SwiftPipeline'],
        'data_loader': ['DataLoaderDispatcher', 'DataLoaderShard', 'BatchSamplerShard'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
