# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from worldfoundry.core.utils.python.lazy_module import _LazyModule

if TYPE_CHECKING:
    # Recommend using `xxx_main`
    from .argument import BaseArguments as BaseArguments
    from .argument import DeployArguments as DeployArguments
    from .argument import InferArguments as InferArguments
    from .infer import (
        AdapterRequest as AdapterRequest,
    )
    from .infer import (
        BaseInferEngine as BaseInferEngine,
    )
    from .infer import (
        InferClient as InferClient,
    )
    from .infer import (
        InferEngine as InferEngine,
    )
    from .infer import (
        LmdeployEngine as LmdeployEngine,
    )
    from .infer import (
        PtEngine as PtEngine,
    )
    from .infer import (
        RequestConfig as RequestConfig,
    )
    from .infer import (
        VllmEngine as VllmEngine,
    )
    from .infer import (
        deploy_main as deploy_main,
    )
    from .infer import (
        infer_main as infer_main,
    )
    from .infer import (
        prepare_model_template as prepare_model_template,
    )
    from .infer import (
        run_deploy as run_deploy,
    )
    from .template import (
        TEMPLATE_MAPPING as TEMPLATE_MAPPING,
    )
    from .template import (
        InferRequest as InferRequest,
    )
    from .template import (
        MaxLengthError as MaxLengthError,
    )
    from .template import (
        Template as Template,
    )
    from .template import (
        TemplateInputs as TemplateInputs,
    )
    from .template import (
        TemplateMeta as TemplateMeta,
    )
    from .template import (
        TemplateType as TemplateType,
    )
    from .template import (
        Word as Word,
    )
    from .template import (
        draw_bbox as draw_bbox,
    )
    from .template import (
        get_template as get_template,
    )
    from .template import (
        get_template_meta as get_template_meta,
    )
    from .template import (
        load_file as load_file,
    )
    from .template import (
        load_image as load_image,
    )
    from .template import (
        register_template as register_template,
    )

    # isort: split
    from .model import (
        MODEL_ARCH_MAPPING as MODEL_ARCH_MAPPING,
    )
    from .model import (
        MODEL_MAPPING as MODEL_MAPPING,
    )
    from .model import (
        HfConfigFactory as HfConfigFactory,
    )
    from .model import (
        Model as Model,
    )
    from .model import (
        ModelArch as ModelArch,
    )
    from .model import (
        ModelGroup as ModelGroup,
    )
    from .model import (
        ModelInfo as ModelInfo,
    )
    from .model import (
        ModelKeys as ModelKeys,
    )
    from .model import (
        ModelMeta as ModelMeta,
    )
    from .model import (
        ModelType as ModelType,
    )
    from .model import (
        MultiModelKeys as MultiModelKeys,
    )
    from .model import (
        get_llm_model as get_llm_model,
    )
    from .model import (
        get_matched_model_meta as get_matched_model_meta,
    )
    from .model import (
        get_model_arch as get_model_arch,
    )
    from .model import (
        get_model_info_meta as get_model_info_meta,
    )
    from .model import (
        get_model_name as get_model_name,
    )
    from .model import (
        get_model_tokenizer as get_model_tokenizer,
    )
    from .model import (
        get_model_tokenizer_multimodal as get_model_tokenizer_multimodal,
    )
    from .model import (
        get_model_tokenizer_with_flash_attn as get_model_tokenizer_with_flash_attn,
    )
    from .model import (
        git_clone_github as git_clone_github,
    )
    from .model import (
        load_by_unsloth as load_by_unsloth,
    )
    from .model import (
        register_model as register_model,
    )
    from .model import (
        register_model_arch as register_model_arch,
    )
    from .model import (
        safe_snapshot_download as safe_snapshot_download,
    )

    # isort: split
    # isort: split
    from .dataset import (
        DATASET_MAPPING as DATASET_MAPPING,
    )
    from .dataset import (
        DATASET_TYPE as DATASET_TYPE,
    )
    from .dataset import (
        AlpacaPreprocessor as AlpacaPreprocessor,
    )
    from .dataset import (
        AutoPreprocessor as AutoPreprocessor,
    )
    from .dataset import (
        DatasetMeta as DatasetMeta,
    )
    from .dataset import (
        EncodePreprocessor as EncodePreprocessor,
    )
    from .dataset import (
        HfDataset as HfDataset,
    )
    from .dataset import (
        LazyLLMDataset as LazyLLMDataset,
    )
    from .dataset import (
        MediaResource as MediaResource,
    )
    from .dataset import (
        MessagesPreprocessor as MessagesPreprocessor,
    )
    from .dataset import (
        ResponsePreprocessor as ResponsePreprocessor,
    )
    from .dataset import (
        RowPreprocessor as RowPreprocessor,
    )
    from .dataset import (
        SubsetDataset as SubsetDataset,
    )
    from .dataset import (
        load_dataset as load_dataset,
    )
    from .dataset import (
        register_dataset as register_dataset,
    )
    from .dataset import (
        register_dataset_info as register_dataset_info,
    )
    from .dataset import (
        sample_dataset as sample_dataset,
    )

    # isort: split
    # isort: split
    from .utils import (
        History as History,
    )
    from .utils import (
        Messages as Messages,
    )
    from .utils import (
        Processor as Processor,
    )
    from .utils import (
        ProcessorMixin as ProcessorMixin,
    )
    from .utils import (
        deep_getattr as deep_getattr,
    )
    from .utils import (
        dynamic_gradient_checkpointing as dynamic_gradient_checkpointing,
    )
    from .utils import (
        get_cache_dir as get_cache_dir,
    )
    from .utils import (
        get_temporary_cache_files_directory as get_temporary_cache_files_directory,
    )
    from .utils import (
        history_to_messages as history_to_messages,
    )
    from .utils import (
        is_moe_model as is_moe_model,
    )
    from .utils import (
        messages_to_history as messages_to_history,
    )
    from .utils import (
        save_checkpoint as save_checkpoint,
    )
    from .utils import (
        to_device as to_device,
    )
    from .utils import (
        to_float_dtype as to_float_dtype,
    )

    # isort: split
    # isort: split
    from .base import SwiftPipeline as SwiftPipeline

    # isort: split
    from .data_loader import BatchSamplerShard as BatchSamplerShard
    from .data_loader import DataLoaderDispatcher as DataLoaderDispatcher
    from .data_loader import DataLoaderShard as DataLoaderShard
else:
    _import_structure = {
        "infer": [
            "deploy_main",
            "VllmEngine",
            "LmdeployEngine",
            "PtEngine",
            "infer_main",
            "InferClient",
            "run_deploy",
            "InferEngine",
            "AdapterRequest",
            "prepare_model_template",
            "BaseInferEngine",
        ],
        "infer.request_config": ["RequestConfig"],
        "argument": ["InferArguments", "DeployArguments", "BaseArguments"],
        "template": [
            "TEMPLATE_MAPPING",
            "Template",
            "Word",
            "get_template",
            "TemplateType",
            "register_template",
            "TemplateInputs",
            "TemplateMeta",
            "get_template_meta",
            "InferRequest",
            "load_image",
            "MaxLengthError",
            "load_file",
            "draw_bbox",
        ],
        "model": [
            "MODEL_MAPPING",
            "ModelType",
            "get_model_tokenizer",
            "safe_snapshot_download",
            "HfConfigFactory",
            "ModelInfo",
            "ModelMeta",
            "ModelKeys",
            "register_model_arch",
            "MultiModelKeys",
            "ModelArch",
            "MODEL_ARCH_MAPPING",
            "get_model_arch",
            "get_model_info_meta",
            "get_model_name",
            "register_model",
            "ModelGroup",
            "Model",
            "get_model_tokenizer_with_flash_attn",
            "get_model_tokenizer_multimodal",
            "load_by_unsloth",
            "git_clone_github",
            "get_matched_model_meta",
            "get_llm_model",
        ],
        "dataset": [
            "AlpacaPreprocessor",
            "MessagesPreprocessor",
            "AutoPreprocessor",
            "DATASET_MAPPING",
            "MediaResource",
            "register_dataset",
            "register_dataset_info",
            "EncodePreprocessor",
            "LazyLLMDataset",
            "load_dataset",
            "DATASET_TYPE",
            "sample_dataset",
            "RowPreprocessor",
            "ResponsePreprocessor",
            "DatasetMeta",
            "HfDataset",
            "SubsetDataset",
        ],
        "utils": [
            "deep_getattr",
            "to_device",
            "to_float_dtype",
            "History",
            "Messages",
            "history_to_messages",
            "messages_to_history",
            "Processor",
            "save_checkpoint",
            "ProcessorMixin",
            "get_temporary_cache_files_directory",
            "get_cache_dir",
            "is_moe_model",
            "dynamic_gradient_checkpointing",
        ],
        "base": ["SwiftPipeline"],
        "data_loader": ["DataLoaderDispatcher", "DataLoaderShard", "BatchSamplerShard"],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
