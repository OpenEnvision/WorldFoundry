# Copyright (c) Alibaba, Inc. and its affiliates.
import datasets.fingerprint
from datasets import Dataset as HfDataset  # noqa: F401 - Re-exported under this name.

from ..utils import get_temporary_cache_files_directory
from . import dataset as dataset
from .loader import DATASET_TYPE as DATASET_TYPE
from .loader import load_dataset as load_dataset
from .media import MediaResource as MediaResource
from .preprocessor import (
    AlpacaPreprocessor as AlpacaPreprocessor,
)
from .preprocessor import (
    AutoPreprocessor as AutoPreprocessor,
)
from .preprocessor import (
    MessagesPreprocessor as MessagesPreprocessor,
)
from .preprocessor import (
    ResponsePreprocessor as ResponsePreprocessor,
)
from .preprocessor import (
    RowPreprocessor as RowPreprocessor,
)
from .register import DATASET_MAPPING as DATASET_MAPPING
from .register import DatasetMeta as DatasetMeta
from .register import SubsetDataset as SubsetDataset
from .register import register_dataset as register_dataset
from .register import register_dataset_info
from .utils import (
    EncodePreprocessor as EncodePreprocessor,
)
from .utils import (
    GetLengthPreprocessor as GetLengthPreprocessor,
)
from .utils import (
    IterablePackingDataset as IterablePackingDataset,
)
from .utils import (
    LazyLLMDataset as LazyLLMDataset,
)
from .utils import (
    PackingDataset as PackingDataset,
)
from .utils import (
    sample_dataset as sample_dataset,
)

datasets.fingerprint.get_temporary_cache_files_directory = get_temporary_cache_files_directory
datasets.arrow_dataset.get_temporary_cache_files_directory = get_temporary_cache_files_directory
register_dataset_info()
