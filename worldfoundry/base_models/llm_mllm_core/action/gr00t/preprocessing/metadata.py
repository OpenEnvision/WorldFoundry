# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from numpydantic import NDArray
from pydantic import BaseModel, Field, field_serializer

from .embodiment_tags import EmbodimentTag
from .schema import DatasetModalities

# Common schema


# LeRobot schema


# Dataset schema (parsed from LeRobot schema and simplified)


class DatasetStatisticalValues(BaseModel):
    max: NDArray = Field(..., description="Maximum values")
    min: NDArray = Field(..., description="Minimum values")
    mean: NDArray = Field(..., description="Mean values")
    std: NDArray = Field(..., description="Standard deviation")
    q01: NDArray = Field(..., description="1st percentile values")
    q99: NDArray = Field(..., description="99th percentile values")

    @field_serializer("*", when_used="json")
    def serialize_ndarray(self, v: NDArray) -> list[float]:
        return v.tolist()  # type: ignore


class DatasetStatistics(BaseModel):
    state: dict[str, DatasetStatisticalValues] = Field(..., description="Statistics of the state")
    action: dict[str, DatasetStatisticalValues] = Field(..., description="Statistics of the action")


class DatasetMetadata(BaseModel):
    """Normalization statistics, modality geometry and the N1 embodiment tag."""

    statistics: DatasetStatistics = Field(..., description="Statistics of the dataset")
    modalities: DatasetModalities = Field(..., description="Metadata of the modalities")
    embodiment_tag: EmbodimentTag = Field(..., description="Embodiment tag of the dataset")
