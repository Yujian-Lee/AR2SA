# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING
# import bert
# from ..bert.configuration_bert import BertConfig
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available
# from registery import registery
# from ...models.bert.configuration_bert import BertConfig

_import_structure = {
    "configuration_mask2former": [
        "MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Mask2FormerConfig",
    ],
}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_mask2former"] = ["Mask2FormerImageProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_mask2former"] = [
        "MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Mask2FormerForUniversalSegmentation",
        "Mask2FormerModel",
        "Mask2FormerPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_mask2former import MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, Mask2FormerConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_mask2former import Mask2FormerImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mask2former import (
            MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            Mask2FormerForUniversalSegmentation,
            Mask2FormerModel,
            Mask2FormerPreTrainedModel,
        )
        from .optical_flow import (
            opticalMask2FormerForUniversalSegmentation,
            Mask2FormerModel,
            Mask2FormerPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
