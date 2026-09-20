""" PyTorch Mask2Former model."""

import math
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
import numpy as np
import torch
from torch import Tensor, nn
from collections import OrderedDict
from ... import AutoBackbone, SwinConfig
from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    replace_return_docstrings,
    requires_backends,
)
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_mask2former import Mask2FormerConfig
from timm.models.layers import DropPath
from fightingcv_attention.attention.AFT import AFT_FULL
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "Mask2FormerConfig"
_CHECKPOINT_FOR_DOC = "facebook/mask2former-swin-small-coco-instance"
_IMAGE_PROCESSOR_FOR_DOC = "opticalMask2FormerImageProcessor"

MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mask2former-swin-small-coco-instance",
]


@dataclass
class Mask2FormerPixelLevelModuleOutput(ModelOutput):
    """
    Mask2Former's pixel level module output. It returns the output of the encoder (optional) and all hidden states
    (multi-scale features) from the `decoder`. By default, the `encoder` is a Swin Backbone and the `decoder` is a
    Multi-Scale Deformable Attention based decoder.

    The `decoder_last_hidden_state` are the **per-pixel embeddings** while `decoder_hidden_states` refer to multi-scale
    feature maps produced using **multi-scaling strategy** defined in the paper.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor`):
            Last hidden states (final feature map of shape `(batch_size, num_channels, height, width)`) of the last
            stage of the encoder.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`. Hidden states (also
            called feature maps) of the model at the output of each stage. Returned if output_hidden_states is set to
            True.
        decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)):
            1/4 scale features from the last Pixel Decoder Layer.
        decoder_hidden_states (`tuple(torch.FloatTensor)`):
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`. Hidden states (also
            called feature maps) of the model at the output of each stage.
    """

    encoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_last_hidden_state: torch.FloatTensor = None
    decoder_hidden_states: Tuple[torch.FloatTensor] = None


class Mask2FormerPixelLevelModule(nn.Module):
    def __init__(self, config: Mask2FormerConfig):
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://arxiv.org/abs/2112.01527). It runs the input image through a backbone and a pixel
        decoder, generating multi-scale feature maps and pixel embeddings.

        Args:
            config ([`Mask2FormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()

        backbone_config_dict = config.backbone_config.to_dict()
        backbone_config = SwinConfig.from_dict(backbone_config_dict)
        self.encoder = AutoBackbone.from_config(backbone_config)
        self.decoder = Mask2FormerPixelDecoder(config, feature_channels=self.encoder.channels)


    def forward(self, pixel_values: Tensor, align_features_img_txt: Tensor, soft_label_text: Tensor, output_hidden_states: bool = False) -> Mask2FormerPixelLevelModuleOutput:
        
        backbone_features = self.encoder(pixel_values).feature_maps
        decoder_output = self.decoder(backbone_features, output_hidden_states=output_hidden_states)

        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=backbone_features[-1],
            encoder_hidden_states=tuple(backbone_features) if output_hidden_states else None,
            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        ),decoder_output.mask_features



MASK2FORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Mask2FormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MASK2FORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.
        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of Detr's decoder attention layers.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~Mask2FormerModelOutput`] instead of a plain tuple.
"""


class Mask2FormerPreTrainedModel(PreTrainedModel):
    config_class = Mask2FormerConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def _init_weights(self, module: nn.Module):
        xavier_std = self.config.init_xavier_std
        std = self.config.init_std

        if isinstance(module, Mask2FormerTransformerModule):
            if module.input_projections is not None:
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        nn.init.xavier_uniform_(input_projection.weight, gain=xavier_std)
                        nn.init.constant_(input_projection.bias, 0)

        elif isinstance(module, Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            thetas = torch.arange(module.n_heads, dtype=torch.float32) * (2.0 * math.pi / module.n_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)

        elif isinstance(module, Mask2FormerMaskedAttentionDecoderLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)

        elif isinstance(module, Mask2FormerPixelLevelModule):
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                    submodule.weight.data.normal_(mean=0.0, std=std)
                    if submodule.bias is not None:
                        submodule.bias.data.zero_()

        elif isinstance(module, Mask2FormerPixelDecoder):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.normal_(module.level_embed, std=0)

        elif isinstance(module, Mask2FormerPixelDecoderEncoderOnly):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if hasattr(module, "reference_points"):
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)



@add_start_docstrings(
    "The bare Mask2Former Model outputting raw hidden-states without any specific head on top.",
    MASK2FORMER_START_DOCSTRING,
)

class Mask2FormerModel(Mask2FormerPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.pixel_level_module = Mask2FormerPixelLevelModule(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(MASK2FORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Mask2FormerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Optional[Tensor] = None,
        prompt_features_projected: Optional[Tensor] = None,
        ###################Diffusion############
        diffusion_img: Optional[Tensor] = None,
        optical: Optional[Tensor] = None,
        audio_inference_text: Optional[Tensor] = None,
        visual_inference_text: Optional[Tensor] = None,
        unified_text: Optional[Tensor] = None,
        ###################Diffusion############
        ###################ICME#################
        extra_prompt_features_projected: Optional[Tensor] = None,
        extra_prompt_features_projected_original: Optional[Tensor] = None,
        # extra_prompt_features_projected_audio:Optional[Tensor]=None,
        ###################ICME#################
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Mask2FormerModelOutput:
        r"""
        Returns:
            `Mask2FormerModelOutput`

        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoImageProcessor, Mask2FormerModel

        >>> # load image
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # load image preprocessor and Mask2FormerModel trained on COCO instance segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
        >>> model = Mask2FormerModel.from_pretrained("facebook/mask2former-swin-small-coco-instance")
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # model outputs last hidden states of shape (batch_size, num_queries, hidden_size)
        >>> print(outputs.transformer_decoder_last_hidden_state.shape)
        torch.Size([1, 100, 256])
        ```
        """
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape 

        pixel_level_module_output, mask_feature_image= self.pixel_level_module(
            pixel_values=pixel_values, align_features_img_txt=extra_prompt_features_projected_original, soft_label_text=None, output_hidden_states=output_hidden_states
        )
 

        return pixel_level_module_output


@add_start_docstrings(
    "The Mask2Former Model with heads on top for instance/semantic/panoptic segmentation.",
    MASK2FORMER_START_DOCSTRING,
)
class opticalMask2FormerForUniversalSegmentation(Mask2FormerPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.model = Mask2FormerModel(config)
        self.post_init()
        

    @add_start_docstrings_to_model_forward(MASK2FORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Mask2FormerForUniversalSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        diffusion_img: Optional[Tensor] = None,
        optical: Optional[Tensor] = None,
        audio_inference_text: Optional[Tensor] = None,
        visual_inference_text: Optional[Tensor] = None,
        unified_text: Optional[Tensor] = None,
        prompt_features_projected: Optional[Tensor] = None,
        extra_prompt_features_projected: Optional[Tensor] = None,
        extra_prompt_features_projected_original: Optional[Tensor] = None,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Mask2FormerForUniversalSegmentationOutput:
        r"""
        mask_labels (`List[torch.Tensor]`, *optional*):
            List of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`List[torch.LongTensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.

        Returns:
            `Mask2FormerUniversalSegmentationOutput`

        Examples:

        Instance segmentation example:

        ```python
        >>> from transformers import AutoImageProcessor, opticalMask2FormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # Load Mask2Former trained on COCO instance segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
        >>> model = opticalMask2FormerForUniversalSegmentation.from_pretrained(
        ...     "facebook/mask2former-swin-small-coco-instance"
        ... )

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # Perform post-processing to get instance segmentation map
        >>> pred_instance_map = image_processor.post_process_semantic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]
        >>> print(pred_instance_map.shape)
        torch.Size([480, 640])
        ```

        Semantic segmentation example:
        ```python
        >>> from transformers import AutoImageProcessor, opticalMask2FormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # Load Mask2Former trained on ADE20k semantic segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        >>> model = opticalMask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")

        >>> url = (
        ...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
        ... )
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # Perform post-processing to get semantic segmentation map
        >>> pred_semantic_map = image_processor.post_process_semantic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]
        >>> 
        (pred_semantic_map.shape)
        torch.Size([512, 683])
        ```

        Panoptic segmentation example:

        ```python
        >>> from transformers import AutoImageProcessor, opticalMask2FormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # Load Mask2Former trained on CityScapes panoptic segmentation dataset
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")
        >>> model = opticalMask2FormerForUniversalSegmentation.from_pretrained(
        ...     "facebook/mask2former-swin-small-cityscapes-panoptic"
        ... )

        >>> url = "https://cdn-media.huggingface.co/Inference-API/Sample-results-on-the-Cityscapes-dataset-The-above-images-show-how-our-method-can-handle.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # Perform post-processing to get panoptic segmentation map
        >>> pred_panoptic_map = image_processor.post_process_panoptic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]["segmentation"]
        >>> print(pred_panoptic_map.shape)
        torch.Size([338, 676])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            prompt_features_projected=prompt_features_projected,
            diffusion_img = diffusion_img,    
            optical=optical,
            audio_inference_text=audio_inference_text,
            visual_inference_text=visual_inference_text,
            unified_text=unified_text,
            extra_prompt_features_projected=extra_prompt_features_projected,
            extra_prompt_features_projected_original=extra_prompt_features_projected_original,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            output_attentions=output_attentions,
            return_dict=True,
        )

 
        return output

        