# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from paddle import nn

from paddlenlp.transformers import LayoutXLMModel, LayoutXLMForTokenClassification, LayoutXLMForRelationExtraction
from paddlenlp.transformers import LayoutLMModel, LayoutLMForTokenClassification
from paddlenlp.transformers import LayoutLMv2Model, LayoutLMv2ForTokenClassification, LayoutLMv2ForRelationExtraction
from paddlenlp.transformers import AutoModel
from paddlenlp.transformers import BertModel, BertForTokenClassification
from ppocr.modeling.backbones.utils.nlp_model import BertForRelationExtraction

__all__ = ["LayoutXLMForSer", "LayoutLMForSer"]

pretrained_model_dict = {
    LayoutXLMModel: {
        "base": "layoutxlm-base-uncased",
        "vi": "vi-layoutxlm-base-uncased",
    },
    LayoutLMModel: {
        "base": "layoutlm-base-uncased",
    },
    LayoutLMv2Model: {
        "base": "layoutlmv2-base-uncased",
        "vi": "vi-layoutlmv2-base-uncased",
    },
    BertModel: {
        "base": "bert-base-chinese"
    }
}


class NLPBaseModel(nn.Layer):
    def __init__(self,
                 base_model_class,
                 model_class,
                 mode="base",
                 type="ser",
                 pretrained=True,
                 checkpoints=None,
                 **kwargs):
        """
        Args:
            base_model_class (class): The base model class of the model to instantiate.
            model_class (class): downstream model class, 
                LayoutLMv2ForRelationExtraction for relation extraction, LayoutXLMForTokenClassification for token classification. 
            mode (str, optional): The model mode of base_model_class. Defaults to "base".
            type (str, optional): The type of downstream model. Defaults to "ser".
            pretrained (bool, optional): Whether to load pretrained model. Defaults to True.
            checkpoints (str, optional): The path or url of pretrained model. Defaults to None.
        """
        super(NLPBaseModel, self).__init__()
        if checkpoints is not None:  # load the trained model
            self.model = model_class.from_pretrained(checkpoints)
        else:  # load the pretrained-model
            pretrained_model_name = pretrained_model_dict[base_model_class][mode]
            base_model = base_model_class.from_pretrained(pretrained_model_name) \
                if pretrained else base_model_class.from_pretrained(pretrained)
            
            if type == "ser":
                self.model = model_class(
                    base_model, num_classes=kwargs["num_classes"], dropout=None)
            elif type=="re":
                self.model = model_class(base_model, dropout=None)
            else:
                raise ValueError("type must be ser or re")
        
        self.out_channels = 1
        self.use_visual_backbone = True


class LayoutLMForSer(NLPBaseModel):
    def __init__(self,
                 num_classes,
                 pretrained=True,
                 checkpoints=None,
                 mode="base",
                 **kwargs):
        super(LayoutLMForSer, self).__init__(
            LayoutLMModel,
            LayoutLMForTokenClassification,
            mode,
            "ser",
            pretrained,
            checkpoints,
            num_classes=num_classes, )
        self.use_visual_backbone = False

    def forward(self, x):
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            position_ids=None,
            output_hidden_states=False)
        return x


class LayoutLMv2ForSer(NLPBaseModel):
    def __init__(self,
                 num_classes,
                 pretrained=True,
                 checkpoints=None,
                 mode="base",
                 **kwargs):
        super(LayoutLMv2ForSer, self).__init__(
            LayoutLMv2Model,
            LayoutLMv2ForTokenClassification,
            mode,
            "ser",
            pretrained,
            checkpoints,
            num_classes=num_classes)
        if hasattr(self.model.layoutlmv2, "use_visual_backbone"
                   ) and self.model.layoutlmv2.use_visual_backbone is False:
            self.use_visual_backbone = False

    def forward(self, x):
        if self.use_visual_backbone is True:
            image = x[4]
        else:
            image = None
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            image=image,
            position_ids=None,
            head_mask=None,
            labels=None)
        if self.training:
            res = {"backbone_out": x[0]}
            res.update(x[1])
            return res
        else:
            return x


class LayoutXLMForSer(NLPBaseModel):
    def __init__(self,
                 num_classes,
                 pretrained=True,
                 checkpoints=None,
                 mode="base",
                 **kwargs):
        super(LayoutXLMForSer, self).__init__(
            LayoutXLMModel,
            LayoutXLMForTokenClassification,
            mode,
            "ser",
            pretrained,
            checkpoints,
            num_classes=num_classes)
        if hasattr(self.model.layoutxlm, "use_visual_backbone"
                   ) and self.model.layoutxlm.use_visual_backbone is False:
            self.use_visual_backbone = False

    def forward(self, x):
        """
        x: ['input_ids', 'bbox', 'attention_mask', 
            'token_type_ids', 'image', 'labels', 
            'segment_offset_id', 'ocr_info', 'entities']
        """
        image = x[4] if self.use_visual_backbone is True else None

        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            image=image,
            position_ids=None,
            head_mask=None,
            labels=None)
        if self.training:
            res = {"backbone_out": x[0]}
            res.update(x[1])
            return res
        else:
            return x


class BertForSer(NLPBaseModel):
    def __init__(self,
                 num_classes,
                 pretrained=True,
                 checkpoints=None,
                 mode="base",
                 **kwargs):
        super(BertForSer, self).__init__(
            BertModel,
            BertForTokenClassification,
            mode,
            type="ser",
            pretrained=pretrained,
            checkpoints=checkpoints,
            num_classes=num_classes
        )

    def forward(self, x):
        x = self.model(input_ids=x[0], token_type_ids=x[3])
        if self.training:
            res = {"backbone_out": x}
            return res
        else:
            return x


class LayoutLMv2ForRe(NLPBaseModel):
    def __init__(self, pretrained=True, checkpoints=None, mode="base",
                 **kwargs):
        super(LayoutLMv2ForRe, self).__init__(
            LayoutLMv2Model, LayoutLMv2ForRelationExtraction, mode, "re",
            pretrained, checkpoints)
        if hasattr(self.model.layoutlmv2, "use_visual_backbone"
                   ) and self.model.layoutlmv2.use_visual_backbone is False:
            self.use_visual_backbone = False

    def forward(self, x):
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            image=x[4],
            position_ids=None,
            head_mask=None,
            labels=None,
            entities=x[5],
            relations=x[6])
        return x


class LayoutXLMForRe(NLPBaseModel):
    def __init__(self, pretrained=True, checkpoints=None, mode="base",
                 **kwargs):
        super(LayoutXLMForRe, self).__init__(
            LayoutXLMModel, LayoutXLMForRelationExtraction, mode, "re",
            pretrained, checkpoints)
        if hasattr(self.model.layoutxlm, "use_visual_backbone"
                   ) and self.model.layoutxlm.use_visual_backbone is False:
            self.use_visual_backbone = False

    def forward(self, x):
        if self.use_visual_backbone is True:
            image = x[4]
            entities = x[5]
            relations = x[6]
        else:
            image = None
            entities = x[4]
            relations = x[5]
        x = self.model(
            input_ids=x[0],
            bbox=x[1],
            attention_mask=x[2],
            token_type_ids=x[3],
            image=image,
            position_ids=None,
            head_mask=None,
            labels=None,
            entities=entities,
            relations=relations)
        return x


class BertForRe(NLPBaseModel):
    def __init__(self,
                 pretrained=True,
                 checkpoints=None,
                 mode="base",
                 **kwargs):
        super(BertForRe, self).__init__(BertModel, BertForRelationExtraction,
                                        mode, "re", pretrained, checkpoints)

    def forward(self, x):
        x = self.model(input_ids=x[0],
                       attention_mask=x[2],
                       entities=x[4],
                       relations=x[5],
                       token_type_ids=x[3])
        return x
