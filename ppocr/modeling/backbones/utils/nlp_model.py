from paddlenlp.transformers import BertModel, BertPretrainedModel
from paddlenlp.transformers.layoutxlm.modeling import REDecoder
import paddle.nn as nn


class BertForRelationExtraction(BertPretrainedModel):

    def __init__(self,
                 bert,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 dropout=None):
        super(BertForRelationExtraction, self).__init__()
        if isinstance(bert, dict):
            self.bert = BertModel(**bert)
        else:
            self.bert = bert

        self.extractor = REDecoder(hidden_size, hidden_dropout_prob)
        self.dropout = nn.Dropout(dropout if dropout is not None else self.bert.
                                  config["hidden_dropout_prob"])
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        entities=None,
        relations=None,
        token_type_ids=None,
        position_ids=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        seq_length = input_ids.shape[1]
        sequence_output = outputs[0][:, :seq_length]

        sequence_output = self.dropout(sequence_output)
        loss, pred_relations = self.extractor(sequence_output, entities,
                                              relations)

        res = dict(loss=loss, pred_relations=pred_relations)
        return res