import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
from transformers import BertForSequenceClassification , BertModel, BertTokenizer, BertConfig
import torch
import torch.nn as nn


class BertPretrain(nn.Module):
    def __init__(
        self,
        model_name,
        num_labels=4,
        freeze=True,
        pretrain=True,
    ):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        self.config = BertConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels

        self.model = BertForSequenceClassification.from_pretrained(model_name, config=self.config).cuda()

        self.freeze = freeze

        if self.freeze:
            self.model.eval()
        else:
            for name, value in self.model.named_parameters():
                if 'layer.11'in name or  'layer.10'in name:
                    value.requires_grad = True
                elif name=='bert.pooler.dense.weight' or name=='bert.pooler.dense.bias' \
                        or name=='classifier.weight' or name=='classifier.bias':
                    value.requires_grad = True
                else:
                    value.requires_grad = False

    def forward(self, x):
        """Takes an input waveform and return its corresponding wav2vec encoding.
        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """
        outputs = self.model(x)
        return outputs
