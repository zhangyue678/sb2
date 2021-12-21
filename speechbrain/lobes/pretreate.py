import os
import torch
from transformers import BertTokenizer


class pretreate(torch.nn.Module):
    def __init__(
        self,
        model_name = 'bert-base-uncased'
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
    
    def forward(self, x):
        token_code = self.tokenizer.encode_plus(x)
        input_id = token_code['input_ids']
        input_id = torch.tensor(input_id)
        print("input_id:", input_id)
        return input_id
