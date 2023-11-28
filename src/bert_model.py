import os

import torch
import torch.nn as nn
from transformers import AutoModel, logging

logging.set_verbosity_error()

# model settings
NUM_LABELS = 2
BERT_ENCODER_OUTPUT_SIZE = 768
CLF_LAYER_1_DIM = 64
CLF_DROPOUT_PROB = 0.4


class BertClassifier(nn.Module):
    def __init__(self, name):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = BERT_ENCODER_OUTPUT_SIZE, CLF_LAYER_1_DIM, NUM_LABELS
        self.bert = AutoModel.from_pretrained(name)

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(CLF_DROPOUT_PROB),
            nn.Linear(H, D_out),
        )

        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits


if __name__ == "__main__":
    model = BertClassifier("distilbert-base-uncased")
