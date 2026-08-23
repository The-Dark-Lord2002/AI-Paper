"""Plain BERT classifier -- the "BERT" baseline column of Table 9 in Zhao et
al. (2025), "Causation Analysis of Marine Traffic Accidents Using Deep
Learning Approaches: A Case Study from China's Coasts" (Systems, 13(4):284).

The paper trains this alongside BERT + BiLSTM (model.py) specifically to
measure what the BiLSTM stage adds (Table 10: 88.7% vs. 89.8% on their own
data). Table 9 describes this baseline as using "BERT's 512-dimensional
hidden layer" with GELU activation and 3 dropout layers, but doesn't spell
out the exact layer-by-layer layout beyond that. This implements it as a
2-layer, 512-unit GELU MLP head on top of BERT's pooled [CLS] representation,
with a dropout before and after each dense layer plus one before the
classifier (3 total) -- a standard way to build a dropout-heavy classification
head at that width; adjust `hidden_dim`/`dropout` if you have the exact
paper architecture.
"""
import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        bert_name: str = "bert-base-uncased",
        hidden_dim: int = 512,
        dropout: float = 0.3,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        bert_hidden = self.bert.config.hidden_size
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(bert_hidden, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # BERT's own pooler: [CLS] hidden state through a Linear+Tanh, (batch, bert_hidden)
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        x = self.dropout1(pooled)
        x = self.activation(self.dense1(x))
        x = self.dropout2(x)
        x = self.activation(self.dense2(x))
        x = self.dropout3(x)
        return self.classifier(x)
