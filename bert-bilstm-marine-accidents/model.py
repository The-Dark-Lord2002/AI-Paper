"""BERT + BiLSTM text classifier.

Architecture follows the hybrid design used in Xu et al. (2025), "Causation
Analysis of Marine Traffic Accidents Using Deep Learning Approaches: A Case
Study from China's Coasts" (Systems, 13(4):284): a pretrained BERT encoder
produces contextual token embeddings, which are fed into a bidirectional LSTM
to model sequential dependencies across the narrative, followed by a dense
softmax classification head over the incident categories.
"""
import torch
import torch.nn as nn
from transformers import BertModel


class BertBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        bert_name: str = "bert-base-uncased",
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        bert_hidden = self.bert.config.hidden_size
        self.bilstm = nn.LSTM(
            input_size=bert_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        sequence_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (batch, seq_len, bert_hidden)

        _, (h_n, _) = self.bilstm(sequence_output)
        # h_n: (num_layers * 2, batch, lstm_hidden); last layer's forward/backward states
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]
        pooled = torch.cat([forward_hidden, backward_hidden], dim=1)

        pooled = self.dropout(pooled)
        return self.classifier(pooled)
