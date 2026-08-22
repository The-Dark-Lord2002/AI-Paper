"""BERT + BiLSTM text classifier.

Architecture and hyperparameters follow Table 9 ("Model parameter settings")
of Zhao et al. (2025), "Causation Analysis of Marine Traffic Accidents Using
Deep Learning Approaches: A Case Study from China's Coasts" (Systems,
13(4):284) for their BERT + BiLSTM configuration: a pretrained BERT encoder
produces contextual token embeddings, which are fed into a 128-unit
bidirectional LSTM to model sequential dependencies across the narrative, a
Mish activation adds nonlinearity over the pooled BiLSTM state, and a dense
softmax classification head produces the final category. Two dropout layers
bracket the BiLSTM, matching the paper's "Dropout Layer: 2" setting.
"""
import torch
import torch.nn as nn
from transformers import BertModel


class BertBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        bert_name: str = "bert-base-uncased",
        lstm_hidden: int = 128,
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
        # Two dropout layers bracketing the BiLSTM, per Table 9 ("Dropout Layer: 2").
        self.dropout_pre_lstm = nn.Dropout(dropout)
        self.dropout_post_lstm = nn.Dropout(dropout)
        self.mish = nn.Mish()
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        sequence_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (batch, seq_len, bert_hidden)
        sequence_output = self.dropout_pre_lstm(sequence_output)

        _, (h_n, _) = self.bilstm(sequence_output)
        # h_n: (num_layers * 2, batch, lstm_hidden); last layer's forward/backward states
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]
        pooled = torch.cat([forward_hidden, backward_hidden], dim=1)

        pooled = self.mish(pooled)
        pooled = self.dropout_post_lstm(pooled)
        return self.classifier(pooled)
