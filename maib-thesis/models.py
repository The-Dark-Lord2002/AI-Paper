"""
The three neural models of the thesis.

All three start the same way: BERT reads the report and produces one 768-number vector per word-piece.
They differ only in how those many vectors are turned into ONE vector for the final classifier:

  bert             take the vector of the special [CLS] token                       (proposal step 2b)
  bert_bilstm      run a BiLSTM over the vectors, take its two final states         (step 2c, reference paper)
  bert_bilstm_att  run a BiLSTM, then attention = weighted average of ALL states    (step 3, proposed)

Why attention: the words that decide the class ("grounded", "smoke") are often in the middle of the
report. The BiLSTM's final state may have mostly forgotten them; attention can give them a high weight.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

import config

MODEL_NAMES = ["bert", "bert_bilstm", "bert_bilstm_att"]


class Attention(nn.Module):
    """
    Additive attention over a sequence of vectors (Zhou et al., 2016):

        score_t = v . tanh(W h_t)        one number per word: how useful is word t?
        alpha   = softmax(score)         turn scores into weights that sum to 1
        output  = sum_t alpha_t * h_t    weighted average of all word vectors

    alpha is also returned, so we can later show which words the model looked at (step 6).
    """

    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, h, mask):
        # h: (batch, words, dim)    mask: (batch, words), 1 = real word, 0 = padding
        scores = self.v(torch.tanh(self.W(h))).squeeze(-1)                  # (batch, words)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)  # padding gets weight 0
        alpha = torch.softmax(scores, dim=-1)                               # (batch, words)
        output = (alpha.unsqueeze(-1) * h).sum(dim=1)                       # (batch, dim)
        return output, alpha


class IncidentClassifier(nn.Module):

    def __init__(self, model_name, num_classes):
        super().__init__()
        assert model_name in MODEL_NAMES, f"model must be one of {MODEL_NAMES}"
        self.model_name = model_name

        self.bert = AutoModel.from_pretrained(config.BERT_NAME)
        bert_dim = self.bert.config.hidden_size                              # 768 for bert-base
        feature_dim = bert_dim

        if model_name in ("bert_bilstm", "bert_bilstm_att"):
            self.lstm = nn.LSTM(bert_dim, config.LSTM_HIDDEN, batch_first=True, bidirectional=True)
            feature_dim = 2 * config.LSTM_HIDDEN                             # forward + backward = 256
        if model_name == "bert_bilstm_att":
            self.attention = Attention(feature_dim, config.ATTENTION_DIM)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def run_bilstm(self, h, mask):
        """
        Run the BiLSTM only over the real words of each report, not over the padding.
        Returns all per-word states (for attention) and the final states (for the reference model).
        """
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        packed_states, (final, _) = self.lstm(packed)
        states, _ = pad_packed_sequence(packed_states, batch_first=True, total_length=h.size(1))
        last = torch.cat([final[0], final[1]], dim=1)     # forward LSTM's last state + backward LSTM's last state
        return states, last

    def forward(self, input_ids, attention_mask):
        """Returns (logits, alpha). alpha = attention weights, or None for the models without attention."""
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (batch, words, 768)

        # The small layers on top run in float32 even when BERT runs in float16:
        # LSTMs and softmax are numerically unstable in float16.
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            alpha = None
            if self.model_name == "bert":
                vector = h[:, 0]                                             # the [CLS] token
            elif self.model_name == "bert_bilstm":
                _, vector = self.run_bilstm(h, attention_mask)
            else:
                states, _ = self.run_bilstm(h, attention_mask)
                vector, alpha = self.attention(states, attention_mask)
            logits = self.classifier(self.dropout(vector))                   # (batch, num_classes)
        return logits, alpha
