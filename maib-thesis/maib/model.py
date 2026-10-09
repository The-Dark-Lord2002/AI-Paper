"""
مدل: رمزگذار BERT + «سر» طبقه‌بندی (head).

head:
  cls         : بردار [CLS] + لایه‌ی خطی                       ← مبنای «BERT ساده» (گام ۲-ب پروپوزال)
  mean        : میانگین بردار توکن‌ها                           ← معادل «BERT + Pooling» مقاله‌ی مرجع
  attention   : توجه افزایشی مستقیم روی خروجی BERT              ← ablation (بدون BiLSTM)
  bilstm      : BiLSTM روی خروجی BERT، فقط حالت پایانی دو جهت    ← بازتولید مقاله‌ی مرجع (گام ۲-ج)
  bilstm_att  : BiLSTM + توجه روی همه‌ی حالت‌های گام‌به‌گام       ← روش پیشنهادی (گام ۳)

توجه افزایشی (Bahdanau / Zhou et al. 2016):
  u_t = tanh(W h_t + b),   a_t = softmax_t(v^T u_t),   z = Σ_t a_t h_t
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

HEADS = ["cls", "mean", "attention", "bilstm", "bilstm_att"]


class AttentionPooling(nn.Module):
    def __init__(self, dim: int, att_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(dim, att_dim)
        self.v = nn.Linear(att_dim, 1, bias=False)

    def forward(self, H, mask):
        scores = self.v(torch.tanh(self.proj(H))).squeeze(-1).float()        # (B, T)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=-1)                                # (B, T), جمع = ۱
        z = torch.bmm(alpha.unsqueeze(1).to(H.dtype), H).squeeze(1)          # (B, dim)
        return z, alpha


class IncidentClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, head: str = "bilstm_att",
                 lstm_hidden: int = 128, att_dim: int = 128, dropout: float = 0.1,
                 grad_checkpointing: bool = False):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}")
        self.head = head
        self.encoder = AutoModel.from_pretrained(model_name)
        if grad_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        hidden = self.encoder.config.hidden_size                             # 768 در bert-base

        self.lstm = None
        feat = hidden
        if head.startswith("bilstm"):
            # BiLSTM با ۱۲۸ واحد در هر جهت (تنظیم مقاله‌ی مرجع) → خروجی ۲۵۶ بعدی
            self.lstm = nn.LSTM(hidden, lstm_hidden, batch_first=True, bidirectional=True)
            feat = 2 * lstm_hidden
        self.att = AttentionPooling(feat, att_dim) if head in ("attention", "bilstm_att") else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat, num_labels)

    def head_parameters(self):
        mods = [self.classifier] + [m for m in (self.lstm, self.att) if m is not None]
        return [p for m in mods for p in m.parameters()]

    def _run_lstm(self, H, mask):
        lengths = mask.sum(1).cpu()
        packed = pack_padded_sequence(H.float(), lengths, batch_first=True, enforce_sorted=False)
        with torch.autocast(device_type=H.device.type, enabled=False):
            out, (h_n, _) = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=H.size(1))
        last = torch.cat([h_n[0], h_n[1]], dim=-1)                           # پایان رو به جلو + پایان رو به عقب
        return out, last

    def forward(self, input_ids, attention_mask, token_type_ids=None, return_attention=False):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        H = self.encoder(**kw).last_hidden_state                             # (B, T, 768)

        alpha = None
        if self.head == "cls":
            z = H[:, 0]
        elif self.head == "mean":
            m = attention_mask.unsqueeze(-1).to(H.dtype)
            z = (H * m).sum(1) / m.sum(1).clamp(min=1.0)
        elif self.head == "attention":
            z, alpha = self.att(H, attention_mask)
        elif self.head == "bilstm":
            _, z = self._run_lstm(H, attention_mask)
        else:  # bilstm_att
            out, _ = self._run_lstm(H, attention_mask)
            z, alpha = self.att(out, attention_mask)

        logits = self.classifier(self.dropout(z.float()))
        return (logits, alpha) if return_attention else logits
