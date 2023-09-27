import torch
torch.manual_seed(123)
import torch.nn as nn
import torch
import torch.nn as nn
from .ops import SelfAttention

class SAN(nn.Module):
    def __init__(self, num_of_dim, vocab_size, embedding_size, r, lstm_hidden_dim=128, da=128, hidden_dim=256) -> None:
        super(SAN, self).__init__()
        self._embedding = nn.Embedding(vocab_size, embedding_size)
        self._bilstm = nn.LSTM(embedding_size, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self._attention = SelfAttention(2 * lstm_hidden_dim, da, r)
        self._classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim * r, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_of_dim)
        )
        
    def forward(self, x: torch.Tensor):
        fmap = self._embedding(x)
        outputs, hc = self._bilstm(fmap)
        attn_mat = self._attention(outputs)
        m = torch.bmm(attn_mat, outputs)
        flatten = m.view(m.size()[0], -1)
        score = self._classifier(flatten)
        return score

    def _get_attention_weight(self, x):
        fmap = self._embedding(x)
        outputs, hc = self._bilstm(fmap)
        attn_mat = self._attention(outputs)
        m = torch.bmm(attn_mat, outputs)
        flatten = m.view(m.size()[0], -1)
        score = self._classifier(flatten)
        return score, attn_mat
