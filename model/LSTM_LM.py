
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD_id = 0
UNK = "<UNK>"
UNK_id = 1

word_emb_dim = 50
use_char_cnn = True
char_emb_dim = 50  # == hidden_dim
char_cnn_window_size = 3
char_cnn_dim = 30
input_lstm_dim = word_emb_dim + char_cnn_dim
hidden_dim = 50
dropout_rate = 0.5
max_char_length = 65

# char_learning_rate = 0.1

class LSTM_LM(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size):
        super(LSTM_LM, self).__init__()
        self.word_emb = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=PAD_id)
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=PAD_id)
        self.char_cnn = nn.Conv1d(char_emb_dim, char_cnn_dim, char_cnn_window_size)
        self.char_pool = nn.MaxPool1d(max_char_length - (char_cnn_window_size - 1), stride=1)
        self.lstm = nn.LSTM(input_lstm_dim, hidden_dim, batch_first=True)
        self.char_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.word_classifier = nn.Linear(hidden_dim, word_vocab_size)
        self.char_classifier = nn.Linear(hidden_dim, char_vocab_size)

        self.smoothing = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, bacthed_word_data, bacthed_char_data, mask):
        """
        parameters
        ----------
            batched_word_data : tensor, size(batch size, maximum length of word tokens), dtype=torch.long
                input word level token indexes
            batched_char_data : tensor,
                size(batch size, maximum length of word tokens, maximum length of char tokens), dtype=torch.long
                input character level token indexes
            mask : tensor, size(batch size, maximum length of word tokens), dtype=torch.uint8
                mask, 0 means padding
        returns
        -------
            loss : tensor, size(1), dtype=torch.float
        """
        batch_size = bacthed_word_data.size(0)
        seq_len = bacthed_word_data.size(1)

        word_emb = self.word_emb(bacthed_word_data)
        if use_char_cnn:
            char_emb = self.char_emb(bacthed_char_data)
            char_emb = self.dropout(char_emb)
            char_cnn = self.char_cnn(
                char_emb.view(
                    batch_size * seq_len,
                    max_char_length,
                    char_emb_dim).transpose(
                    1,
                    2))
            char_cnn = self.char_pool(char_cnn)
            char_cnn = char_cnn.view(batch_size, seq_len, char_cnn_dim)
            word_emb = torch.cat([word_emb, char_cnn], dim=2)

        word_emb = self.dropout(word_emb)
        packed_word_emb = pack_padded_sequence(word_emb, torch.sum(mask, dim=1).long(), batch_first=True)
        word_rep, (_, _) = self.lstm(packed_word_emb)
        # word_rep, (_, _) = self.lstm(word_emb)
        word_rep, _ = pad_packed_sequence(word_rep, batch_first=True)
        word_rep = self.dropout(word_rep)

        next_word_score = self.word_classifier(word_rep)
        next_word_score = next_word_score[:, :-1, :]
        next_word = bacthed_word_data[:, 1:]
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_id, reduction="none")
        word_loss = loss_fn(next_word_score.reshape(batch_size * (seq_len -1), -1), next_word.reshape(-1))

        # char
        input_char_lstm = torch.cat([word_rep.reshape(batch_size, seq_len, 1, -1)[:, :-1], char_emb[:, 1:]], dim=2)
        input_char_lstm = input_char_lstm.reshape(batch_size * (seq_len - 1), max_char_length + 1, hidden_dim)
        char_rep, (_, _) = self.char_lstm(input_char_lstm)
        char_rep = self.dropout(char_rep)
        char_rep = char_rep.reshape(batch_size, (seq_len - 1), max_char_length + 1, hidden_dim)
        next_char_score = self.char_classifier(char_rep)
        next_char_score = next_char_score[:, :, :-1, :]
        next_char = bacthed_char_data[:, 1:, :]
        next_char_score = next_char_score.reshape(batch_size * (seq_len - 1) * max_char_length, -1)
        next_char = next_char.reshape(-1)
        char_loss = loss_fn(next_char_score, next_char)

        # smoothing
        g = self.smoothing(word_rep)
        sigmoid = nn.Sigmoid()
        g = sigmoid(g[:, :-1, 0])

        # calc log prob
        word_log_prob = -1.0 * word_loss.reshape(batch_size, seq_len - 1)
        char_log_prob = -1.0 * char_loss.reshape(batch_size, seq_len - 1, max_char_length)
        word_prob = torch.exp(word_log_prob)
        char_prob = torch.exp(char_log_prob)
        char_prob = torch.prod(char_prob, dim=2)
        true_word_prob = (1 - g) * word_prob + g * char_prob
        true_word_log_prob = torch.log(true_word_prob)
        sent_log_prob = torch.sum(true_word_log_prob, dim=1)
        loss = sent_log_prob.mean() * -1.0
        each_seq_len = torch.sum(mask, dim=1).long() -1
        sum_true_word_log2_prob = 0.0
        sum_true_word_times_true_word_log2_prob = 0.0
        for b in range(each_seq_len.size(0)):
            for t in range(each_seq_len[b]):
                log2_prob = torch.log2(true_word_prob[b, t]).item()
                sum_true_word_log2_prob += log2_prob
                sum_true_word_times_true_word_log2_prob += true_word_prob[b, t].item() * log2_prob

        # 本当は \prod p(w_i) を最大化したい
        # p(w) = (1 - g) * pw(w) + g + pc(w)
        # pc(w) = \prod p(w) pc(c_i)
        # しかし、単純に計算するとオーバーフローやアンダーフローを起こしそうなので簡略化
        # word_loss = -1.0 * torch.sum(torch.log(1 - g) + word_log_prob, dim=1)
        # log_pc_w = torch.sum(char_log_prob, dim=2)
        # char_loss = -1.0 * torch.sum(torch.log(g) + log_pc_w, dim=1)
        # loss = word_loss.mean() + char_learning_rate * char_loss.mean()

        return loss, char_prob.mean(), sent_log_prob, sum_true_word_log2_prob, sum_true_word_times_true_word_log2_prob