import argparse
import random
import math

import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from model.LSTM_LM import LSTM_LM

PAD = "<PAD>"
PAD_id = 0
UNK = "<UNK>"
UNK_id = 1
BOS = "<BOS>"
BOS_id = 2

max_char_length = 65


def build_dict(file_name, threshold=0):
    w2count = {}
    c2count = {}
    for line in open(file_name, "r"):
        line = line.rstrip()
        line_sp = line.split(" ")
        for token in line_sp:
            if token not in w2count:
                w2count[token] = 0
            w2count[token] += 1
            for c in token:
                if c not in c2count:
                    c2count[c] = 0
                c2count[c] += 1

    w2id = {PAD: PAD_id, UNK: UNK_id, BOS: BOS_id}
    c2id = {PAD: PAD_id, UNK: UNK_id, BOS: BOS_id}
    for token, count in w2count.items():
        if count > threshold:
            w2id[token] = len(w2id)
    for c, count in c2count.items():
        if count > threshold:
            c2id[c] = len(c2id)

    return w2id, c2id


def load(file_name, w2id, c2id):
    corpus = []
    for line in open(file_name, "r"):
        line = line.rstrip()
        line_sp = line.split(" ")
        char_bos = [BOS_id] + [PAD_id for i in range(max_char_length - 1)]
        corpus.append([[BOS_id], [char_bos]])  # word sequence, char sequence
        for token in line_sp:
            token_id = w2id.get(token, UNK_id)
            corpus[-1][0].append(token_id)
            corpus[-1][1].append([])
            for c in token:
                c_id = c2id.get(c, UNK_id)
                corpus[-1][1][-1].append(c_id)
            assert len(token) <= max_char_length
            corpus[-1][1][-1] = corpus[-1][1][-1] + [
                PAD_id for i in range(max_char_length - len(corpus[-1][1][-1]))
            ]
        corpus[-1][0] = torch.tensor(corpus[-1][0], dtype=torch.long)
        corpus[-1][1] = torch.tensor(corpus[-1][1], dtype=torch.long)
    return corpus


def main():
    parser = argparse.ArgumentParser(
        description="unsupervised neural word segmentation"
    )
    parser.add_argument("--gpu", "-g", type=int, default=-1, help="GPU ID")
    parser.add_argument(
        "--train", "-t", type=str, required=True, help="input train corpus (file name)"
    )
    parser.add_argument(
        "--test", type=str, required=True, help="input test corpus (file name)"
    )
    parser.add_argument("--epoch", "-e", type=int, default=10, help="epoch size")
    parser.add_argument("--batch", "-b", type=int, default=8, help="mini batch size")
    args = parser.parse_args()

    w2id, c2id = build_dict(args.train)
    train_corpus = load(args.train, w2id, c2id)
    test_corpus = load(args.test, w2id, c2id)

    model = LSTM_LM(len(w2id), len(c2id))
    device = torch.device("cpu")
    if args.gpu >= 0:
        device = torch.device("cuda")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    # train
    for epoch in range(args.epoch):
        model.train()
        corpus = train_corpus
        rand_list = [i for i in range(len(corpus))]
        random.shuffle(rand_list)
        total_loss = 0.0
        total_char = 0.0
        total_sent = 0.0
        total_word_count = 0
        entropy1 = 0.0
        entropy2 = 0.0
        for i in range(0, len(corpus), args.batch):
            # print(epoch, i, flush=True)
            model.zero_grad()
            length_list = []
            token_ids = []
            char_ids = []
            mask = []
            for j in range(i, min(i + args.batch, len(corpus))):
                r = rand_list[j]
                token_len = corpus[r][0].size(0)
                total_word_count += token_len
                length_list.append(token_len)
                token_ids.append(corpus[r][0])
                char_ids.append(corpus[r][1])
                mask.append(torch.ones(token_len, dtype=torch.uint8))
            arg_sorted_tensor = torch.argsort(
                torch.tensor(length_list), descending=True
            )
            sorted_token_ids = [token_ids[j] for j in arg_sorted_tensor]
            sorted_char_ids = [char_ids[j] for j in arg_sorted_tensor]
            sorted_mask = [mask[j] for j in arg_sorted_tensor]
            sorted_token_ids = pad_sequence(
                sorted_token_ids, batch_first=True, padding_value=PAD_id
            )
            sorted_char_ids = pad_sequence(
                sorted_char_ids, batch_first=True, padding_value=PAD_id
            )
            sorted_mask = pad_sequence(sorted_mask, batch_first=True)
            sorted_token_ids = sorted_token_ids.to(device)
            sorted_char_ids = sorted_char_ids.to(device)
            sorted_mask = sorted_mask.to(device)

            (
                loss,
                char_log_prob,
                sent_log_prob,
                sum_true_word_log2_prob,
                sum_true_word_times_true_word_log2_prob,
            ) = model(sorted_token_ids, sorted_char_ids, sorted_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_char += char_log_prob.item()
            total_sent += sent_log_prob.sum().item()
            entropy1 += sum_true_word_log2_prob
            entropy2 += sum_true_word_times_true_word_log2_prob
        print("train total_loss = {0}, total_char = {1}".format(total_loss, total_char))
        average_log_sent_prob = total_sent / float(len(corpus))
        entropy1 = -1.0 * entropy1 / float(total_word_count)
        try:
            perplexity1 = math.pow(2, entropy1)
        except OverflowError:
            perplexity1 = -1.0
        entropy2 = -1.0 * entropy2
        try:
            perplexity2 = math.pow(2, entropy2)
        except OverflowError:
            perplexity2 = -1.0
        print(
            "train average_log_sent_prob = {0}, perplexity1 = {1}, perplexity2 = {2}".format(
                average_log_sent_prob, perplexity1, perplexity2
            )
        )

        model.eval()
        corpus = test_corpus
        total_sent = 0.0
        token_word_count = 0
        entropy1 = 0.0
        entropy2 = 0.0
        for i in range(0, len(corpus), args.batch):
            with torch.no_grad():
                length_list = []
                token_ids = []
                char_ids = []
                mask = []
                for j in range(i, min(i + args.batch, len(corpus))):
                    r = j
                    token_len = corpus[r][0].size(0)
                    total_word_count += token_len
                    length_list.append(token_len)
                    token_ids.append(corpus[r][0])
                    char_ids.append(corpus[r][1])
                    mask.append(torch.ones(token_len, dtype=torch.uint8))
                arg_sorted_tensor = torch.argsort(
                    torch.tensor(length_list), descending=True
                )
                sorted_token_ids = [token_ids[j] for j in arg_sorted_tensor]
                sorted_char_ids = [char_ids[j] for j in arg_sorted_tensor]
                sorted_mask = [mask[j] for j in arg_sorted_tensor]
                sorted_token_ids = pad_sequence(
                    sorted_token_ids, batch_first=True, padding_value=PAD_id
                )
                sorted_char_ids = pad_sequence(
                    sorted_char_ids, batch_first=True, padding_value=PAD_id
                )
                sorted_mask = pad_sequence(sorted_mask, batch_first=True)
                sorted_token_ids = sorted_token_ids.to(device)
                sorted_char_ids = sorted_char_ids.to(device)
                sorted_mask = sorted_mask.to(device)

                (
                    _,
                    _,
                    sent_log_prob,
                    sum_true_word_log2_prob,
                    sum_true_word_times_true_word_log2_prob,
                ) = model(sorted_token_ids, sorted_char_ids, sorted_mask)
                total_sent += sent_log_prob.sum().item()
                entropy1 += sum_true_word_log2_prob
                entropy2 += sum_true_word_times_true_word_log2_prob
        average_log_sent_prob = total_sent / float(len(corpus))
        entropy1 = -1.0 * entropy1 / float(total_word_count)
        try:
            perplexity1 = math.pow(2, entropy1)
        except OverflowError:
            perplexity1 = -1.0
        entropy2 = -1.0 * entropy2
        try:
            perplexity2 = math.pow(2, entropy2)
        except OverflowError:
            perplexity2 = -1.0
        print(
            "test epoch = {0}, average_log_sent_prob = {1}, perplexity1 = {2}, perplexity2 = {3}".format(
                epoch, average_log_sent_prob, perplexity1, perplexity2
            )
        )


if __name__ == "__main__":
    main()
