import pickle
from collections import Counter

import jieba
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import train_filename, dev_filename, test_filename, vocab_file, maxlen_in, maxlen_out, data_file, sos_id, \
    eos_id, unk_id
from utils import encode_text


def build_vocab(token, word2idx, idx2char):
    if token not in word2idx:
        next_index = len(word2idx)
        word2idx[token] = next_index
        idx2char[next_index] = token


def process(file):
    print('processing {}...'.format(file))
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()

    word_freq = Counter()
    lengths = []

    for line in tqdm(data):
        sentences = line.split('\t')
        for sent in sentences:
            sentence = sent.strip()
            seg_list = jieba.cut(sentence.strip())
            tokens = list(seg_list)
            word_freq.update(list(tokens))
            vocab_size = n_tgt_vocab

            lengths.append(len(tokens))

    words = word_freq.most_common(vocab_size - 4)
    word_map = {k[0]: v + 4 for v, k in enumerate(words)}
    word_map['<pad>'] = 0
    word_map['<sos>'] = 1
    word_map['<eos>'] = 2
    word_map['<unk>'] = 3
    print(len(word_map))
    print(words[:100])

    n, bins, patches = plt.hist(lengths, 50, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('Lengths')
    plt.ylabel('Probability')
    plt.title('Histogram of Lengths')
    plt.grid(True)
    plt.show()

    word2idx = word_map
    idx2char = {v: k for k, v in word2idx.items()}

    return word2idx, idx2char


def get_data(in_file):
    print('getting data {}...'.format(in_file))
    with open(in_file, 'r', encoding='utf-8') as file:
        in_lines = file.readlines()

    samples = []

    for i in tqdm(range(len(in_lines))):
        line = in_lines[i].strip()
        sentences = line.split('\t')
        for j in range(1, len(sentences) - 1):
            sentence = sentences[j]
            tokens = jieba.cut(sentence.strip())
            in_data = encode_text(char2idx, tokens)

            sentence = sentences[j + 1]
            tokens = jieba.cut(sentence.strip())
            out_data = [sos_id] + encode_text(char2idx, tokens) + [eos_id]

        if len(in_data) < maxlen_in and len(out_data) < maxlen_out and unk_id not in in_data and unk_id not in out_data:
            samples.append({'in': in_data, 'out': out_data})
    return samples


if __name__ == '__main__':
    char2idx, idx2char = process(train_filename)

    print(len(char2idx))

    data = {
        'dict': {
            'char2idx': char2idx,
            'idx2char': idx2char
        }
    }
    with open(vocab_file, 'wb') as file:
        pickle.dump(data, file)

    train = get_data(train_filename)
    dev = get_data(dev_filename)
    test = get_data(test_filename)

    data = {
        'train': train,
        'dev': dev,
        'test': test
    }

    print('num_train: ' + str(len(train)))
    print('num_dev: ' + str(len(dev)))
    print('num_test: ' + str(len(test)))

    with open(data_file, 'wb') as file:
        pickle.dump(data, file)
