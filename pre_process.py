import pickle

import numpy as np
from tqdm import tqdm

from config import train_filename, vocab_file, maxlen_in, maxlen_out, sos_id, \
    eos_id, unk_id
from utils import encode_text


def build_vocab(token):
    if token not in char2idx:
        next_index = len(char2idx)
        char2idx[token] = next_index
        idx2char[next_index] = token


def process(file):
    print('processing {}...'.format(file))

    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()

    lengths = []

    for line in tqdm(data):
        sentences = line.split('|')
        for sent in sentences:
            sentence = sent.strip()
            lengths.append(len(sentence))
            tokens = list(sentence)
            for token in tokens:
                build_vocab(token)

    np.save('lengths.npz', np.array(lengths))


def get_data(in_file):
    print('getting data {}...'.format(in_file))
    with open(in_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    samples = []
    for line in lines:
        sentences = line.split('|')
        in_sentence = sentences[0]
        out_sentence = sentences[1]

        tokens = list(in_sentence)
        in_data = encode_text(char2idx, tokens)

        tokens = list(out_sentence)
        out_data = [sos_id] + encode_text(char2idx, tokens) + [eos_id]

        # if len(in_data) < maxlen_in and len(out_data) < maxlen_out \
        #         and unk_id not in in_data and unk_id not in out_data:
        samples.append((in_data, out_data))
    return samples


if __name__ == '__main__':
    char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    idx2char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}

    process(train_filename)
    print(len(char2idx))
    print(list(char2idx.items())[:100])

    data = {
        'dict': {
            'char2idx': char2idx,
            'idx2char': idx2char
        }
    }
    with open(vocab_file, 'wb') as file:
        pickle.dump(data, file)

    samples = get_data(train_filename)
    print('num_samples: ' + str(len(samples)))

    # data = {
    #     'train': train,
    #     'dev': dev,
    #     'test': test
    # }
    #
    # print('num_train: ' + str(len(train)))
    # print('num_dev: ' + str(len(dev)))
    # print('num_test: ' + str(len(test)))

    # with open(data_file, 'wb') as file:
    #     pickle.dump(data, file)
