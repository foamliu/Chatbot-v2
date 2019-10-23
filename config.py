import logging

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
d_model = 512
epochs = 10000
embedding_size = 300
hidden_size = 1024
data_file = 'data.pkl'
vocab_file = 'vocab.pkl'
vocab_size = 50000
maxlen_in = 50
maxlen_out = 50
# Training parameters
grad_clip = 1.0  # clip gradients at an absolute value of
print_freq = 50  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
IGNORE_ID = -1
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
num_train = 3216985
num_dev = 161301
num_test = 26835


train_filename = 'data/douban-multiturn-100w/train.txt'
dev_filename = 'data/douban-multiturn-100w/dev.txt'
test_filename = 'data/douban-multiturn-100w/test.txt'



def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()
