import torch
import torchtext

import os
from model import HierarchialAttentionNetwork
from utils import generate_batch, train_func, test
import time
from torch.utils.data.dataset import random_split




BATCH_SIZE = 16
EMBED_DIM = 32
N_EPOCHS = 5
min_valid_loss = float('inf')

vocab_size = len(train_dataset.get_vocab())
emb_size = train_dataset.get_em
n_classes = len(train_dataset.get_labels())

model = HierarchialAttentionNetwork(n_classes=n_classes,
                                    vocab_size=vocab_size,
                                    emb_size=emb_size,
                                    word_rnn_size=word_rnn_size,
                                    sentence_rnn_size=sentence_rnn_size,
                                    word_rnn_layers=word_rnn_layers,
                                    sentence_rnn_layers=sentence_rnn_layers,
                                    word_att_size=word_att_size,
                                    sentence_att_size=sentence_att_size,
                                    dropout=dropout).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_, optimizer, device, model, criterion, scheduler)
    valid_loss, valid_acc = test(sub_valid_, device, model, criterion)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')