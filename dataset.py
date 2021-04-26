import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
from torchtext.data import TabularDataset

import random
import pandas as pd
from torchtext.data import Iterator

def dataset(BATCH_SIZE):
    SEED = 5
    random.seed(SEED)
    torch.manual_seed(SEED)

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

    TEXT = data.Field(sequential=True, batch_first=True, lower=True)
    LABEL = data.Field(sequential=False, batch_first=True, is_target=True)

    #train_df = pd.read_csv('./.data/train.csv')
    #test_df = pd.read_csv('./.data/test.csv')

    train_data, test_data = TabularDataset.splits(path='.', train='./.data/train.csv', test='./.data/test.csv', format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=False)

    # 단어 집합 만들기
    TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
    LABEL.build_vocab(train_data)
    vocab_size = len(TEXT.vocab)


    # 데이터로더 만들기
    train_data, val_data = train_data.split(split_ratio=0.8)
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train_data, val_data, test_data), batch_size=BATCH_SIZE,
            shuffle=True, repeat=False, sort=False)


    train_loader = Iterator(dataset=train_data, batch_size=BATCH_SIZE)
    test_loader = Iterator(dataset=test_data, batch_size=BATCH_SIZE)

    return vocab_size, train_loader, test_loader, train_iter, val_iter, test_iter
