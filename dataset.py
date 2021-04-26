from gensim.models import KeyedVectors
import torch
import torch.nn as nn
from torchtext.vocab import Vectors
from torchtext.datasets import text_classification
import os
from torchtext import data, datasets
from torchtext.vocab import GloVe

# 필드 객체 정
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)


NGRAMS = 2
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root='./.data', ngrams=NGRAMS, vocab=None).splits(TEXT, LABEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
TEXT.build_vocab(train_dataset, vectors=GloVe(name='6B', dim=300), max_size=10000, min_freq=10)
LABEL.build_vocab(train_dataset)

embedding = TEXT.vocab.vectors

print(TEXT.vocab.vectors[9999])