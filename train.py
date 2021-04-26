import torch
import torchtext
from torchtext.datasets import text_classification
NGRAMS = 2
import os
from model import GRU
from utils import train, evaluate
import time
from torch.utils.data.dataset import random_split
from dataset import dataset

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

# 하이퍼파라미터
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 5
N_CLASSES = 2
EMBED_DIM = 32
min_valid_loss = float('inf')

vocab_size, train_loader, test_loader, train_iter, val_iter, test_iter = dataset(BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GRU(1, 256, vocab_size, 128, N_CLASSES, 0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

criterion = torch.nn.CrossEntropyLoss().to(device)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

best_val_loss = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_iter, DEVICE)
    val_loss, val_accuracy = evaluate(model, val_iter, DEVICE)

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss