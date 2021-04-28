import torch
NGRAMS = 2
import os
from model import GRU, HierarchialAttentionNetwork
from utils import *
from dataset import *
import json

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

data_path = "./data/"

if not os.path.isfile("./data/TRAIN_data.pth.tar"):
    create_input_files(csv_folder='./data',
                       output_folder='./data',
                       sentence_limit=15,
                       word_limit=20,
                       min_word_count=5)
if not os.path.isfile("./data/word2vec_model"):
    train_word2vec_model(data_folder='./data', algorithm='skipgram')

word2vec_file = data_path + 'word2vec_model'
with open(data_path + 'word_map.json', 'r') as j:
    word_map = json.load(j)

# 하이퍼파라미터
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 5
N_CLASSES = 2
EMBED_DIM = 32

# Model parameters
word_rnn_size = 50  # word RNN size
sentence_rnn_size = 50  # character RNN size
word_rnn_layers = 1  # number of layers in character RNN
sentence_rnn_layers = 1  # number of layers in word RNN
word_att_size = 100  # size of the word-level attention layer (also the size of the word context vector)
sentence_att_size = 100  # size of the sentence-level attention layer (also the size of the sentence context vector)
dropout = 0.3  # dropout
fine_tune_word_embeddings = True  # fine-tune word embeddings?
vocab_size = len(word_map)
workers = 4


embeddings, emb_size = load_word2vec_embeddings(word2vec_file, word_map)  # load pre-trained word2vec embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#model = GRU(1, 256, vocab_size, 128, N_CLASSES, 0.5).to(device)
model = HierarchialAttentionNetwork(n_classes=N_CLASSES,
                                    vocab_size=vocab_size,
                                    emb_size=EMBED_DIM,
                                    word_rnn_size=word_rnn_size,
                                    sentence_rnn_size=sentence_rnn_size,
                                    word_rnn_layers=word_rnn_layers,
                                    sentence_rnn_layers=sentence_rnn_layers,
                                    word_att_size=word_att_size,
                                    sentence_att_size=sentence_att_size,
                                    dropout=dropout)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

criterion = torch.nn.CrossEntropyLoss().to(device)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_loader = torch.utils.data.DataLoader(HANDataset(data_path, 'train'), batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(HANDataset(data_path, 'test'), batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=workers, pin_memory=True)

best_val_loss = None
for e in range(1, EPOCHS+1):
    train(train_loader=train_loader,
          model=model,
          criterion=criterion,
          optimizer=optimizer,
          epoch=EPOCHS,
          device=DEVICE)
    val_loss, val_accuracy = evaluate(model=model, test_loader=test_loader, criterion=criterion, device=DEVICE)

    # 검증 오차가 가장 적은 최적의 모델을 저장
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        # Save checkpoint
        state = {'epoch': EPOCHS,
                 'model': model,
                 'optimizer': optimizer,
                 'word_map': word_map}
        filename = 'checkpoint_han.pth.tar'
        torch.save(state, filename)
        best_val_loss = val_loss



