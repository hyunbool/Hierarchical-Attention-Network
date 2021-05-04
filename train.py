import torch
NGRAMS = 2
import os
from model import HierarchialAttentionNetwork
from utils import train, evaluate
from dataset import *
import json

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

data_path = "./data/"


if not os.path.isfile("./data/TRAIN_data.pth.tar"):
    create_input_files(csv_folder='./data',
                       output_folder='./data',
                       segment_limit=3,
                       sentence_limit=20,
                       word_limit=20,
                       min_word_count=5)
if not os.path.isfile("./data/word2vec_model"):
    train_word2vec_model(data_folder='./data', algorithm='skipgram')

word2vec_file = data_path + 'word2vec_model'
with open(data_path + 'word_map.json', 'r') as j:
    word_map = json.load(j)

# 하이퍼파라미터
batch_size = 16
LEARNING_RATE = 0.001
epochs = 5
n_classes = 2
embed_dim = 32
word_rnn_size = 50  # word RNN size
sentence_rnn_size = 50  # character RNN size
segment_rnn_size = 50
word_rnn_layers = 1  # number of layers in character RNN
sentence_rnn_layers = 1  # number of layers in word RNN
segment_rnn_layers = 1
word_att_size = 100  # size of the word-level attention layer (also the size of the word context vector)
sentence_att_size = 100  # size of the sentence-level attention layer (also the size of the sentence context vector)
segment_att_size = 100
dropout = 0.3  # dropout
fine_tune_word_embeddings = True  # fine-tune word embeddings?
vocab_size = len(word_map)
workers = 4


embeddings, emb_size = load_word2vec_embeddings(word2vec_file, word_map)  # load pre-trained word2vec embeddings
model = HierarchialAttentionNetwork(n_classes=n_classes,
                                    vocab_size=vocab_size,
                                    emb_size=embed_dim,
                                    word_rnn_size=word_rnn_size,
                                    sentence_rnn_size=sentence_rnn_size,
                                    segment_rnn_size=segment_rnn_size,
                                    word_rnn_layers=word_rnn_layers,
                                    sentence_rnn_layers=sentence_rnn_layers,
                                    segment_rnn_layers=segment_rnn_layers,
                                    word_att_size=word_att_size,
                                    sentence_att_size=sentence_att_size,
                                    segment_att_size=segment_att_size,
                                    dropout=dropout)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

criterion = torch.nn.CrossEntropyLoss().to(device)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)


train_loader = torch.utils.data.DataLoader(HANDataset(data_path, 'train'), batch_size=batch_size, shuffle=True,
                                           num_workers=workers, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(HANDataset(data_path, 'valid'), batch_size=batch_size, shuffle=True,
                                           num_workers=workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(HANDataset(data_path, 'test'), batch_size=batch_size, shuffle=True,
                                           num_workers=workers, pin_memory=True)

best_val_loss = None
for e in range(1, epochs+1):
    train(train_loader=train_loader,
          model=model,
          criterion=criterion,
          optimizer=optimizer,
          epoch=epochs,
          device=device)
    val_loss, val_accuracy = evaluate(model=model, test_loader=valid_loader, criterion=criterion, device=device)

    # 검증 오차가 가장 적은 최적의 모델을 저장
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        # Save checkpoint
        state = {'epoch': epochs,
                 'model': model,
                 'optimizer': optimizer,
                 'word_map': word_map}
        filename = 'checkpoint_han.pth.tar'
        torch.save(state, filename)
        best_val_loss = val_loss



