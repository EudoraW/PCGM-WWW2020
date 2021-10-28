# coding=utf-8
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


use_cuda = False
# use_cuda = torch.cuda.is_available()


class PCGM(nn.Module):
    def __init__(self, pretrain_embedding=None):
        super(PCGM, self).__init__()

        self.gain_embed = 1 # enable gain embedding
        self.embed_size = 768 # embedding size of BERT's output
        self.gain_embed_size = 0
        if self.gain_embed == 1:
            self.gain_embed_size = 150
        self.vocab_size = 11513
        self.gain_vocab_size = 4
        self.input_size = 768
        self.hidden_size = 100
        self.dropout_rate = 0.1
        self.padding_idx = 0
        self.max_passage_num = 20

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, self.padding_idx)
        if self.gain_embed == 1:
            self.gain_embedding = nn.Embedding(self.gain_vocab_size, self.gain_embed_size)

        if pretrain_embedding is not None: # load BERT embedding
            self.embedding.weight = nn.Parameter(torch.from_numpy(pretrain_embedding))
            self.embedding.weight.requires_grad = False

        self.passage_lstm = nn.LSTM(input_size=self.embed_size+self.gain_embed_size, hidden_size=self.hidden_size,
                                    num_layers=1, dropout=self.dropout_rate, batch_first=True)
        self.pred_mlp = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.Tanh(),
                                      nn.Dropout(p=self.dropout_rate),
                                      nn.Linear(self.hidden_size, 4))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, data, test_tag, mask_flag):
        passage = Variable(torch.from_numpy(np.array(data['passages_id'], dtype=np.int64)))

        if use_cuda:
            passage = passage.cuda()

        # print 'passage size:', passage.size(),
        batch_size = passage.size()[0]
        passage = self.embedding(passage) # batch_size, max_pass_num, embed_size
        # print passage.size()

        h0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        all_pred = np.array([[[1., 0., 0., 0.]] for _ in range(batch_size)], dtype=np.float32) # batch_size, 1, 4
        all_pred = Variable(torch.from_numpy(all_pred))
        # print all_pred.size()
        if use_cuda:
            h0, c0 = h0.cuda(), c0.cuda()
            all_pred = all_pred.cuda()

        for i in range(passage.size()[1]):
            if test_tag == 1: # use predicted gain of the previous passage as part of the input of current passage
                last_pred_p = all_pred[:,i].data.cpu().numpy().tolist() # batch_size, 1, 4
                last_passage_label = []
                for j in range(len(last_pred_p)):
                    p = np.array(last_pred_p[j], dtype=np.float32)
                    last_passage_label.append(np.random.choice([0, 1, 2, 3], p=p.ravel()))
                last_passage_label = np.array(last_passage_label, dtype=np.int64)
            else: # use gain label of the previous passage as part of the input of current passage
                last_passage_label = np.array(data['pre_gain'], dtype=np.int64)[:, i]
            # print last_passage_label.shape, last_passage_label
            p_mask = []
            for j in range(last_passage_label.shape[0]):
                if last_passage_label[j] == 0:
                    p_mask.append([[0., 0., 0., 0.]])
                elif last_passage_label[j] == 1:
                    p_mask.append([[1., 0., 0., 0.]])
                elif last_passage_label[j] == 2:
                    p_mask.append([[1., 1., 0., 0.]])
                elif last_passage_label[j] == 3:
                    p_mask.append([[1., 1., 1., 0.]])
            p_mask = Variable(torch.from_numpy(np.array(p_mask, dtype=np.float32))) # batch_size, 1, 4

            last_passage_label = Variable(torch.from_numpy(last_passage_label)).contiguous().view(batch_size, 1) # batch_size, 1
            if use_cuda:
                p_mask = p_mask.cuda()
                last_passage_label = last_passage_label.cuda()
            # print 'p_mask:', p_mask.size(), last_passage_label.size(),

            if self.gain_embed == 1:
                last_passage_label = self.gain_embedding(last_passage_label.contiguous().view(batch_size, 1))
                # print last_passage_label.size()
                lstm_input = torch.cat((passage[:,i].contiguous().view(batch_size, 1, -1), last_passage_label), dim=2)
            else:
                lstm_input = passage[:,i].contiguous().view(batch_size, 1, -1)
            # print 'lstm_input size:', lstm_input.size()

            p_pred, (h0, c0) = self.passage_lstm(lstm_input, (h0, c0))
            p_pred = self.pred_mlp(p_pred)

            if mask_flag == 0:
                norm_p_pred = self.softmax(p_pred)
            else:
                norm_p_pred = self.softmax(-1e31 * p_mask + p_pred)

            all_pred = torch.cat((all_pred, norm_p_pred), dim=1)
            # print all_pred

        all_pred = all_pred[:,1:] # batch_size, 20, 4
        # print 'pred size:', all_pred.size()

        return all_pred


if __name__ == '__main__':
    # a simple example
    gain = [[0 for _ in range(5)]+[1 for _ in range(5)]+[2 for _ in range(5)]+[3 for _ in range(5)] for j in range(3)]
    data = {'passages_id': [[i for i in range(20)] for j in range(3)],
            'gain': gain, # passage-level cumulative gain label
            'pre_gain': [[0]+gain[j][:len(gain[j])-1] for j in range(3)]}
    my_model = PCGM()
    test_tag = 0 # use gain label of the previous passage as the input of current passage
    mask_flag = 1 # enable gain mask
    pred = my_model(data, test_tag, mask_flag)
    print pred


