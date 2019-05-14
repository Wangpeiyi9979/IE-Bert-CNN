from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import torch.nn as nn
import numpy as np

class Embedder(nn.Module):
    def __init__(self, emb_method='glove', glove_param=None, elmo_param=None, use_gpu=True):
        super(Embedder, self).__init__()

        if 'glove' in emb_method and glove_param is None:
            raise RuntimeError('glove_param is not provided')

        if 'elmo' in emb_method and elmo_param is None:
            raise RuntimeError('elmo_param is not provided')

        if emb_method == 'glove' and glove_param['use_id'] == False and glove_param['word2id_file'] is None:
            raise RuntimeError('word2id_file is not provided')

        self.use_gpu = use_gpu
        self.emb_method = emb_method

        if emb_method == 'elmo':
            self.elmo_param = elmo_param
            self.init_elmo()

        elif emb_method == 'glove':
            self.glove_param = glove_param
            self.init_glove()

        elif emb_method == 'elmo_glove':
            self.glove_param = glove_param
            self.elmo_param = elmo_param
            self.init_elmo()
            self.init_glove()
            self.word_dim = elmo_param['elmo_dim'] + glove_param['glove_dim']

    def init_elmo(self):
        self.elmo = Elmo(self.elmo_param['elmo_options_file'], self.elmo_param['elmo_weight_file'], 1, requires_grad=self.elmo_param['requires_grad'])
        self.word_dim = self.elmo_param['elmo_dim']

    def init_glove(self):

        if self.glove_param['use_id'] == False:
            self.word2id = np.load(self.glove_param['word2id_file']).tolist()
        self.glove = nn.Embedding(self.glove_param['vocab_size'], self.glove_param['glove_dim'])
        if not self.glove_param['glove_file'] is None:
            emb = torch.from_numpy(np.load(self.glove_param['glove_file']))
            if self.use_gpu is True:
                emb = emb.cuda()
            self.glove.weight.data.copy_(emb)
        if self.glove_param['requires_grad'] == False:
            self.glove.weight.requires_grad = False

        self.word_dim = self.glove_param['glove_dim']


    def get_elmo(self, sentence_lists):
        character_ids = batch_to_ids(sentence_lists)
        if self.use_gpu is True:
            character_ids = character_ids.cuda()
        embeddings = self.elmo(character_ids)
        return embeddings['elmo_representations'][0]

    def get_glove(self, sentence_lists):

        if self.glove_param['use_id']  is True:
            sentence_lists = torch.LongTensor(sentence_lists)
            if self.use_gpu is True:
                sentence_lists = sentence_lists.cuda()
            return self.glove(sentence_lists)

        max_len = max(map(lambda x: len(x), sentence_lists))
        sentence_lists = list(map(lambda x: list(map(lambda w: self.word2id.get(w, 0), x)), sentence_lists))
        sentence_lists = list(map(lambda x: x + [self.glove_param['vocab_size']-1] * (max_len - len(x)), sentence_lists))
        sentence_lists = torch.LongTensor(sentence_lists)
        if self.use_gpu is True:
            sentence_lists = sentence_lists.cuda()
        embeddings = self.glove(sentence_lists)
        return embeddings

    def forward(self, sentence_lists):

        if self.emb_method == 'elmo':
            return self.get_elmo(sentence_lists)

        elif self.emb_method == 'glove':
            return self.get_glove(sentence_lists)

        elif self.emb_method == 'elmo_glove':
            elmo_embeddings = self.get_elmo(sentence_lists)
            glove_embeddings = self.get_glove(sentence_lists)
            return torch.cat([elmo_embeddings, glove_embeddings], -1)
