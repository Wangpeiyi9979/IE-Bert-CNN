import torch
import torch.nn as nn
import torch.nn.functional as F

class Attenter(nn.Module):

    def __init__(self, att_method='Hdot', f_dim=None, q_dim=None, q_num=None):
        super(Attenter, self).__init__()
        """
        input:
            f_dim: dimension of input vectors
            q_dim: dimension of query vectors
            att_method:
                'Hdot':  QHW: a very simple att att_method

                'Cat'*:  tanh(H[W;q]+b): A complex att att_method, we just accept 1-dimension query vector

                'Tdot1': QTanh(HW+b):
                        att_method from paper:
                        "Coevolutionary Recommendation Model: Mutual Learning between Ratings and Reviews"

                'Tdot2': Tanh(QHW+b):
                        att_method from parer:
                        "HAMI: Neural Gender Prediction for Chinese Microblogging with Hierarchical Attention and Multi-channel Input
        """
        self.att_method = att_method

        if self.att_method == 'Hdot':
            self.H = nn.Parameter(torch.randn(f_dim, q_dim))

        elif self.att_method == 'Tdot1':
            self.L = nn.Linear(f_dim, q_dim)

        elif self.att_method == 'Tdot2':
            self.H = nn.Parameter(torch.randn(f_dim, q_dim))
            self.att_bias = nn.Parameter(torch.randn(1, 1, q_num))

        elif self.att_method == 'Cat':
            self.L = nn.Linear(f_dim+q_dim, 1)

        self.init_weight()

    def init_weight(self):
        if self.att_method == 'Hdot':
            nn.init.xavier_normal_(self.H)

        elif self.att_method == 'Tdot1' or self.att_method == 'Cat':
            nn.init.xavier_normal_(self.L.weight)
            nn.init.uniform_(self.L.bias)

        elif self.att_method == 'Tdot2':
            nn.init.xavier_normal_(self.H)
            nn.init.uniform_(self.att_bias)

    def sequence_mask(self, sequence_length, max_len=None):
        """
        Accept length vector, and return mask matrix.
        refer:https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/masked_cross_entropy.py#L5
        """
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.LongTensor(range(0, max_len))
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand

    def Mask(self, inputs, sqe_len=None):
        """
        A simper vision mask att_method just for input size (B, L, K)
        refer: https://github.com/bojone/attention/blob/master/attention_tf.py
        """

        """
        inputs: (B, L, K), the possibilities we need to mask
        sqe_len: (B)

        """
        if sqe_len is None:
            return inputs
        mask = self.sequence_mask(sqe_len, input.size(1))  # (B, L)
        mask = mask.unsqueeze(-1)      # (B, L, 1)
        outputs = inputs - (1 - mask).float() * 1e12
        return outputs

    def forward(self, W, Q, sqe_len=None):
        """
        input:
            W: sematic vectors    (B:batchSize/None, L: numberOfVector, f_dim: featureDim)
            Q: query vectors      (K:numberOfQueryVector, q_dim: queryVectorFeatureDim)
        sqe_len:
            the lengths of sentence in word level

        tip: K = 1 when attention att_method is Cat
        ouput:
            result: (B, K, f_dim)
        """
        if len(W.size()) == 2:
            W = W.unsqueeze(0)

        if self.att_method == 'Hdot':
            V = torch.matmul(W, self.H)                               # (B, L, q_dim)
            A = torch.matmul(V, Q.t())                                # (B, L, K)

        elif self.att_method == 'Tdot1':
            V = self.L(W).tanh()                                      # (B, L, q_dim)
            A = torch.matmul(V, Q.t())                                # (B, L, K)

        elif self.att_method == 'Tdot2':
            V = torch.matmul(W, self.H)                               # (B, L, q_dim)
            A = torch.matmul(V, Q.t()) + self.att_bias                # (B, L, K)
            A = A.tanh()                                              # (B, L, K)

        elif self.att_method == 'Cat':
            zeros = torch.zeros(W.size(0), W.size(1), Q.size(1))
            if Q.is_cuda:
                zeros = zeros.cuda()
            extend_q = zeros + Q.squeeze(0)                           # (B, L, q_dim)
            V = torch.cat([W, extend_q], -1)                          # (B, L, f_dim + q_dim)
            A = self.L(V).tanh()                                      # (B, L, K)  tip:K = 1

        if not sqe_len is None:
            A = self.Mask(A, sqe_len)
        A = F.softmax(A, 1).permute(0,2,1)                            # (B, K, L)
        result = torch.matmul(A, W)                                   # (B, K, f_dim)

        return result
