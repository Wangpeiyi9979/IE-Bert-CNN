import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, enc_method='cnn', filters_num=None, filters=None, f_dim=None, input_size=None, hidden_size=None, bidirectional=True):
        super(Encoder, self).__init__()

        if enc_method == 'cnn':
            if filters_num is None or filters is None or f_dim is None:
                raise RuntimeError("filters_num/filters/f_dim are not provided")
        else:
            if input_size is None or hidden_size is None:
                raise RuntimeError("input_size/hidden_size are not provided")

        self.enc_method = enc_method
        if enc_method == 'cnn':
            self.convs = nn.ModuleList([nn.Conv2d(1, filters_num, (k, f_dim), padding=(int(k / 2), 0)) for k in filters])
            self.init_model_weight()

        elif enc_method == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)

        elif enc_method == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)

        elif enc_method == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)

    def init_model_weight(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            #nn.init.constant_(conv.bias, 0.0)

    def sequence_mask(self, sequence_length, max_len=None):
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
        if sqe_len is None:
            return inputs
        mask = self.sequence_mask(sqe_len, inputs.size(1))  # (B, L)
        mask = mask.unsqueeze(-1)      # (B, L, 1)
        outputs = inputs * mask.float()
        return outputs

    def forward(self, inputs, lengths=None):

        if self.enc_method == 'cnn':
            if not lengths is None:
                inputs = self.Mask(inputs, lengths)
            x = inputs.unsqueeze(1)
            x = [conv(x).squeeze(3) for conv in self.convs]
            return x

        else:
            if not lengths is None:
                sorted_lengths, sorted_indexs = torch.sort(lengths, descending=True)
                tmp1, desorted_indexs = torch.sort(sorted_indexs, descending=False)
                x_rnn = inputs[sorted_indexs]
                packed_x_rnn = nn.utils.rnn.pack_padded_sequence(x_rnn, sorted_lengths.cpu().numpy(), batch_first=True)
                packed_rnn_output, tmp2 = self.rnn(packed_x_rnn)
                sort_rnn_output, tmp3 = nn.utils.rnn.pad_packed_sequence(packed_rnn_output, batch_first=True)
                rnn_output = sort_rnn_output[desorted_indexs]  # (B, N, 2/1hidden_size)
                return rnn_output
            else:
                rnn_output, tmp = self.rnn(inputs)
                return rnn_output                              # (B, L, 2/1hidden_size)
