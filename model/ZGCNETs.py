import torch
import torch.nn as nn
from ZGCNCELL import NLSGCRNCNNCell


class NNLSDCRNNCNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1, window_len = 12):
        super(NNLSDCRNNCNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.window_len = window_len
        self.nlsdcrnncnn_cells = nn.ModuleList()
        for _ in range(0, num_layers-1):
            self.nlsdcrnncnn_cells.append(NLSGCRNCNNCell(node_num, dim_in, dim_out, window_len, cheb_k, embed_dim))
        for _ in range(num_layers-1, num_layers):
            self.nlsdcrnncnn_cells.append(NLSGCRNCNNCell(node_num, dim_out, dim_out, window_len, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, zigzag_PI):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.nlsdcrnncnn_cells[i](current_inputs[:, t, :, :], state, current_inputs,
                                                  node_embeddings, zigzag_PI[:, :, :].view(-1, 1, 100, 100)) # ZPI input shape
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.nlsdcrnncnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0) # (num_layers, B, N, hidden_dim)

# ************************************************* #
class NNLSGCRNCNN(nn.Module):
    def __init__(self, args):
        super(NNLSGCRNCNN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = NNLSDCRNNCNN(args.num_nodes, args.input_dim, args.rnn_units, args.link_len,
                                args.embed_dim, args.num_layers, args.window_len)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets, zigzag_PI, teacher_forcing_ratio=0.5):

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings, zigzag_PI) # (B, T, N, hidden)
        output = output[:, -1:, :, :] # (B, 1, N, hidden)

        #CNN based predictor
        output = self.end_conv((output)) # (B, T*C, N, 1)
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2) # (B, T, N, C)

        return output
