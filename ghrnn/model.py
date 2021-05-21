import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class GHRNN(nn.Module):
    def __init__(self,t_size,v_size,e_size,embedding_size,hidden_size,output_timestamp1,
                 output_timestamp2,output_vertex1,output_edge,output_vertex2,num_layers=1,
                 batch_first=True,device=torch.device('cpu')):
        super(GHRNN, self).__init__()
        self.t_size = t_size
        self.v_size = v_size
        self.e_size = e_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.device = device
        # 构建网络结构：GRU, input(t,v,e的维度不同，所以至少需要三个不同的mlp来转化一下维度), 预测网络MLP
        self.GRU = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=self.batch_first)
        self.input_t1 = nn.Linear(self.t_size, self.embedding_size)
        self.input_t2 = nn.Linear(self.t_size, self.embedding_size)
        self.input_v1 = nn.Linear(self.v_size, self.embedding_size)
        self.input_e = nn.Linear(self.e_size, self.embedding_size)
        #构建预测网络MLP
        self.output_timestamp1 = output_timestamp1
        self.output_timestamp2 = output_timestamp2
        self.output_vertex1 = output_vertex1
        self.output_edge = output_edge
        self.output_vertex2 = output_vertex2
        # GRU权重参数初始化
        for name, param in self.GRU.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(
                    param, gain=nn.init.calculate_gain('sigmoid'))
        # input网络参数初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, hidden, *input):
        if len(input) != 0:
            (x_t1, x_t2, x_v1, x_e) = input
        batch_size = hidden.size()[0]
        all_t1 = self.output_timestamp1(hidden)
        for i in range(hidden.size()[1]):
            h0 = hidden[:, i:i+1, :].view(1,batch_size,self.hidden_size).contiguous()
            for j in range(4):
                if j == 0:
                    if len(input) != 0:
                        input0 = self.input_t1(x_t1[:,i:i+1,:])
                    else:
                        input0 = self.input_t1(all_t1[:, i:i+1, :])
                    output1, h1 = self.GRU(input0, h0)
                    t2 = self.output_timestamp2(output1)
                    if i == 0:
                        all_t2 = t2
                    else:
                        all_t2 = torch.cat((all_t2,t2),dim=1)
                elif j == 1:
                    if len(input) != 0:
                        input1 = self.input_t2(x_t2[:,i:i+1,:])
                    else:
                        input1 = self.input_t2(all_t2[:,i:i+1,:])
                    output2, h2 = self.GRU(input1,h1)
                    v1 = self.output_vertex1(output2)
                    if i == 0 :
                        all_v1 = v1
                    else:
                        all_v1 = torch.cat((all_v1,v1),dim=1)
                elif j == 2:
                    if len(input) != 0:
                        input2 = self.input_v1(x_v1[:,i:i+1,:])
                    else:
                        input2 = self.input_v1(all_v1[:,i:i+1,:])
                    output3, h3 = self.GRU(input2, h2)
                    # e = self.output_vertex1(output3)
                    e = self.output_edge(output3)
                    if i == 0 :
                        all_e = e
                    else:
                        all_e = torch.cat((all_e,e),dim=1)
                else:
                    if len(input) != 0:
                        input3 = self.input_e(x_e[:,i:i+1,:])
                    else:
                        input3 = self.input_e(all_e[:,i:i+1,:])
                    output4, h4 = self.GRU(input3, h3)
                    v2 = self.output_vertex2(output4)
                    if i == 0 :
                        all_v2 = v2
                    else:
                        all_v2 = torch.cat((all_v2,v2),dim=1)

        return all_t1,all_t2,all_v1,all_e,all_v2


class MLP_Softmax(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            MLP_Plain(input_size, embedding_size, output_size, dropout),
            nn.Softmax(dim=2)
        )

    def forward(self, input):
        return self.mlp(input)


class MLP_Log_Softmax(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Log_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            MLP_Plain(input_size, embedding_size, output_size, dropout),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, input):
        return self.mlp(input)


class MLP_Plain(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Plain, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.Linear(embedding_size, embedding_size),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self.mlp(input)


class RNN(nn.Module):
    """
    Custom GRU layer
    :param input_size: Size of input vector
    :param embedding_size: Embedding layer size (finally this size is input to RNN)
    :param hidden_size: Size of hidden state of vector
    :param num_layers: No. of RNN layers
    :param rnn_type: Currently only GRU and LSTM supported
    :param dropout: Dropout probability for dropout layers between rnn layers
    :param output_size: If provided, a MLP softmax is run on hidden state with output of size 'output_size'
    :param output_embedding_size: If provided, the MLP softmax middle layer is of this size, else 
        middle layer size is same as 'embedding size'
    :param device: torch device to instanstiate the hidden state on right device
    """

    def __init__(
        self, input_size, embedding_size, hidden_size, num_layers, rnn_type='GRU',
        dropout=0, output_size=None, output_embedding_size=None,
        device=torch.device('cpu')
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.output_size = output_size
        self.device = device

        self.input = nn.Linear(input_size, embedding_size)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout
            )
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout
            )

        # self.relu = nn.ReLU()

        self.hidden = None  # Need initialization before forward run

        if self.output_size is not None:
            if output_embedding_size is None:
                self.output = MLP_Softmax(
                    hidden_size, embedding_size, self.output_size)
            else:
                self.output = MLP_Softmax(
                    hidden_size, output_embedding_size, self.output_size)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(
                    param, gain=nn.init.calculate_gain('sigmoid'))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        if self.rnn_type == 'GRU':
            # h0
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        elif self.rnn_type == 'LSTM':
            # (h0, c0)
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))

    def forward(self, input, input_len=None):
        input = self.input(input)
        # input = self.relu(input)

        if input_len is not None:
            input = pack_padded_sequence(
                input, input_len, batch_first=True, enforce_sorted=False)

        output, self.hidden = self.rnn(input, self.hidden)

        if input_len is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)

        if self.output_size is not None:
            output = self.output(output)

        return output


def create_model(args, feature_map):
    max_nodes = feature_map['max_nodes']
    len_node_vec, len_edge_vec = len(
        feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1

    if args.note == 'DFScodeRNN':
        feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_edge_vec
    else:
        feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec

    if args.loss_type == 'BCE':
        MLP_layer = MLP_Softmax
    elif args.loss_type == 'NLL':
        MLP_layer = MLP_Log_Softmax

    dfs_code_rnn = RNN(
        input_size=feature_len, embedding_size=args.embedding_size_dfscode_rnn,
        hidden_size=args.hidden_size_dfscode_rnn, num_layers=args.num_layers,
        rnn_type=args.rnn_type, dropout=args.dfscode_rnn_dropout,
        device=args.device).to(device=args.device)

    output_timestamp1 = MLP_layer(
        input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_timestamp_output,
        output_size=max_nodes + 1, dropout=args.dfscode_rnn_dropout).to(device=args.device)

    output_timestamp2 = MLP_layer(
        input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_timestamp_output,
        output_size=max_nodes + 1, dropout=args.dfscode_rnn_dropout).to(device=args.device)

    output_vertex1 = MLP_layer(
        input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_vertex_output,
        output_size=len_node_vec, dropout=args.dfscode_rnn_dropout).to(device=args.device)

    output_vertex2 = MLP_layer(
        input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_vertex_output,
        output_size=len_node_vec, dropout=args.dfscode_rnn_dropout).to(device=args.device)

    if args.note == 'DFScodeRNN' and args.create_ghrnn == False:
        model = {
            'dfs_code_rnn': dfs_code_rnn,
            'output_timestamp1': output_timestamp1,
            'output_timestamp2': output_timestamp2,
            'output_vertex1': output_vertex1,
            'output_vertex2': output_vertex2
        }
        output_edge = MLP_layer(
            input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_edge_output,
            output_size=len_edge_vec, dropout=args.dfscode_rnn_dropout).to(device=args.device)
        model['output_edge'] = output_edge

    elif args.note == 'DFScodeRNN' and args.create_ghrnn == True:
        output_edge = MLP_layer(
            input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_edge_output,
            output_size=len_edge_vec, dropout=args.dfscode_rnn_dropout).to(device=args.device)
        dfs_code_ghrnn = GHRNN(max_nodes+1,len_node_vec,len_edge_vec,embedding_size=args.embedding_size_dfscode_ghrnn,
                             hidden_size=args.hidden_size_dfscode_rnn,output_timestamp1=output_timestamp1,
                             output_timestamp2=output_timestamp2,output_vertex1=output_vertex1,output_edge=output_edge,
                             output_vertex2=output_vertex2,device=args.device).to(device=args.device)
        model = {
            'dfs_code_rnn': dfs_code_rnn,
            'dfs_code_ghrnn':dfs_code_ghrnn}

    return model
