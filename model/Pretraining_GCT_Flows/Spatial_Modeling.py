import numpy as np

def create_matrix(n, k):
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(k):
            if i-j >= 0:
                mat[i][i-j] = 1
    return mat


def pearson_corr2(tensor): # all: (64,1,n,dim)
  # Input tensor shape: (batch_size, num_nodes, time_steps)
  batch_size, num_nodes, _ = tensor.shape
  tensor = tensor - tensor.mean(dim=2, keepdim=True)
  std = tensor.std(dim=2, keepdim=True)
  tensor = tensor / (std + 1e-8)
  correlation_matrix = torch.matmul(tensor, tensor.transpose(1, 2))
  correlation_matrix = correlation_matrix / (tensor.shape[2] - 1)
  return correlation_matrix


def topK(attn, top_num ):

  # Get the top K values and their indices for each row
  top_k_values, top_k_indices = attn.topk(top_num, dim=3)

  # Create a mask with the same shape as the input tensor, filled with zeros
  mask = torch.zeros_like(attn)

  # Set the top K values in the mask to 1
  mask.scatter_(3, top_k_indices, 1)

  # Multiply the input tensor with the mask to get the result
  attn = attn * mask

  return  attn



# this efficient implementation comes from https://github.com/xptree/DeepInf/
class S_BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_heads, num_nodes, dropout, bias=True):
        super(S_BatchMultiHeadGraphAttention, self).__init__()

        print('S_BatchMultiHeadGraphAttention', n_heads, num_nodes, dropout)
        self.n_head = n_heads
        self.f_in = num_nodes
        #self.w = nn.Parameter(torch.Tensor(self.n_head, num_nodes, num_nodes))
        self.a_src = nn.Parameter(torch.Tensor(self.n_head*2, num_nodes, 1))
        self.a_dst = nn.Parameter(torch.Tensor(self.n_head*2, num_nodes, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_nodes))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        #nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

        '''
        self.W_si = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_si.data, gain=1.414)
        self.W_ei = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_ei.data, gain=1.414)
        '''
        self.W_si = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_si.data, gain=1.414)
        self.W_ei = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_ei.data, gain=1.414)

    def forward(self, h, a):


        bs, ch, n, dim = h.size()
        #h_prime = torch.matmul(h, self.w)
        h_prime = h
        attn_src = torch.matmul(torch.tanh(h), self.a_src)
        attn_dst = torch.matmul(torch.tanh(h), self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )

        #attn = abs(self.W_ei)*attn+abs(self.W_si)*a

        #zero_vec = -9e15*torch.ones_like(attn)
        #attn_2 = torch.where(a > 0, attn, zero_vec)
        attn_2 = self.leaky_relu(attn)
        attn_2 = self.softmax(attn_2)
        attn_2 = self.dropout(attn_2)

        attn_2 = abs(self.W_ei)*attn_2+abs(self.W_si)*a

        output_2 = torch.matmul(attn_2, h_prime)

        return output_2, attn

# MutiChannel_GAT(kern, dilation_factor, n_heads, num_nodes, mlp, mlp2, dropout)
class S_MutiChannel_GAT(nn.Module):
    def __init__(self, kern, dilation_factor, n_heads, num_nodes, mlp, mlp2, dropout):
        super(S_MutiChannel_GAT, self).__init__()

        print('S_MutiChannel_GAT', n_heads, num_nodes, dropout)

        self.gat_layer = S_BatchMultiHeadGraphAttention(
            n_heads, num_nodes, dropout
        )
        self.gat_layer2 = S_BatchMultiHeadGraphAttention(
            n_heads, num_nodes, dropout
        )

        self.mlp1 =  nn.Conv2d(in_channels=32,
                                    out_channels=16,
                                    kernel_size=(1, 1))

        self.mlp2 =  nn.Conv2d(in_channels=48,
                                    out_channels=32,
                                    kernel_size=(1, 1))

    def forward(self,x_input, a_f):
        x_input_cpy = x_input

        x = x_input
        x = self.mlp1(x)

        x1, attn = self.gat_layer(x,a_f)
        a_f = torch.matmul(a_f,a_f)
        x2, attn = self.gat_layer2(x1,a_f)

        x_out = self.mlp2(torch.cat([x,x1,x2],dim=1))

        return x_out