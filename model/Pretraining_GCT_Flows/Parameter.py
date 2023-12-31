

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda',help='')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')

parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')


parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')

parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')

parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')

parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--clip',type=int,default=5,help='clip')


parser.add_argument('--model_type',type=str,default='GMAT',help='model type')
parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
parser.add_argument('--end_channels',type=int,default=128,help='end channels')


parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--save',type=str,default='./save/',help='save path')

parser.add_argument('--log_print', type=str_to_bool, default=False ,help='whether to load static feature')

parser.add_argument('--learning_rate',type=float,default=0.0005,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')

parser.add_argument('--step_size1',type=int,default=300,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')

target = 'v7_GCT_flow_34nodes_84edges'
parser.add_argument('--data',type=str,default='../Data/'+target ,help='data path')
parser.add_argument('--adj_data',type=str,default='../Data/'+target+'/adj_mat_input.pkl',help='adj data path')
parser.add_argument('--num_nodes',type=int,default=34,help='number of nodes/variables')

parser.add_argument('--expid',type=int,default=202308271352,help='experiment id')
parser.add_argument('--runs',type=int,default=30,help='number of runs')
parser.add_argument('--epochs',type=int,default=200,help='')

parser.add_argument('--seq_in_len',type=int,default=8,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=4,help='output sequence length')

parser.add_argument('--layers',type=int,default=6,help='number of layers')

#args = parser.parse_args()
args=parser.parse_args(args=[])
torch.set_num_threads(3)

#args = parser.parse_args()
args=parser.parse_args(args=[])
print('# args', args)

device = torch.device(args.device)

writer = SummaryWriter()