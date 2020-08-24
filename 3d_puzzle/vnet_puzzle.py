import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hungarian import Hungarian

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        # self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        #x16 = torch.cat((x, x, x, x, x, x, x, x,
        #                 x, x, x, x, x, x, x, x), 0)
        out = self.relu1(torch.add(out, x))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        #out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        #out = out.view(out.numel() // 2, 2)
        out = self.softmax(out, dim=1)
        # treat channel 0 as the predicted output
        return out[:,0:1,:,:,:]


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, feature_len, elu=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)

        self.fc_output = nn.Linear(256 * (2 ** 3), feature_len) 

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)

        out256 = out256.view(out256.shape[0], -1)
        out = self.fc_output(out256)
        return out

class PuzzleNet(nn.Module):
    def __init__(self, feature_len, puzzle_num, iter_num, flag_pair):
        super(PuzzleNet, self).__init__()
        self.puzzle_num = puzzle_num
        self.iter_num = iter_num
        self.feature_len = feature_len
        self.flag_pair = flag_pair

        self.vnet = VNet(self.feature_len)

        self.unary_fc1 = nn.Linear(self.feature_len * puzzle_num, 4096)
        self.unary_fc2 = nn.Linear(4096, puzzle_num ** 2)
        self.u_log_softmax = nn.LogSoftmax(dim=2)

        if self.flag_pair:
            self.binary_fc1 = nn.Linear(self.feature_len * 2, 512)
            self.binary_fc2 = nn.Linear(512, 7)
            self.b_log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, y):
        tower_size = x.shape[0]

        # get feature_list : tower X puzzle_num X features
        feature_list = []
        for i in range(self.puzzle_num):
            feature_list.append(self.vnet(x[:,i,:,:,:,:]))

        # unary loss
        unary_list = []
        perm_list = []
        cur_perm = y

        ## first iter
        feature_stack = torch.stack(feature_list, dim=1)
        features = torch.reshape(feature_stack, \
                                (tower_size, \
                                 self.puzzle_num * self.feature_len))

        u_out1 = self.unary_fc1(features)
        u_out2 = self.unary_fc2(u_out1)
        u_out2 = u_out2.view(tower_size, self.puzzle_num, self.puzzle_num)
        u_out = self.u_log_softmax(u_out2)

        unary_list.append(u_out)
        perm_list.append(cur_perm)

        ## other iters
        for iter_id in range(self.iter_num - 1):
            ### hungarian algorithm for new permutation
            out_detach = u_out.detach().cpu().numpy()
            feature_stack_detach = feature_stack.detach().cpu().numpy()
            hungarian = Hungarian()

            new_feature_stack = np.zeros_like(feature_stack_detach)
            results_stack = np.zeros((tower_size, self.puzzle_num))

            for i in range(tower_size):
                hungarian.calculate(-1 * out_detach[i,:,:])
                results = hungarian.get_results()

                for j in range(self.puzzle_num):
                    new_feature_stack[i, results[j][1], :] = \
                        feature_stack_detach[i, results[j][0], :]
                    results_stack[i, results[j][1]] = results[j][0]

            results_stack = torch.from_numpy(results_stack).long().cuda()
            cur_perm = torch.gather(cur_perm, 1, results_stack)
            perm_list.append(cur_perm)

            ### new iteration
            feature_stack = torch.from_numpy(new_feature_stack).float().cuda()
            features = torch.reshape(feature_stack, \
                                    (tower_size, \
                                     self.puzzle_num * self.feature_len))

            u_out1 = self.unary_fc1(features)
            u_out2 = self.unary_fc2(u_out1)
            u_out2 = u_out2.view(tower_size, self.puzzle_num, self.puzzle_num)
            u_out = self.u_log_softmax(u_out2)
            unary_list.append(u_out)

        if not self.flag_pair: 
            return unary_list, perm_list
        else:
            # binary loss
            binary_list = []
            for i in range(self.puzzle_num):
                for j in xrange(i + 1, self.puzzle_num):
                    feature_pair = torch.cat([feature_list[i], \
                                              feature_list[j]], dim=1)
                    b_out1 = self.binary_fc1(feature_pair)
                    b_out2 = self.binary_fc2(b_out1)
                    b_out = self.b_log_softmax(b_out2)
                    binary_list.append(b_out)

            binary_stack = torch.stack(binary_list, dim=1)
            binary_stack = binary_stack.view(-1, 7)

            return unary_list, perm_list, binary_stack


def puzzlenet(feature_len, puzzle_num, iter_num, flag_pair):
    return PuzzleNet(feature_len, puzzle_num, iter_num, flag_pair)
