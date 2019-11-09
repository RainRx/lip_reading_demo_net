import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.modules import _DenseBlock, _Transition
from torch.autograd import Variable
import opt
import math


class LipSeqLoss(nn.Module):
 
    def __init__(self):
        super(LipSeqLoss, self).__init__()
        self.criterion = nn.NLLLoss(reduction='none')

    def forward(self, input, length, target):
        loss = []
        transposed = input.transpose(0, 1).contiguous()
        for i in range(transposed.size(0)):
            loss.append(self.criterion(transposed[i, ], target.squeeze(1)).unsqueeze(1))
        loss = torch.cat(loss, 1)

        # GPU version
        mask = torch.zeros(loss.size(0), loss.size(1)).float().cuda()
        # Cpu version
#         mask = torch.zeros(loss.size(0), loss.size(1)).float()   

        for i in range(length.size(0)):
            L = min(mask.size(1), length[i])
            mask[i, L-1] = 1.0
        loss = (loss * mask).sum() / mask.sum()
        return loss


class LipNet(torch.nn.Module):
    def __init__(self, growth_rate=opt.growth_rate, num_init_features=opt.init_features_num,
                 drop_rate=opt.drop_rate, type_class=opt.num_classes, bn_size=opt.BN_size):
        super(LipNet, self).__init__()
        block_config = (4, 8, 12, 8)
        self.drop_rate = drop_rate
        self.type_class = type_class 

        # Cnn
        self.features = nn.Sequential(OrderedDict([
            ('conv0',
             nn.Conv3d(3, num_init_features, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))),
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):

            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        # Rnn
        self.gru1 = nn.GRU(64*28*28, 256, bidirectional=True, batch_first=True) 
        self.gru2 = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        # Fc
        self.fc = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(512, self.type_class)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        # Cnn
        cnn = self.features(x)
        cnn = cnn.permute(0, 2, 1, 3, 4).contiguous()
        batch, seq, channel, high, width = cnn.size()
        cnn = cnn.view(batch, seq, -1)
        # Rnn
        rnn, _ = self.gru1(cnn)
        rnn, _ = self.gru2(rnn)
        # Fc
        fc = self.fc(rnn).log_softmax(-1)
        return fc
