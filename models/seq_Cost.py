from torch.nn.modules.module import Module
import torch
import numpy as np
from torch.autograd import Variable


class GetCostVolume(Module):
    def __init__(self, maxdisp):
        super(GetCostVolume, self).__init__()
        self.maxdisp = maxdisp + 1

    def forward(self, x, y):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
            cost = x.new().resize_(num, channels * 2, self.maxdisp, height, width).zero_()
#            cost = Variable(torch.FloatTensor(x.size()[0], x.size()[1]*2, self.maxdisp,  x.size()[2],  x.size()[3]).zero_(), volatile= not self.training).cuda()
            for i in range(self.maxdisp):
                if i > 0 :
                    cost[:, :x.size()[1], i, :,i:]   = x[:,:,:,i:]
                    cost[:, x.size()[1]:, i, :,i:]   = y[:,:,:,:-i]
                else:
                    cost[:, :x.size()[1], i, :,:]   = x
                    cost[:, x.size()[1]:, i, :,:]   = y

            cost = cost.contiguous()
        return cost
 
class DisparityRegression(Module):
    def __init__(self, maxdisp):
       super(DisparityRegression, self).__init__()
       self.maxdisp = maxdisp + 1
#        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1,self.maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1, self.maxdisp, 1, 1])).cuda(), requires_grad=False)
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out

