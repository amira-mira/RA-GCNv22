import torch
from torch import nn
from torch.nn import functional as F


class Mask(nn.Module):
    def __init__(self, model_stream, module):
        super(Mask, self).__init__()
        self.model_stream = model_stream
        self.module = module

    def forward(self, weight, feature):
        mask_matrices = []
        for i in range(self.model_stream):
            temp = self.CAM(weight[i], feature[i])
            mask_matrices.append(temp)
        for i in range(1, self.model_stream):
            for j in range(i):
                if j == 0:
                    mask = mask_matrices[j]
                else:
                    mask *= mask_matrices[j]
            mask = torch.cat([mask.unsqueeze(1)] * 4, dim=1)
            self.module.mask_stream[i].data = mask.view(-1).detach()
    
    def CAM(self, weight, feature):
        N, C = weight.shape
        weight = weight.view(N, C, 1, 1, 1).expand_as(feature)
        result = (weight * feature).sum(dim=1)
        result = result.mean(dim=0)

        T, V, M = result.shape
        result = result.view(-1)
        result = 1 - F.softmax(result, dim=0)
        result[result > 0.3] = 1
        result[result <= 0.3] = 0
        result = result.view(T, V, M)
        return result

