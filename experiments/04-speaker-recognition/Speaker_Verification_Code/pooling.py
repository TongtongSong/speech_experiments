import torch
import torch.nn as nn

class StatisticsPooling(torch.nn.Module):
    """
    Mean and Standard deviation pooling
    """
    def __init__(self):
        """

        """
        super(StatisticsPooling, self).__init__()
        pass

    def forward(self, x):
        """

        :param x:
        :return:
        """
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        return torch.cat([mean, std], dim=1)


class GlobalAveragePooling(torch.nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    # ==========> code <===========
    def forward(self, x):
        '''
        input: (64, 1500, 86)
        output: (64, 1500)
        '''
        pass
    # ==========> code <===========

