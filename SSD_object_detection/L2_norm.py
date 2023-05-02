from lib import *

class L2Norm(nn.Module):
    def __init__(self, input_channels =512, scale = 20):
        super(L2Norm, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameter()
        self.eps = 1e-10

    def reset_parameter(self):
        init.constant_(self.weight, self.scale)

    def forward(self, x):
        #L2norm
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        
        weights = self.weights.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x*weights
