import torch
from torch import nn
from torchvision import transforms

class Encoder(torch.nn.Module):
    '''
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1          [-1, 5, 128, 128]           3,380
                  ReLU-2          [-1, 5, 128, 128]               0
                Conv2d-3            [-1, 8, 64, 64]           4,848
                  ReLU-4            [-1, 8, 64, 64]               0
                Conv2d-5           [-1, 12, 32, 32]           4,716
                  ReLU-6           [-1, 12, 32, 32]               0
                Conv2d-7           [-1, 16, 16, 16]           1,744
    ================================================================
    Total params: 14,688
    Trainable params: 14,688
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.75
    Forward/backward pass size (MB): 1.97
    Params size (MB): 0.06
    Estimated Total Size (MB): 2.77
    ----------------------------------------------------------------
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,5,kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.Conv2d(5, 8, kernel_size=11, stride=2, padding=5),
            nn.ReLU(),
            nn.Conv2d(8, 12, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(12, 16, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, img):
        x = transforms.ToTensor()(img).unsqueeze(0)
        x = self.encoder(x)
        return x.cpu().detach().half().numpy()

class Decoder(torch.nn.Module):
    '''
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
       ConvTranspose2d-1           [-1, 12, 32, 32]           1,740
                  ReLU-2           [-1, 12, 32, 32]               0
       ConvTranspose2d-3            [-1, 8, 64, 64]           4,712
                  ReLU-4            [-1, 8, 64, 64]               0
       ConvTranspose2d-5          [-1, 5, 128, 128]           4,845
                  ReLU-6          [-1, 5, 128, 128]               0
       ConvTranspose2d-7          [-1, 3, 256, 256]           3,378
                  Tanh-8          [-1, 3, 256, 256]               0
    ================================================================
    Total params: 14,675
    Trainable params: 14,675
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.02
    Forward/backward pass size (MB): 4.94
    Params size (MB): 0.06
    Estimated Total Size (MB): 5.01
    ----------------------------------------------------------------
    '''
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 12, kernel_size=3, stride=2, padding=1, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(12,8,kernel_size=7, stride=2, padding=3, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,5,kernel_size=11, stride=2, padding=5, output_padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(5,3,kernel_size=15, stride=2, padding=7, output_padding = 1),
            nn.Tanh()
        )

    def forward(self, img):
        x = torch.tensor(img).float()
        x = self.decoder(x)
        return transforms.ToPILImage()(x.select(0,0))