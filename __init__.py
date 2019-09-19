import numpy as np
import PIL
import torch
from torch import nn

from .models import Encoder, Decoder
encode = Encoder()
decode = Decoder()

enc_state_dict = torch.load('./shallow_enc_16_2000.pth', map_location='cpu')
encode.load_state_dict(enc_state_dict)
dec_state_dict = torch.load('./shallow_dec_16_2000.pth', map_location='cpu')
decode.load_state_dict(dec_state_dict)

encode.eval()
decode.eval()
