import torch
import torch.nn as nn
from general_blocks import *
from encoder import *
from decoder import *


class DenoisingModel(nn.Module):
    def __init__(self,  encoder_parameters: dict = dict(in_channels=3,
                                                        out_channels = [64, 96, 128],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = True, 
                                                        act_func = nn.SiLU(),
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3),
                                                        
                        decoder_parameters: dict = dict(in_channels=128,
                                                        out_channels = [96, 64, 3],
                                                        kernel_sizes = [3, 5, 7],
                                                        use_mobile = True, 
                                                        act_func = nn.SiLU(),
                                                        do_bn = True,
                                                        do_sc = True,
                                                        dp = 0.4,
                                                        num_blocks = 3),
                        hidden_gru: int = 512,
                        num_gru_cells: int = 2,
                        dp_gru = 0.3):
        
        super().__init__()


        assert encoder_parameters['out_channels'][-1] == decoder_parameters['in_channels']

        self.encoder = SpectrumEncoder(**encoder_parameters)
        self.decoder = SpectrumDecoder(**decoder_parameters)
        
        self.gru = nn.GRU(input_size=encoder_parameters['out_channels'][-1], 
                          hidden_size=hidden_gru,
                          batch_first=True,
                          dropout=dp_gru,
                          num_layers=num_gru_cells,
                          bias=False)
        
        self.linear = nn.Linear(hidden_gru, encoder_parameters['out_channels'][-1])


    def forward(self, x):

        encoded = self.encoder(x)
        batch_size, channels, time, frequency = encoded.shape
        # we can use adaptive avg pool like [bs, channels, time, 1]
        # were time - hyperparameter of encoder
        # 
        # [batch size, channels, time, frequency]
        # [batch size, channels, time, frequency] <-> (?) [batch size, time, hidden dim] 

        input_size = frequency * channels
        
        x = encoded.permute(0, 2, 3, 1)

        x = x.reshape(batch_size, time, input_size)
        
        gru_out, hidden = self.gru(x)     

        out = self.linear(gru_out)

        decoder_input = out.view(batch_size, channels, time, frequency)

        decoded = self.decoder(decoder_input)

        return decoded