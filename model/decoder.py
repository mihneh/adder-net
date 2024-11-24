import torch
import torch.nn as nn
from general_blocks import *
from typing import List
from collections import OrderedDict

class SpectrumDecoder(nn.Sequential):
    def __init__(self,  in_channels: int,
                        out_channels: List[int] | int = [128, 96, 64],
                        kernel_sizes: List[int] | int = [3, 5, 7],
                        use_mobile: List[bool] | bool = True, 
                        act_func: List[nn.Module]  | nn.Module  = nn.SiLU(),
                        do_bn: List[bool] | bool = True,
                        do_sc: List[bool] | bool = True,
                        dp: float | int | List[float] | int = 0.4,
                        num_blocks: int = 3):
        
        super().__init__()

        out_channels = list_fn(num_blocks, out_channels)
        kernel_sizes = list_fn(num_blocks, kernel_sizes)
        use_mobile = list_fn(num_blocks, use_mobile)
        act_func = list_fn(num_blocks, act_func)
        do_bn = list_fn(num_blocks, do_bn)
        do_sc = list_fn(num_blocks, do_sc)
        dp = list_fn(num_blocks, dp)

        blocks = [AdaptiveResBlock(in_channels = in_channels if i == 0 else out_channels[i - 1],
                                   out_channels = out_channels[i],
                                   kernel_size = kernel_sizes[i],
                                   act_func = act_func[i],
                                   use_mobile = use_mobile[i],
                                   upscale = True,
                                   inverted = True,
                                   dp = dp[i],
                                   do_bn = do_bn[i],
                                   do_sc = do_sc[i]) for i in range(num_blocks)]
        
        self.decoder_features = nn.Sequential(OrderedDict([(f"layer_{i}", layer) for i, layer
                                              in enumerate(blocks)]))