import torch
import torch.nn as nn
from typing import Optional


def list_fn(num_blocks, params):
    if isinstance(params, list):
        if len(params) < num_blocks:
            params += [params[-1]] * (num_blocks - len(params))
        else:
            params = params[:num_blocks]
        
        return params
    else:
        return list_fn(num_blocks, [params])


class BaseBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                        inverted: bool = False,
                        act_func: nn.Module = nn.SiLU(),
                        do_bn: bool = True,
                        dp: float | int = 0.4):
        super().__init__()

        assert kernel_size % 2 != 0, 'Kernel must be odd'
            
        pad = kernel_size // 2
        if inverted:
            self.conv_inverted1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,padding = pad)
            self.conv_inverted2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size,padding = pad)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,padding = pad)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding = pad)
        self.nb =  nn.BatchNorm2d(out_channels) if do_bn else nn.Identity()
        self.dropout = nn.Dropout2d(dp) if dp > 0 else nn.Identity()
        self.act = act_func


class MobileBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                        inverted: bool = False,
                        act_func: nn.Module = nn.SiLU(),
                        do_bn: bool = True,
                        dp: float | int = 0.4):
        super().__init__()
    

        assert kernel_size % 2 != 0, 'Kernel must be odd'
            
          
        pad = kernel_size // 2  
        if inverted:
            self.conv_inverted = nn.ConvTranspose2d(in_channels, out_channels,kernel_size=kernel_size,padding=pad)
            
            self.depth_wise_inverted = nn.ConvTranspose2d(out_channels, out_channels,groups=out_channels,
                                          kernel_size=kernel_size)
                                          
            self.point_wise_inverte = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=1,padding=pad)
        else:
            
            self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,padding=pad)
            
            self.depth_wise = nn.Conv2d(out_channels, out_channels,groups=out_channels,
                                          kernel_size=kernel_size)
                                          
            self.point_wise = nn.Conv2d(out_channels, out_channels, kernel_size=1,padding=pad)
      
        
        self.nb =  nn.BatchNorm2d(out_channels) if do_bn else nn.Identity()
        self.dropout = nn.Dropout2d(dp) if dp > 0 else nn.Identity()
        self.act = act_func

class AdaptiveResBlock(nn.Module):
    def __init__(self, in_channels: int,
                        out_channels: int,
                        kernel_size: int,
                        use_mobile: bool = True, 
                        upscale: bool = False,
                        inverted: bool = False,
                        act_func: nn.Module = nn.SiLU(),
                        do_bn: bool = True,
                        do_sc: bool = True,
                        dp: float | int = 0.4):
        
        super().__init__()

        self.do_sc = do_sc

        if use_mobile:
            self.block = MobileBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       act_func=act_func,
                                       inverted=inverted,
                                       dp=dp,
                                       do_bn=do_bn)
        else:
            self.block = BaseBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       act_func=act_func,
                                       dp=dp,
                                       inverted=inverted,
                                       do_bn=do_bn)
        if upscale:
            self.scale = nn.ConvTranspose2d(in_channels=out_channels,
                                            out_channels=out_channels,
                                            kernel_size=3,
                                            stride=2,
                                            bias=False)
        else:
            self.scale = nn.MaxPool2d(kernel_size=2)

        if self.do_sc:

            if inverted:
                self.adapt_res = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else \
                                 nn.Identity()
            else:
                self.adapt_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else \
                                 nn.Identity()
            
        
    def forward(self, x):
        if self.do_sc:
            return self.scale(self.block(x) + self.adapt_res(x))
        else:
            return self.scale(self.block(x))