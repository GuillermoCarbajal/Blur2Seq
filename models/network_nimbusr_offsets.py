import torch
import torch.nn as nn

from . import basicblock as B
import numpy as np
from math import sqrt
import os

import subprocess

from utils.homographies import reblur_offsets


#
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""
def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
    st = 0
    return x[st::sf, st::sf, ...]


def filter_tensor(x, sf=3):
    z = torch.zeros(x.shape)
    z[..., ::sf, ::sf].copy_(x[..., ::sf, ::sf])
    return z


def hadamard(x, kmap):
    # Compute hadamard product (pixel-wise)
    # x: input of shape (C,H,W)
    # kmap: input of shape (H,W)

    C,H,W = x.shape
    kmap = kmap.view(1, H, W)
    kmap = kmap.repeat(C, 1, 1)
    return (x * kmap)


def convolve_tensor(x, k):
    # Compute product convolution
    # x: input of shape (C,H,W)
    # k: input of shape (H_k,W_k)

    H_k, W_k = k.shape
    C, H, W = x.shape
    k = torch.flip(k, dims =(0,1))
    k = k.view(1, 1, H_k, W_k).repeat(C, 1, 1, 1)
    x = x[None]
    x = torch.nn.functional.pad(x, (W_k//2,W_k//2,H_k//2,H_k//2), mode='circular')
    o = torch.nn.functional.conv2d(x, k, groups=C, padding=0, stride=1)
    return o[0]


def cross_correlate_tensor(x, k):
    # x: input of shape (C,H,W)
    # k: input of shape (H_k,W_k)
    
    C, H, W = x.shape
    H_k, W_k = k.shape
    k = k.view(1, 1, H_k, W_k).repeat(C, 1, 1, 1)
    x = x[None]
    x = torch.nn.functional.pad(x, (W_k//2,W_k//2,H_k//2,H_k//2), mode='circular')
    o = torch.nn.functional.conv2d(x, k, groups=C, padding=0, stride=1)
    return o[0]


def apply_saturation_function(img, max_value=0.5, get_derivative=False):
    '''
    Implements the saturated function proposed by Whyte
    https://www.di.ens.fr/willow/research/saturation/whyte11.pdf
    :param img: input image may have values above max_value
    :param max_value: maximum value
    :return:
    '''

    a=50
    img[img>max_value+0.5] = max_value+0.5  # to avoid overflow in exponential

    if get_derivative==False:
        saturated_image = img - 1.0/a*torch.log(1+torch.exp(a*(img - max_value)))
        output_image = torch.nn.functional.relu(saturated_image + (1 - max_value)) - (1 - max_value)

    else:
        output_image = 1.0 / ( 1 + torch.exp(a*(img - max_value)) )

    return output_image

def pmbm(x, offsets):
    # Apply pmbm model blurry = sum(K_i P_i K^{-1} x)
    # x: input of shape (B,C,H,W)
    # offsets: input of shape (B,Px2,H,W)
    
    assert len(x) == len(offsets), print("Batch size must be the same for all inputs")

    B,n_pos,H,W = offsets.shape
    n_pos=n_pos//2
    y = reblur_offsets(x, offsets.reshape(B,n_pos,2,H,W), forward=True)
    return y[0]



def pmbm_batch(x, offsets):
    # Apply pmbm model blurry = sum(K_i P_i K^{-1} x)
    # x: input of shape (B,C,H,W)
    # offsets: input of shape (B,Px2,H,W)

    assert len(x) == len(offsets), print("Batch size must be the same for all inputs")
    
    B,n_pos,H,W = offsets.shape
    n_pos=n_pos//2
    y = reblur_offsets(x, offsets.reshape(B,n_pos,2,H,W), forward=True)
    
    return y #torch.cat([pmbm(x[i:i+1], offsets[i:i+1]) for i in range(len(x))])


def transpose_pmbm(x, offsets, adjunct_offsets=False):
    # Apply the transpose of pmbm model blurry = sum(K_i P_i K^{-1} x)
    # x: input of shape (C,H,W)
    # offsets: input of shape (P,2,H,W)
    
    assert len(x) == len(offsets)
    B,n_pos,H,W = offsets.shape
    n_pos=n_pos//2
    y = reblur_offsets(x,offsets.reshape(B,n_pos,2,H,W),forward=False)
    if adjunct_offsets:
        # when the offsets correspond to the adjoint operator
        y = reblur_offsets(x,offsets.reshape(B,n_pos,2,H,W),forward=True)
    else:
        # by default BT is approximated
        y = reblur_offsets(x,offsets.reshape(B,n_pos,2,H,W),forward=False)
    
    
    return y[0]


def transpose_pmbm_batch(x, offsets, adjunct_offsets=False):
    # Apply the transpose of pmbm model model blurry = sum(H_i^T U_i x)
    # x: input of shape (B,C,H,W)
    # offsets: input of shape (B,P,H,W)

    assert len(x) == len(offsets) , print("Batch size must be the same for all inputs")

    B,n_pos,H,W = offsets.shape
    n_pos=n_pos//2
    if adjunct_offsets:
        # when the offsets correspond to the adjoint operator
        y = reblur_offsets(x,offsets.reshape(B,n_pos,2,H,W),forward=True)
    else:
        # by default BT is approximated
        y = reblur_offsets(x,offsets.reshape(B,n_pos,2,H,W),forward=False)
    
    return y #torch.cat([transpose_pmbm(x[i:i+1], offsets[i:i+1]) for i in range(len(x))])

"""
# --------------------------------------------
# (1) Prior module; ResUNet: act as a non-blind denoiser
# x_k = P(z_k, beta_k)
# --------------------------------------------
"""


class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return x


"""
# --------------------------------------------
# (2) Data module, closed-form solution
# --------------------------------------------
"""


class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()
        
    def forward_pos(self, x, STy, alpha, sf):
        I = torch.ones_like(STy) * alpha
        I[...,::sf,::sf] += 1
        return (STy + alpha * x) / I
        
    def forward_zer(self, x, STy, sf):
        res = x
        res[...,::sf,::sf] = STy[...,::sf,::sf]
        return res

    def forward(self, x, STy, alpha, sf, sigma):
        index_zer = (sigma.view(-1) == 0)
        index_pos = (sigma.view(-1) > 0)
        
        res = torch.zeros_like(x)
        
        res[index_zer,...] = self.forward_zer(x[index_zer,...], STy[index_zer,...], sf)
        res[index_pos,...] = self.forward_pos(x[index_pos,...], STy[index_pos,...], alpha[index_pos,...], sf)
        
        return res

"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""

class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x

"""
# --------------------------------------------
#   Main
# --------------------------------------------
"""


class NIMBUSR_Offsets(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(NIMBUSR_Offsets, self).__init__()

        self.d = DataNet()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        self.h = HyPaNet(in_nc=2, out_nc=(n_iter+1)*3, channel=h_nc)
        self.n = n_iter

    def forward(self, y, offsets, sf, sigma, offsets_BT=None):
        '''
        y: tensor, NxCxHxW
        offsets: tensor, NxPx2xHxW
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        '''
        
        adjunct_offsets = offsets_BT is not None
        offsets = torch.clone(offsets)
        # Initialization
        STy = upsample(y, sf)
        x_0 = nn.functional.interpolate(y, scale_factor=sf, mode='nearest')
        z_0 = x_0
        h_0 = pmbm_batch(x_0, offsets)
        u_0 = torch.zeros_like(z_0)
        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))
        #print(x_0.shape)
        #aux = [tensor2img(x_0)[:,:,[2,1,0]]]
        #print(aux.shape)
        #imssave(aux, 'initial_image.png')
        #imssave([tensor2img(h_0.detach())[:,:,[2,1,0]]], 'initial_h0.png')

        for i in range(self.n):
            # Hyper-params
            alpha = ab[:, i:i+1, ...]
            beta = ab[:, i+(self.n+1):i+(self.n+1)+1, ...]
            gamma = ab[:, i+2*(self.n+1):i+2*(self.n+1)+1, ...]

            # ADMM steps
            if adjunct_offsets:
                i_0 = x_0 - beta * transpose_pmbm_batch(h_0 - z_0 + u_0, offsets_BT,adjunct_offsets=True)
            else:
                # default behaviour
                i_0 = x_0 - beta * transpose_pmbm_batch(h_0 - z_0 + u_0, offsets)
            x_0 = self.p(torch.cat((i_0, gamma.repeat(1, 1, i_0.size(2), i_0.size(3))), dim=1))
            #aux.append(tensor2img(x_0.detach())[:,:,[2,1,0]])
            h_0 = pmbm_batch(x_0, offsets)
            z_0 = self.d(h_0 + u_0, STy, alpha, sf, sigma)
            u_0 = u_0 + h_0 - z_0
        
        
        #imssave(aux, 'img%d_step.png' % (np.random.randint(10)))
        # Hyper-params
        beta = ab[:, 2*self.n+1:2*(self.n+1), ...]
        gamma = ab[:, 3*self.n+2:, ...]

        if adjunct_offsets:
            i_0 = x_0 - beta * transpose_pmbm_batch(h_0 - z_0 + u_0, offsets_BT,adjunct_offsets=True)
        else:
            # default behaviour
            i_0 = x_0 - beta * transpose_pmbm_batch(h_0 - z_0 + u_0, offsets)

        x_0 = self.p(torch.cat((i_0, gamma.repeat(1, 1, i_0.size(2), i_0.size(3))), dim=1))

        return x_0


class NIMBUSR_Offsets_for_SI(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(NIMBUSR_Offsets_for_SI, self).__init__()

        self.d = DataNet()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        self.h = HyPaNet(in_nc=2, out_nc=(n_iter+1)*3, channel=h_nc)
        self.n = n_iter

    def forward(self, y, offsets, sf, sigma, offsets_BT=None):
        '''
        y: tensor, NxCxHxW
        offsets: tensor, NxPx2xHxW
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        '''
        
        adjunct_offsets = offsets_BT is not None
        offsets = torch.clone(offsets)
        # Initialization
        STy = upsample(y, sf)
        x_0 = nn.functional.interpolate(y, scale_factor=sf, mode='nearest')
        z_0 = x_0
        h_0 = pmbm_batch(x_0, offsets)
        u_0 = torch.zeros_like(z_0)
        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))
        #print(x_0.shape)
        #aux = [tensor2img(x_0)[:,:,[2,1,0]]]
        #print(aux.shape)
        #imssave(aux, 'initial_image.png')
        #imssave([tensor2img(h_0.detach())[:,:,[2,1,0]]], 'initial_h0.png')

        for i in range(self.n):
            # Hyper-params
            alpha = ab[:, i:i+1, ...]
            beta = ab[:, i+(self.n+1):i+(self.n+1)+1, ...]
            gamma = ab[:, i+2*(self.n+1):i+2*(self.n+1)+1, ...]

            # ADMM steps
            if adjunct_offsets:
                i_0 = x_0 - beta * transpose_pmbm_batch(h_0 - z_0 + u_0, offsets_BT,adjunct_offsets=True)
            else:
                # default behaviour
                i_0 = x_0 - beta * transpose_pmbm_batch(h_0 - z_0 + u_0, offsets)
            x_0 = self.p(torch.cat((i_0, gamma.repeat(1, 1, i_0.size(2), i_0.size(3))), dim=1))
            #aux.append(tensor2img(x_0.detach())[:,:,[2,1,0]])
            h_0 = pmbm_batch(x_0, offsets)

            #############################################################################
            #z_0 = self.d(h_0 + u_0, STy, alpha, sf, sigma)
            R = apply_saturation_function(z_0, max_value=1)
            R_prime = apply_saturation_function(z_0, max_value=1, get_derivative=True)
            #z_0 = h_0 + u_0 - 1./alpha * (R-y) * R_prime  # da mal
            #print('Saturated pixels: ' , np.count_nonzero(1-R_prime.detach().cpu().numpy())/(R_prime.shape[0]*R_prime.shape[1]*R_prime.shape[2]*R_prime.shape[3]))
            
            Lz = (1.0/(sigma**2))* R_prime
            aux = Lz*sigma**2
            z_0 = 1.0/( aux+ alpha) * (sigma**2 *Lz *z_0 + (y-R)*R_prime  + alpha*(h_0+u_0))
            ############################################################################################
            


            u_0 = u_0 + h_0 - z_0
        
        
        #imssave(aux, 'img%d_step.png' % (np.random.randint(10)))
        # Hyper-params
        beta = ab[:, 2*self.n+1:2*(self.n+1), ...]
        gamma = ab[:, 3*self.n+2:, ...]

        if adjunct_offsets:
            i_0 = x_0 - beta * transpose_pmbm_batch(h_0 - z_0 + u_0, offsets_BT,adjunct_offsets=True)
        else:
            # default behaviour
            i_0 = x_0 - beta * transpose_pmbm_batch(h_0 - z_0 + u_0, offsets)

        x_0 = self.p(torch.cat((i_0, gamma.repeat(1, 1, i_0.size(2), i_0.size(3))), dim=1))

        return x_0
