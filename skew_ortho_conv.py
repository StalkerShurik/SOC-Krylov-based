from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
import time
import einops

import rakhuba_utils

from soc_arnoldi import emv_arnoldi_conv, arnoldi_dynamic_sheduler, naive_dynamic_sheduler, emv_lanczos_conv

def fantastic_four(conv_filter, num_iters=50):
    out_ch, in_ch, h, w = conv_filter.shape
    
    u1 = torch.randn((1, in_ch, 1, w), device='cuda', requires_grad=False)
    u1.data = l2_normalize(u1.data)

    u2 = torch.randn((1, in_ch, h, 1), device='cuda', requires_grad=False)
    u2.data = l2_normalize(u2.data)

    u3 = torch.randn((1, in_ch, h, w), device='cuda', requires_grad=False)
    u3.data = l2_normalize(u3.data)

    u4 = torch.randn((out_ch, 1, h, w), device='cuda', requires_grad=False)
    u4.data = l2_normalize(u4.data)
        
    v1 = torch.randn((out_ch, 1, h, 1), device='cuda', requires_grad=False)
    v1.data = l2_normalize(v1.data)

    v2 = torch.randn((out_ch, 1, 1, w), device='cuda', requires_grad=False)
    v2.data = l2_normalize(v2.data)

    v3 = torch.randn((out_ch, 1, 1, 1), device='cuda', requires_grad=False)
    v3.data = l2_normalize(v3.data)

    v4 = torch.randn((1, in_ch, 1, 1), device='cuda', requires_grad=False)
    v4.data = l2_normalize(v4.data)

    for i in range(num_iters):
        v1.data = l2_normalize((conv_filter.data*u1.data).sum((1, 3), keepdim=True).data)
        u1.data = l2_normalize((conv_filter.data*v1.data).sum((0, 2), keepdim=True).data)
        
        v2.data = l2_normalize((conv_filter.data*u2.data).sum((1, 2), keepdim=True).data)
        u2.data = l2_normalize((conv_filter.data*v2.data).sum((0, 3), keepdim=True).data)
        
        v3.data = l2_normalize((conv_filter.data*u3.data).sum((1, 2, 3), keepdim=True).data)
        u3.data = l2_normalize((conv_filter.data*v3.data).sum(0, keepdim=True).data)
        
        v4.data = l2_normalize((conv_filter.data*u4.data).sum((0, 2, 3), keepdim=True).data)
        u4.data = l2_normalize((conv_filter.data*v4.data).sum(1, keepdim=True).data)

    return u1, v1, u2, v2, u3, v3, u4, v4
    
def l2_normalize(tensor, eps=1e-12):
    norm = float(torch.sqrt(torch.sum(tensor.float() * tensor.float())))
    norm = max(norm, eps)
    ans = tensor / norm
    return ans

def transpose_filter(conv_filter):
    conv_filter_T = torch.transpose(conv_filter, 0, 1)
    conv_filter_T = torch.flip(conv_filter_T, [2, 3])
    return conv_filter_T

class SOC_Function(Function):
    @staticmethod
    def forward(ctx, curr_z, conv_filter):
        ctx.conv_filter = conv_filter
        kernel_size = conv_filter.shape[2]
        z = curr_z
        curr_fact = 1.
        print("SOMETHING STRANGE HAPPEN")
        for i in range(1, 14):
            curr_z = F.conv2d(curr_z, conv_filter, 
                              padding=(kernel_size//2, 
                                       kernel_size//2))/float(i)
            z = z + curr_z
        return z

    @staticmethod
    def backward(ctx, grad_output):
        conv_filter = ctx.conv_filter
        kernel_size = conv_filter.shape[2]
        grad_input = grad_output
        curr_fact = 1.
        for i in range(1, 14):
            grad_output = F.conv2d(grad_output, -conv_filter, 
                              padding=(kernel_size//2, 
                                       kernel_size//2))/float(i)
            grad_input = grad_input + grad_output

        return grad_input, None

class SOC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, 
                 bias=True, train_terms=6, eval_terms=12, init_iters=50, update_iters=1, 
                 update_freq=200, correction=0.7):
        super(SOC, self).__init__()

        self.was_training = 1
        self.epoch = 0

        assert (stride==1) or (stride==2)

        # train_terms = 2
        # eval_terms = 12
        #print(f"INIT: TRAIN_TERMS={train_terms} EVAL_TERMS={eval_terms}")

        self.time_conv_sum = 0
        self.time_exp_sum = 0
        self.counter = 0

        self.init_iters = init_iters
        self.out_channels = out_channels
        self.in_channels = in_channels*stride*stride
        self.max_channels = max(self.out_channels, self.in_channels)
        
        self.stride = stride
        self.kernel_size = kernel_size
        self.update_iters = update_iters
        self.update_freq = update_freq
        self.total_iters = 0
        self.train_terms = train_terms
        self.eval_terms = eval_terms
        
        if kernel_size == 1:
            correction = 1.0
        
        self.random_conv_filter = nn.Parameter(torch.Tensor(torch.randn(self.max_channels,
                                               self.max_channels, self.kernel_size, 
                                               self.kernel_size)).cuda(),
                                               requires_grad=True)
        random_conv_filter_T = transpose_filter(self.random_conv_filter)
        conv_filter = 0.5*(self.random_conv_filter - random_conv_filter_T)
        
        with torch.no_grad():
            u1, v1, u2, v2, u3, v3, u4, v4 = fantastic_four(conv_filter, 
                                                num_iters=self.init_iters)
            self.u1 = nn.Parameter(u1, requires_grad=False)
            self.v1 = nn.Parameter(v1, requires_grad=False)
            self.u2 = nn.Parameter(u2, requires_grad=False)
            self.v2 = nn.Parameter(v2, requires_grad=False)
            self.u3 = nn.Parameter(u3, requires_grad=False)
            self.v3 = nn.Parameter(v3, requires_grad=False)
            self.u4 = nn.Parameter(u4, requires_grad=False)
            self.v4 = nn.Parameter(v4, requires_grad=False)
            
        self.correction = nn.Parameter(torch.Tensor([correction]).cuda(), requires_grad=False)
            
        self.enable_bias = bias
        if self.enable_bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.out_channels).cuda(), requires_grad=True)
        else:
            self.bias = None
        self.reset_parameters()

        #print(f"INIT: {self.random_conv_filter.dtype} {self.bias.dtype}")
            
    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.max_channels)
        nn.init.normal_(self.random_conv_filter, std=stdv)
        
        stdv = 1.0 / np.sqrt(self.out_channels)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)
            
    def update_sigma(self):
        if self.training:
            if self.total_iters % self.update_freq == 0:
                update_iters = self.init_iters
            else:
                update_iters = self.update_iters
            self.total_iters = self.total_iters + 1
        else:
            update_iters = 0
        
        random_conv_filter_T = transpose_filter(self.random_conv_filter)
        conv_filter = 0.5*(self.random_conv_filter - random_conv_filter_T)
        pad_size = conv_filter.shape[2]//2
        with torch.no_grad():
            for i in range(update_iters):
                self.v1.data = l2_normalize((conv_filter*self.u1).sum(
                                            (1, 3), keepdim=True).data)
                self.u1.data = l2_normalize((conv_filter*self.v1).sum(
                                            (0, 2), keepdim=True).data)
                self.v2.data = l2_normalize((conv_filter*self.u2).sum(
                                            (1, 2), keepdim=True).data)
                self.u2.data = l2_normalize((conv_filter*self.v2).sum(
                                            (0, 3), keepdim=True).data)
                self.v3.data = l2_normalize((conv_filter*self.u3).sum(
                                            (1, 2, 3), keepdim=True).data)
                self.u3.data = l2_normalize((conv_filter*self.v3).sum(
                                            0, keepdim=True).data)
                self.v4.data = l2_normalize((conv_filter*self.u4).sum(
                                            (0, 2, 3), keepdim=True).data)
                self.u4.data = l2_normalize((conv_filter*self.v4).sum(
                                            1, keepdim=True).data)

        func = torch.min
        sigma1 = torch.sum(conv_filter*self.u1*self.v1)
        sigma2 = torch.sum(conv_filter*self.u2*self.v2)
        sigma3 = torch.sum(conv_filter*self.u3*self.v3)
        sigma4 = torch.sum(conv_filter*self.u4*self.v4)
        sigma = func(func(func(sigma1, sigma2), sigma3), sigma4)
        return sigma

    def forward(self, x):

        random_conv_filter_T = transpose_filter(self.random_conv_filter)
        conv_filter_skew = 0.5*(self.random_conv_filter - random_conv_filter_T)
        sigma = self.update_sigma()
        conv_filter_n = (self.correction * conv_filter_skew)/sigma
        
        if self.training:
            self.was_training = 1
            num_terms = self.train_terms
        else:
            if self.was_training:
                self.was_training = 0
                self.epoch += 1

            num_terms = self.eval_terms
        
        if self.stride > 1:
            x = einops.rearrange(x, "b c (w k1) (h k2) -> b (c k1 k2) w h", 
                                 k1=self.stride, k2=self.stride)
        
        if self.out_channels > self.in_channels:
            diff_channels = self.out_channels - self.in_channels
            p4d = (0, 0, 0, 0, 0, diff_channels, 0, 0)
            curr_z = F.pad(x, p4d)
        else:
            curr_z = x
            
        #RADICALLY DECREASE PERFOMANCE
        #torch.autograd.set_detect_anomaly(True) 

        #curr_z = curr_z.float()
        #conv_filter_n = conv_filter_n.float()

        # curr_z.to(torch.float64)
        # conv_filter_n.to(torch.float64)

        # orth_err = rakhuba_utils.check_convolution_orthogonality_naive(conv_filter_n, curr_z, num_terms, self.kernel_size)

        z = curr_z

        #z=z.double()
        #curr_z = curr_z.double()
        #conv_filter_n = conv_filter_n.double()
        
        curr_z_copy = curr_z.clone().detach()

        #num_terms = naive_dynamic_sheduler(self.epoch)

        #num_terms = 5

        #NAIVE--------------------------------------

        # time_start_glob = time.time()
        # print(10 * "----")
        # print(curr_z.shape)
        #num_terms = 20
        # if self.training:
        #    num_terms = 3
        #for i in range(1, num_terms+1):
            #time_start = time.time()
            #curr_z = F.conv2d(curr_z, conv_filter_n, 
            #             padding=(self.kernel_size//2, 
            #                      self.kernel_size//2))/float(i)
            #torch.cuda.synchronize()
            #time_end = time.time()
            #self.time_conv_sum += (time_end - time_start)
            #print("naive one conv", time_end - time_start)
            #z = z + curr_z
        #torch.cuda.synchronize()
        #time_end_glob = time.time()
        # self.time_exp_sum += (time_end_glob - time_start_glob)
        # print("naive time for convs", time_end_glob - time_start_glob)
        
        # self.counter += 1

        # mean_naive_one_conv = self.time_conv_sum / (num_terms * self.counter)
        # mean_naive_exp = self.time_exp_sum / self.counter

        #print("mean naive one conv", mean_naive_one_conv)
        #print("mean naive exp", mean_naive_exp)
        
        # print(f"%{(num_terms * mean_naive_one_conv) / mean_naive_exp}")

        #NAIVE--------------------------------------

        # if self.training:
        # print(conv_filter_n.shape, curr_z.shape)
        # naive_true_err = rakhuba_utils.compare_with_true(conv_filter_n, curr_z_copy, self.kernel_size, z)
        # print(f"naive_true_err {naive_true_err}")
        
        # orthogonality_error_naive = rakhuba_utils.check_convolution_orthogonality_naive(conv_filter_n, z, num_terms, self.kernel_size, curr_z_copy)
        # print(f"orthogonality_error_naive {orthogonality_error_naive}")

        # hatch_error_naive = rakhuba_utils.hatchinson_test_naive(conv_filter_n, curr_z, num_terms, self.kernel_size)
        # print(f"hatchinson_error_naive {hatch_error_naive}")

        #with open('errors','a') as f:
        #    f.write(f'{str(orth_err.item())} {str(brute_err.item())}\n')


        #ARNOLDI-------------------------------------
        #time_start = time.time()
        BASIS_SIZE = 8
        EXP_TERMS = 30

        if self.training:
            BASIS_SIZE = 3
            EXP_TERMS = 30
        
        #print(self.epoch)
        #BASIS_SIZE, EXP_TERMS = arnoldi_dynamic_sheduler(self.epoch)
        #print(10 * "----------")
        #print(conv_filter_n.shape, curr_z.shape, torch.norm(curr_z).item())
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:

        non_ort = 15

        z = emv_arnoldi_conv(conv_filter_n, curr_z, BASIS_SIZE, self.kernel_size, EXP_TERMS, non_ort)

        #print(prof)
        #torch.cuda.synchronize()
        #time_end = time.time()
        #print("arnoldi time for covs", time_end - time_start)
        #ARNOLDI-------------------------------------

        # if self.training:
            
        #     print(conv_filter_n.shape, curr_z.shape)

        #     arnoldi_true_err = rakhuba_utils.compare_with_true(conv_filter_n, curr_z_copy, self.kernel_size, z)
        #     print(f"arnoldi_true_err {arnoldi_true_err}")        

        #     orthogonality_error_arnoldi = rakhuba_utils.check_convolution_orthogonality_arnoldi(conv_filter_n, z, BASIS_SIZE, EXP_TERMS, self.kernel_size, curr_z_copy, non_ort)
        #     print(f"orthogonality_error_arnoldi {orthogonality_error_arnoldi}")

        #     hatch_error_arnoldi = rakhuba_utils.hatchinson_test_arnoldi(conv_filter_n, curr_z, BASIS_SIZE, EXP_TERMS, self.kernel_size, non_ort)
        #     print(f"hatchinson_error_arnoldi {hatch_error_arnoldi}")


        #LANCZOS

        # BASIS_SIZE = 7
        # EXP_TERMS = 15

        # if self.training:
        #     BASIS_SIZE = 5
        #     EXP_TERMS = 15

        # z = emv_lanczos_conv(conv_filter_n, curr_z, BASIS_SIZE, self.kernel_size, EXP_TERMS)

        # if self.training:
        #     #print(conv_filter_n.shape, curr_z.shape)            
        #     lanczos_true_err = rakhuba_utils.compare_with_true(conv_filter_n, curr_z_copy, self.kernel_size, z)
        #     print(f"lanczos_true_err {lanczos_true_err}") 
        #     orthogonality_error_lanczos = rakhuba_utils.check_convolution_orthogonality_lanczos(conv_filter_n, z, BASIS_SIZE, EXP_TERMS, self.kernel_size, curr_z_copy)
        #     print(f"orthogonality_error_lanczos {orthogonality_error_lanczos}")

        #     hatch_error_lanczos = rakhuba_utils.hatchinson_test_lanczos(conv_filter_n, curr_z, BASIS_SIZE, EXP_TERMS, self.kernel_size)
        #     print(f"hatchinson_error_lanczos {hatch_error_lanczos}")


        if self.out_channels < self.in_channels:
            z = z[:, :self.out_channels, :, :]
            
        if self.enable_bias:
            z = z + self.bias.view(1, -1, 1, 1)
        return z