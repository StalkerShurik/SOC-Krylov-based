import numpy as np
import torch
import torch.nn.functional as F

import time

#z = torch.load("z.pt")

device = "cuda"

# z = torch.rand((64, 64, 64, 64)).float().to(device)
# curr_z = z

def transpose_filter(conv_filter):
    conv_filter_T = torch.transpose(conv_filter, 0, 1)
    conv_filter_T = torch.flip(conv_filter_T, [2, 3])
    return conv_filter_T

def basis_arnoldi_conv_qr(L, X, m, kernel_size):
    '''
    X - image tensoк B x M x H x W
    L - filter M x M x Kh x Kw
    m - number of iters
    return list of v and h for every element in batch B
    '''

    device = X.device
    dtype = X.dtype
    
    #h = torch.zeros([X.shape[0], m+1, m+1], device=device, dtype=dtype, requires_grad=False)
    
    #v = torch.nn.functional.normalize(X, dim = (-1,-2,-3)).unsqueeze(0)

    v = X.unsqueeze(0)

    for j in range(m):
        w  = F.conv2d(v[j], L, padding=(kernel_size//2, kernel_size//2))
        v = torch.cat((v, w.unsqueeze(0)))

    v = torch.permute(v, (1,0,2,3,4))
    v_shape = v.shape
    v = v.view((v.shape[0], v.shape[1], -1)).permute((0, 2, 1))

    #batch x m x in x out x sz

    gramm_v = torch.matmul(torch.permute(v, (0, 2, 1)), v)

    eps_eye = torch.eye(gramm_v.shape[1], device=device) * 1e-6
    eps_eye = eps_eye.unsqueeze(0)
    eps_eye_batched = eps_eye.repeat(gramm_v.shape[0], 1, 1)

    R = torch.linalg.cholesky(gramm_v + eps_eye_batched, upper=True)
    Ri = torch.linalg.inv(R)
    Q = torch.matmul(v, Ri)

    #Q, R = torch.linalg.qr(v)
    
    #print(Q.shape, Q1.shape)
    #print(R.shape, R1.shape)

    #Ri = torch.linalg.inv(R)

    #print(torch.linalg.norm(torch.matmul(Q, R) - v) / torch.linalg.norm(v))
    
    #print(torch.allclose(torch.matmul(Q, R), v))

    #print(Q.shape)

    #print(Q.view(v_shape)[:,:m].shape)

    #print(v.permute((0, 2, 1)).view(v_shape)[0,1] - F.conv2d(v.permute((0, 2, 1)).view(v_shape)[0], L, padding=(kernel_size//2, kernel_size//2))[0])

    #tmp1 = F.conv2d(Q.permute((0, 2, 1)).view(v_shape)[0,:m], L, padding=(kernel_size//2, kernel_size//2))
    #tmp2 = v[0][:,1:] @ Ri[0][:m,:m]
    
    #tmp2 = tmp2.permute((1, 0)).view(tmp1.shape)
    #print(tmp1.shape, tmp2.shape)

    #print(torch.linalg.norm(tmp1 - tmp2) / torch.linalg.norm(tmp1))

    #print(Q[0] @ Q[0].T)

    #print(Q.view(v_shape)[0][-1])

    #print(torch.linalg.norm(Q[3] @ R[3] - v[3]) / torch.linalg.norm(v[3]))

    #print(torch.linalg.norm(v[3] @ Ri[3] - Q[3]) / torch.linalg.norm(Q[3]))

    return Q.permute((0,2,1)).view(v_shape)[:,:m], torch.linalg.matmul(R[:,:m,1:m+1], Ri[:,:m,:m])

def emv_naive_batch(A, vec, iters):
    result = vec
    for i in range(iters-1):
        coef = iters - i - 1
        if coef == 0:
            coef = 1
        result = vec + (A * result.unsqueeze(1)).sum(dim=2) / coef
    return result

def emv_arnoldi_conv_qr(L, X, m, kernel_size, iter):
    device = X.device
    dtype = X.dtype

    v, h = basis_arnoldi_conv_qr(L, X, m, kernel_size)
    #print(h.shape)
    beta = torch.linalg.norm(torch.linalg.norm(X, dim=(-1,-2)), dim=-1)
    
    with torch.no_grad():
        vec = torch.zeros((h.shape[0], h.shape[1]), device=device, dtype=dtype, requires_grad=False)
        vec[:,0] = 1

        exp = emv_naive_batch(h, vec, iter)

    ans = (v * exp.view(exp.shape[0], exp.shape[1], 1, 1, 1)).sum(dim=1) * beta.view(beta.shape[0], 1, 1, 1)
    return ans

def emv_naive_conv(conv_filter_n, curr_z, num_terms, kernel_size):
    z = curr_z
    for i in range(1, num_terms+1):
        curr_z = F.conv2d(curr_z, conv_filter_n, 
                        padding=(kernel_size//2, 
                                kernel_size//2))/float(i)
        z = z + curr_z

    return z

kernel_size = 3
max_channels = 64
corection = 0.001

batch_size = 32

random_conv_filter = torch.nn.Parameter(torch.Tensor(torch.randn(max_channels,
                                        max_channels, kernel_size, 
                                        kernel_size)).cuda(),
                                        requires_grad=True)
random_conv_filter_T = transpose_filter(random_conv_filter)
conv_filter_n = 0.5*(random_conv_filter - random_conv_filter_T)

conv_filter_n = (conv_filter_n * corection).float().to(device)

z = torch.rand((batch_size, max_channels, 16, 16)).float().to(device)
curr_z = z

x = emv_arnoldi_conv_qr(conv_filter_n, z, 8, kernel_size, 15)

#x_true = emv_arnoldi_conv_qr(conv_filter_n, z, 2, kernel_size, 15)

x_true = emv_naive_conv(conv_filter_n, curr_z, 12, kernel_size)

#print(x_true.shape)
#print(x.shape)

#print(x[0] - x_true[0])

print(torch.linalg.norm(x - x_true) / torch.linalg.norm(x_true))

#true_conv = emv_naive_conv(conv_filter_n, curr_z.clone().detach(), 50, kernel_size)

#print(kernel_size, max_channels)

