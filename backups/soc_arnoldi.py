import torch
import torch.nn.functional as F

import time

def arnoldi_dynamic_sheduler(epoch, total_epochs=100):
    if epoch < 0.7 * total_epochs:
        return 2, 10
    elif epoch < 0.8 * total_epochs:
        return 3, 10
    elif epoch < 0.9 * total_epochs:
        return 4, 10
    else:
        return 5, 10

def naive_dynamic_sheduler(epoch, total_epochs=200):
    if epoch < 0.55 * total_epochs:
        return 2
    elif epoch < 0.6 * total_epochs:
        return 3
    elif epoch < 0.65 * total_epochs:
        return 4
    elif epoch < 0.7 * total_epochs:
        return 5
    elif epoch < 0.75 * total_epochs:
        return 6
    elif epoch < 0.8 * total_epochs:
        return 7
    elif epoch < 0.85 * total_epochs:
        return 8
    elif epoch < 0.9 * total_epochs:
        return 9
    elif epoch < 0.95 * total_epochs:
        return 10
    else:
        return 3

def basis_arnoldi_conv(L, X, m, kernel_size):
    '''
    X - image tensoк B x M x H x W
    L - filter M x M x Kh x Kw
    m - number of iters
    return list of v and h for every element in batch B
    '''

    #h = torch.zeros([X.shape[0], m+1, m+1]).cuda().double()
    h = torch.zeros([X.shape[0], m+1, m+1]).cuda().half()
    norms = torch.linalg.norm(torch.linalg.norm(X, dim=(-1,-2)), dim=-1)
    
    #if torch.any(norms == 0):
    #    print("HERE ZERO NORM!!!")

    v = (X * torch.reciprocal(norms.view(norms.shape[0], 1, 1, 1)))[None,:,:,:,:].cuda()
    for j in range(m):
        #time_begin = time.time()
        w  = F.conv2d(v[j], L, padding=(kernel_size//2, kernel_size//2))
        #torch.cuda.synchronize()
        #time_end = time.time()
        #print(time_end - time_begin)

        for i in range(j+1):
            h[:,i,j] = torch.sum(w * v[i], axis=(1,2,3))
            #w = w - h[:,i,j].view(h.shape[0], 1, 1, 1).clone().detach() * v[i]
            w = w - h[:,i,j].clone().view(h.shape[0], 1, 1, 1) * v[i]

        h[:,j+1,j] = torch.linalg.norm(torch.linalg.norm(w, dim=-1), dim=(-1,-2))
        
        if torch.any(h[:,j+1,j] == 0):
            h = h[:,:j+1,:j+1]
            break

        #new_v = w * torch.reciprocal(h[:,j+1,j]).clone().detach().view(h.shape[0], 1, 1, 1)
        new_v = w * torch.reciprocal(h[:,j+1,j]).clone().view(h.shape[0], 1, 1, 1)

        v = torch.cat((v, new_v[None,:,:,:,:]))
    return v[:m],h[:,:m,:m]

def emv_naive_batch(A, vec, iters): #exp mul vec
    #A - batch of matrices B x N x N
    #vec - batch of vectors B x N
    #return batch of exp(A) @ vec
    result = vec
    for i in range(iters-1):
        coef = iters - i - 1
        if coef == 0:
            coef = 1
        result = vec + (A * result.unsqueeze(1)).sum(dim=2) / coef
    return result

def emv_arnoldi_conv(L, X, m, kernel_size, iter):
    v, h = basis_arnoldi_conv(L, X, m, kernel_size)
    beta = torch.linalg.norm(torch.linalg.norm(X, dim=(-1,-2)), dim=-1)
    v = torch.permute(v, (1,0,2,3,4))

    vec = torch.zeros((h.shape[0], h.shape[1])).cuda()
    vec[:,0] = 1
    #vec = vec.double()
    vec = vec.half()

    exp = emv_naive_batch(h, vec, iter)

    ans = (v * exp.view(exp.shape[0], exp.shape[1], 1, 1, 1)).sum(dim=1) * beta.view(beta.shape[0], 1, 1, 1)
    return ans
