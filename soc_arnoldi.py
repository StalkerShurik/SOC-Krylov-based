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

def basis_arnoldi_conv(L, X, m, kernel_size, non_ort = 15):
    '''
    X - image tensoк B x M x H x W
    L - filter M x M x Kh x Kw
    m - number of iters
    return list of v and h for every element in batch B
    '''

    device = X.device
    dtype = X.dtype
    
    h = torch.zeros([X.shape[0], m+1, m+1], device=device, dtype=dtype, requires_grad=False)
    
    v = torch.nn.functional.normalize(X, dim = (-1,-2,-3)).unsqueeze(0)

    for j in range(m):
        w  = F.conv2d(v[j], L, padding=(kernel_size//2, kernel_size//2))
        for i in range(j+1):
            with torch.no_grad():
                h[:,i,j] = torch.sum(w * v[i], axis=(1,2,3))

            if j <= non_ort:
                w = w - h[:,i,j].clone().view(h.shape[0], 1, 1, 1) * v[i]

        with torch.no_grad():
            h[:,j+1,j] = torch.linalg.norm(torch.linalg.norm(w, dim=-1), dim=(-1,-2))

        new_v = w * torch.reciprocal(h[:,j+1,j]).clone().view(h.shape[0], 1, 1, 1)

        v = torch.cat((v, new_v.unsqueeze(0)))
    return v[:m], h[:,:m,:m]

def basis_arnoldi_conv_qr(L, X, m, kernel_size):
    '''
    X - image tensoк B x M x H x W
    L - filter M x M x Kh x Kw
    m - number of iters
    return list of v and h for every element in batch B
    '''

    device = X.device
    dtype = X.dtype
    
    v = X.unsqueeze(0)

    for j in range(m):
        w  = F.conv2d(v[j], L, padding=(kernel_size//2, kernel_size//2))
        v = torch.cat((v, w.unsqueeze(0)))

    v = torch.permute(v, (1,0,2,3,4))
    v_shape = v.shape
    v = v.view((v.shape[0], v.shape[1], -1)).permute((0, 2, 1))

    # eps_eye = torch.eye(gramm_v.shape[1], device=device) * 1e-6
    # eps_eye = eps_eye.unsqueeze(0)
    # eps_eye_batched = eps_eye.repeat(gramm_v.shape[0], 1, 1)


    #with torch.no_grad():
    #    gramm_v = torch.matmul(torch.permute(v, (0, 2, 1)), v)
    #    R = torch.linalg.cholesky(gramm_v, upper=True)
    #    Ri = torch.linalg.inv(R)
    
    #Q = torch.matmul(v, Ri)

    Q, R = torch.linalg.qr(v)
    Ri = torch.linalg.inv(R)

    return Q.permute((0,2,1)).view(v_shape)[:,:m], torch.linalg.matmul(R[:,:m,1:m+1], Ri[:,:m,:m])

def emv_arnoldi_conv_qr(L, X, m, kernel_size, iter):
    device = X.device
    dtype = X.dtype

    v, h = basis_arnoldi_conv_qr(L, X, m, kernel_size)
    beta = torch.linalg.norm(torch.linalg.norm(X, dim=(-1,-2)), dim=-1)
    
    with torch.no_grad():
        vec = torch.zeros((h.shape[0], h.shape[1]), device=device, dtype=dtype, requires_grad=False)
        vec[:,0] = 1

        exp = emv_naive_batch(h, vec, iter)

    ans = (v * exp.view(exp.shape[0], exp.shape[1], 1, 1, 1)).sum(dim=1) * beta.view(beta.shape[0], 1, 1, 1)
    return ans

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

def emv_arnoldi_conv(L, X, m, kernel_size, iter, non_ort = 15):
    device = X.device
    dtype = X.dtype

    v, h = basis_arnoldi_conv(L, X, m, kernel_size, non_ort)
    beta = torch.linalg.norm(torch.linalg.norm(X, dim=(-1,-2)), dim=-1)
    v = torch.permute(v, (1,0,2,3,4))

    with torch.no_grad():
        vec = torch.zeros((h.shape[0], h.shape[1]), device=device, dtype=dtype, requires_grad=False)
        vec[:,0] = 1

        exp = emv_naive_batch(h, vec, iter)

    ans = (v * exp.view(exp.shape[0], exp.shape[1], 1, 1, 1)).sum(dim=1) * beta.view(beta.shape[0], 1, 1, 1)
    return ans

def basis_lancsoz(L, X, m, kernel_size):
    
    device = X.device
    dtype = X.dtype

    batch_size = X.shape[0]

    alpha = torch.zeros((batch_size, m+2), device=device, dtype=dtype, requires_grad=False)
    beta = torch.zeros((batch_size, m+2), device=device, dtype=dtype, requires_grad=False)
    delta = torch.zeros((batch_size, m+2), device=device, dtype=dtype, requires_grad=False)
    
    v = torch.zeros(X.shape, device=device, dtype=dtype).unsqueeze(0)
    v = torch.cat((v, torch.nn.functional.normalize(X, dim = (-1,-2,-3)).unsqueeze(0)))

    w = torch.zeros(X.shape, device=device, dtype=dtype).unsqueeze(0)
    w = torch.cat((w, torch.nn.functional.normalize(X, dim = (-1,-2,-3)).unsqueeze(0)))

    for j in range(1,m+1):

        v_conv = F.conv2d(v[j], L, padding=(kernel_size//2, kernel_size//2))

        with torch.no_grad():
            alpha = alpha.clone()
            alpha[:,j] = (v_conv * w[j]).sum((1,2,3))

        new_v = v_conv - alpha[:,j].view(batch_size, 1, 1, 1) * v[j] - beta[:,j].view(batch_size, 1, 1, 1) * v[j-1]
        new_w = F.conv2d(w[j], -L, padding=(kernel_size//2, kernel_size//2)) - alpha[:,j].view(batch_size, 1, 1, 1) * w[j] - delta[:,j].view(batch_size, 1, 1, 1) * w[j-1]

        dot_prod = (new_v * new_w).sum((1,2,3))

        with torch.no_grad():
            beta = beta.clone()
            beta[:,j+1] = torch.sqrt(torch.abs(dot_prod))
            delta = delta.clone()
            delta[:,j+1] = beta[:,j+1] * torch.sign(dot_prod)

        new_v = new_v / delta[:,j+1].view(batch_size, 1, 1, 1)
        new_w = new_w / beta[:,j+1].view(batch_size, 1, 1, 1)

        v = torch.cat((v, new_v.unsqueeze(0)))
        w = torch.cat((w, new_w.unsqueeze(0)))

    return v[1:-1], alpha[:,1:], beta[:,2:], delta[:,2:]

def emv_lanczos_conv(L, X, m, kernel_size, iter):
    device = X.device
    dtype = X.dtype

    v, alpha, beta, delta = basis_lancsoz(L, X, m, kernel_size)
        
    norm = torch.linalg.norm(torch.linalg.norm(X, dim=(-1,-2)), dim=-1)
    v = torch.permute(v, (1,0,2,3,4))
    
    with torch.no_grad():
        T = torch.zeros((X.shape[0], m, m), device=device, dtype=dtype, requires_grad=False)

        for i in range(m):
            T[:, i, i] = alpha[:,i]

        for i in range(m-1):
            T[:, i, i+1] = beta[:,i]
            T[:, i+1, i] = delta[:,i]

    with torch.no_grad():
        vec = torch.zeros((X.shape[0], m), device=device, dtype=dtype, requires_grad=False)
        vec[:,0] = 1
        exp = emv_naive_batch(T, vec, iter)

    ans = (v * exp.view(exp.shape[0], exp.shape[1], 1, 1, 1)).sum(dim=1) * norm.view(norm.shape[0], 1, 1, 1)
    
    return ans