import numpy as np
import torch
import torch.nn.functional as F

import time

#z = torch.load("z.pt")

device = "cuda"

# z = torch.rand((64, 64, 64, 64)).float().to(device)
# curr_z = z
# conv_filter_n = torch.load("conv_filter_n.pt").float().to(device)


def transpose_filter(conv_filter):
    conv_filter_T = torch.transpose(conv_filter, 0, 1)
    conv_filter_T = torch.flip(conv_filter_T, [2, 3])
    return conv_filter_T

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

    time_conv = 0
    time_ort = 0

    for j in range(m):
        #time_1 = time.time()
        w  = F.conv2d(v[j], L, padding=(kernel_size//2, kernel_size//2))
        #torch.cuda.synchronize()
        #time_2 = time.time()
        for i in range(j+1):
            with torch.no_grad():
                h[:,i,j] = torch.sum(w * v[i], axis=(1,2,3))

            if j <= non_ort:
                w = w - h[:,i,j].clone().view(h.shape[0], 1, 1, 1) * v[i]
        #torch.cuda.synchronize()
        #time_3 = time.time()
        #time_conv += (time_2 - time_1)
        #time_ort += (time_3 - time_2)

        with torch.no_grad():
            h[:,j+1,j] = torch.linalg.norm(torch.linalg.norm(w, dim=-1), dim=(-1,-2))

        new_v = w * torch.reciprocal(h[:,j+1,j]).clone().view(h.shape[0], 1, 1, 1)

        v = torch.cat((v, new_v.unsqueeze(0)))
    return v[:m], h[:,:m,:m]#, time_conv, time_ort

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
    return ans#, time_conv, time_ort

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

def emv_naive_conv(conv_filter_n, curr_z, num_terms, kernel_size):
    z = curr_z
    for i in range(1, num_terms+1):
        curr_z = F.conv2d(curr_z, conv_filter_n, 
                        padding=(kernel_size//2, 
                                kernel_size//2))/float(i)
        z = z + curr_z

    return z

@torch.no_grad()
def hatchinson_test_arnoldi(conv_filter_n, curr_z, basis_size, exp_terms, kernel_size, ort):

    conv_filter_n = conv_filter_n.float()

    conv_filter_n_T =  -conv_filter_n
    
    N_g_vectors = 5
    summ_g = 0

    for i in range(N_g_vectors):
        g = torch.randn(curr_z.shape).float().cuda()

        tmp1 = emv_arnoldi_conv(conv_filter_n, g, basis_size, kernel_size, exp_terms, ort)
        tmp2 = (emv_arnoldi_conv(conv_filter_n_T, tmp1, basis_size, kernel_size, exp_terms, ort) - g)

        summ_g += torch.norm(tmp2)**2

    return (summ_g.item() / N_g_vectors)

@torch.no_grad()
def hatchinson_test_naive(conv_filter_n, curr_z, num_terms, kernel_size):
     
    N_g_vectors = 5
    summ_g = 0

    conv_filter_n = conv_filter_n.float()

    conv_filter_n_T = -conv_filter_n

    for i in range(N_g_vectors):
        g = torch.randn(curr_z.shape).float().cuda()
        
        g_copy = g.clone()
        z = g

        for i in range(1, num_terms+1):
            g = F.conv2d(g, conv_filter_n, 
                              padding=(kernel_size//2, 
                                       kernel_size//2))/float(i)
            z = z + g

        z1 = z
        for i in range(1, num_terms+1):
            z = F.conv2d(z, conv_filter_n_T, 
                              padding=(kernel_size//2, 
                                       kernel_size//2))/float(i)
            z1 = z1 + z      

        summ_g += torch.norm(z1 - g_copy) **2


    return (summ_g.item() / N_g_vectors)

torch.no_grad()
def hatchinson_test_lanczos(conv_filter_n, curr_z, basis_size, exp_terms, kernel_size):

    conv_filter_n = conv_filter_n.float()

    conv_filter_n_T =  -conv_filter_n
    
    N_g_vectors = 5
    summ_g = 0

    for i in range(N_g_vectors):
        g = torch.randn(curr_z.shape).float().cuda()

        tmp1 = emv_lanczos_conv(conv_filter_n, g, basis_size, kernel_size, exp_terms)

        tmp2 = (emv_lanczos_conv(conv_filter_n_T, tmp1, basis_size, kernel_size, exp_terms) - g)

        summ_g += torch.norm(tmp2)**2

    return (summ_g.item() / N_g_vectors)

#print(conv_filter_n.shape, curr_z.shape)

#torch.autograd.set_detect_anomaly(True)

#_ = emv_arnoldi_conv(conv_filter_n, curr_z, 5, kernel_size, 10)

#with torch.autograd.profiler.profile(use_cuda=True) as prof:
#time_1 = time.time()
#res = emv_naive_conv(conv_filter_n, curr_z, 12, kernel_size)
#print("---------")
#torch.cuda.synchronize(device=None)

BASIS_SIZE_ARNOLDI = 3
BASIS_SIZE_LANCZOS = 8 
EXP_TERMS_ARNOLDI=30

NAIVE_TERMS = 12

# time_2 = time.time()
# res_arnoldi_1 = emv_lanczos_conv(conv_filter_n, curr_z, TERMS, kernel_size, 20)
# torch.cuda.synchronize(device=None)
# time_3 = time.time()
# res_arnoldi_2 = emv_arnoldi_conv(conv_filter_n, curr_z, 12, kernel_size, 20)
# torch.cuda.synchronize(device=None)
# time_4 = time.time()
#print(time_4 - time_3, time_3 - time_2)

kernel_size = 3
max_channels = 100
corection = 0.01

batch_size = 16

for i in range(5):

    random_conv_filter = torch.nn.Parameter(torch.Tensor(torch.randn(max_channels,
                                            max_channels, kernel_size, 
                                            kernel_size)).cuda(),
                                            requires_grad=True)
    random_conv_filter_T = transpose_filter(random_conv_filter)
    conv_filter_n = 0.5*(random_conv_filter - random_conv_filter_T)

    conv_filter_n = (conv_filter_n * corection)

    z = torch.rand((batch_size, max_channels, 64, 64)).float().to(device)
    curr_z = z

    true_conv = emv_naive_conv(conv_filter_n, curr_z.clone().detach(), 50, kernel_size)

    print(kernel_size, max_channels)

    # time_loop_1 = time.time()
    # res_arnoldi_loop = emv_arnoldi_conv(conv_filter_n, curr_z.clone().detach(), BASIS_SIZE_ARNOLDI, kernel_size, EXP_TERMS_ARNOLDI)
    # torch.cuda.synchronize(device=None)
    # time_loop_2 = time.time()
    # print(f"arnnoldi_forward {time_loop_2 - time_loop_1}")
    # torch.mean(res_arnoldi_loop).backward()
    # torch.cuda.synchronize(device=None)
    # time_loop_3 = time.time()
    # print(f"arnoldi_backward {time_loop_3 - time_loop_2}")
    # arnoldi_time = time_loop_2 - time_loop_1
    # print(torch.linalg.norm(true_conv - res_arnoldi_loop) / torch.linalg.norm(true_conv).item())
    # print("hatch_arnoldi", hatchinson_test_arnoldi(conv_filter_n, curr_z, BASIS_SIZE_ARNOLDI, EXP_TERMS_ARNOLDI, kernel_size, 30))

    time_loop_1 = time.time()
    res_lanczos_loop = emv_lanczos_conv(conv_filter_n, curr_z.clone().detach(), BASIS_SIZE_LANCZOS, kernel_size, EXP_TERMS_ARNOLDI)
    torch.cuda.synchronize(device=None)
    time_loop_2 = time.time()
    print(f"forward_lanzcos {time_loop_2 - time_loop_1}")
    torch.mean(res_lanczos_loop).backward()
    torch.cuda.synchronize(device=None)
    time_loop_3 = time.time()
    print(f"backward_lanzcos {time_loop_3 - time_loop_2}")
    print(torch.linalg.norm(true_conv - res_lanczos_loop) / torch.linalg.norm(true_conv))
    #print("hatch_lanczos", hatchinson_test_lanczos(conv_filter_n, curr_z, BASIS_SIZE_LANCZOS, EXP_TERMS_ARNOLDI, kernel_size))

    #res_naive_loop = emv_naive_conv(conv_filter_n, curr_z.clone().detach(), NAIVE_TERMS, kernel_size)
    #torch.mean(res_naive_loop).backward()
    #torch.cuda.synchronize(device=None)
    #time_loop_3 = time.time()
    
    #print(hatchinson_test_naive(conv_filter_n, curr_z, NAIVE_TERMS, kernel_size))

    #print((time_loop_2 - time_loop_1) / (time_loop_3 - time_loop_2))

#print(f"naive time:{time_2 - time_1}")
#print(f"lanczos time:{time_3 - time_2}")

#print(res.requires_grad)

#torch.mean(res).backward()

#print("ERROR:", torch.linalg.norm(res_arnoldi - res_lanczos) / torch.linalg.norm(res_arnoldi))

#print(prof)


#ground_truth = emv_naive_conv(conv_filter_n, curr_z, 50, kernel_size) 

# errors1 = []
# errors2 = []

# range_to_iterate = list(range(1,11))

# for i in range_to_iterate:
#     res1 = emv_arnoldi_conv(conv_filter_n, curr_z, i, kernel_size, 20)
#     res2 = emv_naive_conv(conv_filter_n, curr_z, i, kernel_size)
#     errors1.append(error(res1, ground_truth).cpu().detach().numpy())
#     errors2.append(error(res2, ground_truth).cpu().detach().numpy())


# import matplotlib.pyplot as plt

# fig,ax = plt.subplots()
# line1, = ax.plot(range_to_iterate, errors1, label="arnoldi")
# line2, = ax.plot(range_to_iterate, errors2, label="naive")
# ax.legend(handles=[line1, line2])
# ax.set_xlabel("k")
# ax.set_ylabel("relative error")
# ax.set_yscale('log')
# ax.set_title('relative error of exp(X) mul v using aproximations with k iters')
# plt.show()
