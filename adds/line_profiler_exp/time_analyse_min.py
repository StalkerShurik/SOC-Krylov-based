import torch
import torch.nn.functional as F

import time


device = "cuda"

z = torch.rand((128, 64, 32, 32)).half().to(device)
curr_z = z
conv_filter_n = torch.load("conv_filter_n.pt").to(device)

# num_terms = 5
kernel_size=3

import time

#@profile
@torch.no_grad()
def emv_naive_conv(L, X, iters, kernel_size): #exp mul vec
    curr_z = X
    z = curr_z

    for i in range(1, iters+1):

        time_start = time.time()

        curr_z = F.conv2d(curr_z, L, padding=(kernel_size//2, kernel_size//2))/float(i)

        torch.cuda.synchronize()

        time_end = time.time()

        print(time_end - time_start)

        z = z + curr_z

    return z

@torch.no_grad()
def error(res1, res2):
    return torch.linalg.norm(res1-res2) / (torch.linalg.norm(res2))

time_1 = time.time()
#print(conv_filter_n.shape, curr_z.shape)
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    res = emv_naive_conv(conv_filter_n, curr_z, 10, kernel_size)
print(prof)
#print("---------")
time_2 = time.time()
#res = emv_arnoldi_conv(conv_filter_n, curr_z, 5, kernel_size, 10)
time_3 = time.time()
torch.cuda.synchronize()
#print(time_3 - time_2, time_2 - time_1)

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
