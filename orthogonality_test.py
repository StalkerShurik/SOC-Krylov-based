import torch.nn.functional as F
import torch.nn as nn
import torch

from soc_arnoldi import emv_arnoldi_conv, emv_lanczos_conv

torch.no_grad()
def transpose_filter(conv_filter):
    conv_filter_T = torch.transpose(conv_filter, 0, 1)
    conv_filter_T = torch.flip(conv_filter_T, [2, 3])
    return conv_filter_T

torch.no_grad()
def zero_percent_in_tensor(z):
    return (z == 0).sum().item() / z.numel()

torch.no_grad()
def compare_with_true(conv_filter_n, curr_z, kernel_size, candidate):
        conv_filter_n = conv_filter_n.clone().detach()
        curr_z = curr_z.clone().detach()

        true = curr_z
        
        for i in range(1, 50+1):
            curr_z = F.conv2d(curr_z, conv_filter_n, 
                              padding=(kernel_size//2, 
                                       kernel_size//2))/float(i)
            true = true + curr_z
        
        return torch.linalg.norm(candidate - true) / torch.linalg.norm(true)
     
torch.no_grad()
def check_convolution_orthogonality_naive(conv_filter_n, curr_z, num_terms, kernel_size, initial_z):
        
        conv_filter_n = conv_filter_n.clone().detach()
        curr_z = curr_z.clone().detach()

        #because filter is skew-symmetric
        conv_filter_n = -conv_filter_n#transpose_filter(conv_filter_n)

        z = curr_z

        for i in range(1, num_terms+1):
            curr_z = F.conv2d(curr_z, conv_filter_n, 
                              padding=(kernel_size//2, 
                                       kernel_size//2))/float(i)
            z = z + curr_z

        return (torch.linalg.norm(z - initial_z)/torch.linalg.norm(initial_z))

torch.no_grad()
def check_convolution_orthogonality_arnoldi(conv_filter_n, curr_z, basis_size, exp_terms, kernel_size, initial_z, ort):
     
        conv_filter_n = conv_filter_n.clone().detach()
        curr_z = curr_z.clone().detach()

        #because filter is skew-symmetric
        conv_filter_n = -conv_filter_n#transpose_filter(conv_filter_n)

        z = emv_arnoldi_conv(conv_filter_n, curr_z, basis_size, kernel_size, exp_terms, ort)
        
        return (torch.linalg.norm(z - initial_z)/torch.linalg.norm(initial_z))

torch.no_grad()
def check_convolution_orthogonality_lanczos(conv_filter_n, curr_z, basis_size, exp_terms, kernel_size, initial_z):
     
        conv_filter_n = conv_filter_n.clone().detach()
        curr_z = curr_z.clone().detach()

        #because filter is skew-symmetric
        conv_filter_n = -conv_filter_n#transpose_filter(conv_filter_n)

        z = emv_lanczos_conv(conv_filter_n, curr_z, basis_size, kernel_size, exp_terms)
        
        return (torch.linalg.norm(z - initial_z)/torch.linalg.norm(initial_z))

torch.no_grad()
def hatchinson_test_naive(conv_filter_n, curr_z, num_terms, kernel_size):
     
    N_g_vectors = 5
    summ_g = 0

    conv_filter_n = conv_filter_n.float()

    conv_filter_n_T =  -conv_filter_n

    for i in range(N_g_vectors):
        #g = torch.randn(curr_z.shape).half().cuda()
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
def hatchinson_test_arnoldi(conv_filter_n, curr_z, basis_size, exp_terms, kernel_size, ort):

    #g^T (AA^T-I)^T (AA^T-I) g ~ tr((AA^T-I)^T (AA^T-I)) == |AA^T - I|**2
    #AA^T = -A^2 = A^TA

    conv_filter_n = conv_filter_n.float()

    conv_filter_n_T =  -conv_filter_n
    
    N_g_vectors = 5
    summ_g = 0

    for i in range(N_g_vectors):
        #g = torch.randn(curr_z.shape).half().cuda()
        g = torch.randn(curr_z.shape).float().cuda()

        #A^Tg

        tmp1 = emv_arnoldi_conv(conv_filter_n, g, basis_size, kernel_size, exp_terms, ort)

        # (AA^T-I)g

        tmp2 = (emv_arnoldi_conv(conv_filter_n_T, tmp1, basis_size, kernel_size, exp_terms, ort) - g)

        summ_g += torch.norm(tmp2)**2

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