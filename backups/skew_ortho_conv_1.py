def basis_arnoldi_conv(L, X, m, kernel_size):
    '''
    X - image tensoк B x M x H x W
    L - filter M x M x Kh x Kw
    m - number of iters
    return list of v and h for every element in batch B
    '''

    h = torch.zeros([X.shape[0], m+1, m+1]).cuda().half()
    norms = torch.linalg.norm(torch.linalg.norm(X, dim=(-1,-2)), dim=-1)
    
    #if torch.any(norms == 0):
    #    print("HERE ZERO NORM!!!")


    #v = torch.einsum('bijk,b->bijk', X, 1/norms)[None,:,:,:,:].cuda()
    v = (X * torch.reciprocal(norms.view(norms.shape[0], 1, 1, 1)))[None,:,:,:,:].cuda()
    for j in range(m):
        w  = F.conv2d(v[j], L, padding=(kernel_size//2, kernel_size//2))

        for i in range(j+1):
            #h[:,i,j] = torch.einsum("bijk,bijk->b", w, v[i]).half()
            h[:,i,j] = torch.sum(w * v[i], axis=(1,2,3))
            #w = w - torch.einsum("b,bijk->bijk", h[:,i,j].clone().detach(), v[i]).half()
            w = w - h[:,i,j].view(h.shape[0], 1, 1, 1).clone().detach() * v[i]


        h[:,j+1,j] = torch.linalg.norm(torch.linalg.norm(w, dim=-1), dim=(-1,-2))
        
        if torch.any(h[:,j+1,j] == 0):
            h = h[:,:j+1,:j+1]
            break
        #new_v = torch.einsum("bijk,b->bijk", w, 1/h[:,j+1,j]).cuda()

        new_v = w * torch.reciprocal(h[:,j+1,j]).clone().detach().view(h.shape[0], 1, 1, 1)

        #new_v = w * torch.reciprocal(h[:,j+1,j]).view(h.shape[0], 1, 1, 1)


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
        #result = vec + torch.einsum("bij,bj->bi", A, result) / coef
        result = vec + (A * result.unsqueeze(1)).sum(dim=2) / coef
    return result

def emv_arnoldi_conv(L, X, m, kernel_size, iter):
    v, h = SOC.basis_arnoldi_conv(L, X, m, kernel_size)
    beta = torch.linalg.norm(torch.linalg.norm(X, dim=(-1,-2)), dim=-1)
    v = torch.permute(v, (1,0,2,3,4))

    vec = torch.zeros((h.shape[0], h.shape[1])).cuda()
    vec[:,0] = 1
    vec = vec.half()

    exp = SOC.emv_naive_batch(h, vec, iter)

    #ans = torch.einsum("blijk,bl,b->bijk", v, exp, beta)
    ans = (v * exp.view(exp.shape[0], exp.shape[1], 1, 1, 1)).sum(dim=1) * beta.view(beta.shape[0], 1, 1, 1)
    return ans