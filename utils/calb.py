import torch


def calculate_C(B):
    U,s,V = torch.linalg.svd(B,full_matrices=False)
    # print(U.shape) # n*r
    # print(V.shape) # r*r
    rankB = torch.linalg.matrix_rank(B)
    print(rankB)
    if rankB==B.shape[1]:
        return U @ V.t() # n*64
    elif rankB<B.shape[1]:
        # print(U.shape) # n*r
        # print(V.shape) # r*r
        # print(rankB)
        n = B.shape[0]
        I = torch.eye(n)
        QU,_ = torch.linalg.qr(I-U @ U.t()) # QU n*n
        QU = QU[:,range(B.shape[1]-rankB)]
        # print(QU.shape)
        I = torch.eye(B.shape[1])
        QV,_ = torch.linalg.qr(I-V @ V.t()) # QV r*r
        QV = QV[:,range(B.shape[1]-rankB)]
        # print(QV.shape)
        return torch.hstack(U,QU) @ torch.hstack(V,QV).t() # n*(r+r') x r*(r+r').T
    else:
        return None