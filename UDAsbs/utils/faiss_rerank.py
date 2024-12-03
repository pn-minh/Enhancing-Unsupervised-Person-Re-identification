#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import os, sys
import time
import numpy as np
from scipy.spatial.distance import cdist
import gc
import faiss

import torch
import torch.nn.functional as F

from .faiss_utils import search_index_pytorch, search_raw_array_pytorch, \
                            index_init_gpu, index_init_cpu

#def k_reciprocal_neigh(initial_rank, i, k1):
#     forward_k_neigh_index = initial_rank[i,:k1+1]
#     backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
#     fi = np.where(backward_k_neigh_index==i)[0]
#     return forward_k_neigh_index[fi]

# def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False):
#     end = time.time()
#     if print_flag:
#         print('Computing jaccard distance...')

#     ngpus = faiss.get_num_gpus()

#     N = target_features.size(0)
#     mat_type = np.float16 if use_float16 else np.float32

#     if (search_option==0):
#         # GPU + PyTorch CUDA Tensors (1)
#         res = faiss.StandardGpuResources()
#         res.setDefaultNullStreamAllDevices()
#         _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
#         initial_rank = initial_rank.cpu().numpy()
#     elif (search_option==1):
#         # GPU + PyTorch CUDA Tensors (2)
#         res = faiss.StandardGpuResources()
#         index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
#         index.add(target_features.cpu().numpy())
#         _, initial_rank = search_index_pytorch(index, target_features, k1)
#         res.syncDefaultStreamCurrentDevice()
#         initial_rank = initial_rank.cpu().numpy()
#     elif (search_option==2):
#         # GPU
#         index = index_init_gpu(ngpus, target_features.size(-1))
#         index.add(target_features.cpu().numpy())
#         _, initial_rank = index.search(target_features.cpu().numpy(), k1)
#     else:
#         # CPU
#         index = index_init_cpu(target_features.size(-1))
#         index.add(target_features.cpu().numpy())
#         _, initial_rank = index.search(target_features.cpu().numpy(), k1)


#     nn_k1 = []
#     nn_k1_half = []
#     for i in range(N):
#         nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
#         nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

#     V = np.zeros((N, N), dtype=mat_type)
#     for i in range(N):
#         k_reciprocal_index = nn_k1[i]
#         k_reciprocal_expansion_index = k_reciprocal_index
#         for candidate in k_reciprocal_index:
#             candidate_k_reciprocal_index = nn_k1_half[candidate]
#             if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
#                 k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

#         k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
#         dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
#         if use_float16:
#             V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
#         else:
#             V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

#     del nn_k1, nn_k1_half

#     if k2 != 1:
#         V_qe = np.zeros_like(V, dtype=mat_type)
#         for i in range(N):
#             V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
#         V = V_qe
#         del V_qe

#     del initial_rank

#     invIndex = []
#     for i in range(N):
#         invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

#     jaccard_dist = np.zeros((N, N), dtype=mat_type)
#     for i in range(N):
#         temp_min = np.zeros((1,N), dtype=mat_type)
#         # temp_max = np.zeros((1,N), dtype=mat_type)
#         indNonZero = np.where(V[i,:] != 0)[0]
#         indImages = []
#         indImages = [invIndex[ind] for ind in indNonZero]
#         for j in range(len(indNonZero)):
#             temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
#             # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

#         jaccard_dist[i] = 1-temp_min/(2-temp_min)
#         # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

#     del invIndex, V

#     pos_bool = (jaccard_dist < 0)
#     jaccard_dist[pos_bool] = 0.0
#     if print_flag:
#         print ("Jaccard distance computing time cost: {}".format(time.time()-end))

#     return jaccard_dist



import numpy as np
import torch
import time
import torch.nn.functional as F

def intersect1d_gpu(t1, t2):
  """
  Tìm giao của hai tensor 1 chiều trên GPU.

  Args:
    t1: Tensor 1 chiều.
    t2: Tensor 1 chiều.

  Returns:
    Tensor 1 chiều chứa các phần tử chung của t1 và t2.
  """
  combined = torch.cat((t1, t2))
  uniques, counts = torch.unique(combined, return_counts=True)
  intersection = uniques[counts > 1]
  return intersection

def k_reciprocal_neigh(initial_rank, i, k1):
    # Đảm bảo initial_rank nằm trên GPU
    initial_rank = initial_rank.cuda()  
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = torch.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')
    N = target_features.size(0)

    # Chuyển target_features sang GPU
    target_features = target_features.cuda() 

    # Tính khoảng cách Euclidean bằng PyTorch trên GPU
    distances = torch.cdist(target_features, target_features)

    # Tìm kiếm hàng xóm gần nhất bằng PyTorch trên GPU
    _, initial_rank = torch.topk(distances, k=k1+1, dim=1, largest=False)

    nn_k1 = []
    nn_k1_half = []
    for i in range(N): 
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1).cpu().numpy()) # Chuyển kết quả về CPU để sử dụng với np.append

        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))).cpu().numpy())  # Sử dụng np.around, chuyển kết quả về CPU

    V = torch.zeros((N, N), dtype=torch.float32, device=target_features.device)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = torch.tensor(k_reciprocal_index).cuda() # Chuyển sang tensor và đưa lên GPU
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            # Đảm bảo candidate_k_reciprocal_index và k_reciprocal_index là tensor trên GPU
            if (len(intersect1d_gpu(torch.tensor(candidate_k_reciprocal_index).cuda(), torch.tensor(k_reciprocal_index).cuda())) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = torch.cat([k_reciprocal_expansion_index, torch.tensor(candidate_k_reciprocal_index).cuda()])

        k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1)

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = torch.zeros_like(V, dtype=torch.float32, device=target_features.device)
        for i in range(N):
            V_qe[i,:] = torch.mean(V[initial_rank[i,:k2],:], dim=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(torch.where(V[:,i] != 0)[0])

    jaccard_dist = torch.zeros((N, N), dtype=torch.float32, device=target_features.device)
    for i in range(N):
        temp_min = torch.zeros((1,N), dtype=torch.float32, device=target_features.device)
        indNonZero = torch.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+torch.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)

    del invIndex, V

    pos_bool = (jaccard_dist < 0) 

    jaccard_dist[pos_bool] = 0.0
    
    # Chuyển đổi sang mảng NumPy trước khi trả về
    jaccard_dist_np = np.zeros_like( jaccard_dist.cpu().numpy())
    batch_size = 500  # Điều chỉnh kích thước batch tùy thuộc vào bộ nhớ khả dụng
    for i in range(0, len( jaccard_dist), batch_size):
         jaccard_dist_np[i:i+batch_size] =  jaccard_dist[i:i+batch_size].cpu().numpy()
    jaccard_dist =  jaccard_dist_np
    torch.cuda.empty_cache()  
    if print_flag:
        print ("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist