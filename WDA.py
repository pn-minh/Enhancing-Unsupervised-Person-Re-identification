from __future__ import print_function, absolute_import
import sys
import os
import os.path as osp
from datetime import datetime
import argparse
import os
from tqdm import tqdm
from UDAsbs.evaluation_metrics import cmc
import random
from sklearn.metrics import average_precision_score
import hdbscan
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
from UDAsbs.utils.faiss_rerank import compute_jaccard_distance
from torch.backends import cudnn
from UDAsbs import datasets
import argparse
import os.path as osp
import random
import numpy as np
import sys
import torch
import torchvision.models as models
import hdbscan
from sklearn import metrics
import collections
import time
import torch
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F
from UDAsbs import datasets
from UDAsbs import models
from UDAsbs.utils.serialization import load_checkpoint  # , copy_state_dict
import hdbscan  # Nhớ import thư viện hdbscan
from sbs_traindbscan import *
from sklearn.linear_model import LinearRegression


def get_jpg_files(root_folder):
    images = []
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".jpg"):
                # Add the full path of the .jpg file to the list
                path = os.path.join(root, file)
                parts = path.split('/')[-1].split('_')
                person_id = int(parts[0])         # '0001' - ID of the person
                # Number between 'c' and 's' in 'c1s1'
                camera_number = int(parts[1][1])
                images.append((path, person_id, camera_number))
    return images
# Cbi buoc 1


def create_model2(args, ncs, wopre=False):
    """
    Tạo và khởi tạo model và model_ema từ các checkpoint.

    Args:
        args: Các tham số của chương trình (bao gồm đường dẫn đến checkpoint của model và model_ema).
        ncs: Danh sách số lượng lớp cho mỗi domain.
        wopre: Tham số không sử dụng.

    Returns:
        model: Mô hình chính.
        model_ema: Mô hình EMA.
    """

    # Tạo mô hình
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=ncs)
    model_ema = models.create(
        args.arch, num_features=args.features, dropout=args.dropout, num_classes=ncs)
    model1 = models.create(args.arch, num_features=args.features,
                           dropout=args.dropout, num_classes=ncs)
    # Nạp trọng số từ checkpoint
    if args.init_1:
        initial_weights = load_checkpoint(args.init_1)
        copy_state_dict(initial_weights['state_dict'], model)
        print('load pretrain model:{}'.format(args.init_1))
    if args.init_2:
        initial_weights_ema = load_checkpoint(args.init_2)
        copy_state_dict(initial_weights_ema['state_dict'], model_ema)
        print('load pretrain model_ema:{}'.format(args.init_2))
    if args.init_3:
        initial_weights_ema = load_checkpoint(args.init_3)
        copy_state_dict(initial_weights_ema['state_dict'], model1)
        print('load pretrain model1:{}'.format(args.init_3))
    # Chuyển mô hình lên GPU
    model.cuda()
    model_ema.cuda()
    model1.cuda()

    # Sử dụng DataParallel
    model = nn.DataParallel(model)
    model_ema = nn.DataParallel(model_ema)
    model1 = nn.DataParallel(model1)

    return model, model_ema, model1


def hierarchical_clustering(pl, rerank_dist, target_features, args):
    """
    Phân cụm lại các cụm không chắc chắn bằng thuật toán DBSCAN.

    Args:
        target_features: Đặc trưng của các mẫu.
        pseudo_labels: Nhãn giả của các mẫu.
        rerank_dist: Ma trận khoảng cách giữa các mẫu.
        max_class: Nhãn lớp lớn nhất hiện tại.
        args: Các tham số của chương trình.
        sc_score_sample: Silhouette score của các mẫu.

    Returns:
        pseudo_labels: Nhãn giả sau khi phân cụm lại.
    """
    HC_start_time = time.time()
    max_class = pl.max()
    negative_one_index = (pl == -1).nonzero()
    sc_pseudo_labels = np.delete(pl, negative_one_index)
    dist = np.delete(rerank_dist, negative_one_index, 0)
    dist = np.delete(dist, negative_one_index, 1)
    sc_score_sample = metrics.silhouette_samples(
        dist, sc_pseudo_labels, metric='precomputed')
    uncer_id = get_uncer_id_by_sc_score(sc_pseudo_labels, sc_score_sample)
    pseudo_labels = reCluster(
        target_features, pl, uncer_id, max_class, args, sc_score_sample)
    num_outliers = sum(pseudo_labels == -1)
    num_cluster_label = len(set(pseudo_labels)) - \
        1 if num_outliers != 0 else len(set(pseudo_labels))
    print(f'HC finish! Cost time={time.time() - HC_start_time}s')
    return pseudo_labels, num_outliers, num_cluster_label


def apply_UCIS(pseudo_labels, pseudo_labels_ema, dataset_target, num_cluster_label, num_cluster_label_ema, args):
    """
    Apply uncertainty-aware collaborative instance selection.
    """
    print('Applying uncertainty-aware collaborative instance selection')
    UCIS_start_time = time.time()
    pseudo_labels_noNeg1, outlier_oneHot = generate_pseudo_labels(
        pseudo_labels, num_cluster_label, dataset_target)
    pseudo_labels_noNeg1_ema, outlier_oneHot_ema = generate_pseudo_labels(
        pseudo_labels_ema, num_cluster_label_ema, dataset_target)
    N = pseudo_labels_noNeg1.size(0)
    label_sim = pseudo_labels_noNeg1.expand(N, N).eq(
        pseudo_labels_noNeg1.expand(N, N).t()).float()
    label_sim_ema = pseudo_labels_noNeg1_ema.expand(N, N).eq(
        pseudo_labels_noNeg1_ema.expand(N, N).t()).float()
    label_sim_new = label_sim - outlier_oneHot
    label_sim_new_ema = label_sim_ema - outlier_oneHot_ema
    label_share = torch.min(label_sim_new, label_sim_new_ema)
    uncer = label_share.sum(-1) / label_sim.sum(-1)
    index_zero = torch.le(uncer, 0.8).type(torch.uint8) * \
        torch.gt(label_sim.sum(-1), 1).type(torch.uint8)
    index_zero = torch.nonzero(index_zero)
    pseudo_labels[index_zero] = -1
    num_noisy_outliers = sum(pseudo_labels == -1)
    num_clean_cluster_label = len(set(
        pseudo_labels)) - 1 if sum(pseudo_labels == -1) != 0 else len(set(pseudo_labels))
    print(f'UCIS finish! Cost time={time.time() - UCIS_start_time}s')
    return pseudo_labels, num_clean_cluster_label, num_noisy_outliers


def cosine_similarity(tensor1, tensor2):
    # Ensure the tensors are flattened if they're not 1-dimensional
    tensor1 = tensor1.view(-1)
    tensor2 = tensor2.view(-1)
    # Calculate the cosine similarity using torch's functional API
    cos_sim = F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0))
    return cos_sim.item()


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    # with torch.no_grad():
    inputs = to_torch(inputs).cpu()
    if modules is None:
        outputs = model(inputs)
        outputs = outputs.data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())


def k_nearest_neighbors(similarity_matrix, k):

    # Ensure similarity_matrix is a square matrix
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "Matrix must be square"

    # Create a tensor to hold the indices of the k-nearest neighbors
    neighbors = torch.zeros((similarity_matrix.shape[0], k), dtype=torch.long)

    # For each row (instance) in the similarity matrix
    for i in range(similarity_matrix.shape[0]):
        # Get the similarity scores for instance i, and exclude self-similarity
        similarities = similarity_matrix[i]
        # Set self-similarity to -inf to exclude it
        similarities[i] = -float('inf')

        # Get the indices of the k largest similarity values (k-nearest neighbors)
        _, indices = torch.topk(similarities, k)

        # Store the indices of the k-nearest neighbors
        neighbors[i] = indices

    return neighbors


def refined_features(features, sim_matrix, neighbors):
    refined_feats = torch.zeros(len(features), len(features[0]))

    for i in range(len(features)):
        for j in range(len(neighbors[i])):
            weight = sim_matrix[i][neighbors[i][j]] / sum(neighbors[i])
            refined_feats[i] += weight * features[neighbors[i][j].item()]

    return refined_feats


class Dataset(Dataset):
    def __init__(self, records, transform=None):
        """
        Args:
            records (list): List of tuples containing (image_path, label, cam_angle).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image_path, label, cam_angle = self.records[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Convert label and cam_angle to tensors
        label = torch.tensor(label, dtype=torch.long)
        cam_angle = torch.tensor(cam_angle, dtype=torch.long)

        return image, label, cam_angle


class TripletDataset(Dataset):
    def __init__(self, dataset):
        """
        Initialize the dataset with records containing transformed images, labels, and camera angles.

        Args:
            dataset (Dataset): Original dataset with images, labels, and camera information.
        """
        self.dataset = dataset
        self.label_to_indices = self._create_label_dict()

    def _create_label_dict(self):
        """
        Creates a dictionary to map each label to the indices of all images with that label.
        """
        label_to_indices = {}
        for idx, (_, label, _) in enumerate(self.dataset):
            if label.item() not in label_to_indices:
                label_to_indices[label.item()] = []
            label_to_indices[label.item()].append(idx)
        return label_to_indices

    def __getitem__(self, index):
        """
        Retrieves an anchor, positive, and negative sample for triplet loss training.
        """
        # Anchor sample
        anchor_img, anchor_label, anchor_camera = self.dataset[index]

        # Positive sample (same label as anchor)
        positive_indices = self.label_to_indices[anchor_label.item()]
        # Exclude the anchor's index
        positive_indices = [i for i in positive_indices if i != index]
        if len(positive_indices) < 1:
            # If only one instance, skip this anchor by returning None or handle differently
            positive_indices.append(index)
        positive_index = random.choice(positive_indices)
        positive_img, positive_label, positive_camera = self.dataset[positive_index]

        # Negative sample (different label than anchor)
        negative_label = random.choice(
            [label for label in self.label_to_indices.keys() if label != anchor_label.item()])
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative_img, negative_label, negative_camera = self.dataset[negative_index]

        return (index, positive_index, negative_index), (anchor_label, positive_label, negative_label), (anchor_camera, positive_camera, negative_camera)

    def __len__(self):
        return len(self.dataset)

# Buoc1


def prepare_inf(model, tar_cluster_loader, args, cluster):
    target_features = get_features(model, tar_cluster_loader)
    rerank_dist = compute_jaccard_distance(
        target_features, k1=args.k1, k2=args.k2)
    torch.cuda.empty_cache()
  # Xóa tensor không cần thiết
    n_samples = rerank_dist.shape[0]
    batch_size = 500  # Chọn kích thước batch phù hợp
    # Khởi tạo nhãn là -1 (chưa gán cụm)
    pseudo_labels = np.full(n_samples, -1)
    for i in range(0, n_samples, batch_size):
        start = i
        end = min(i + batch_size, n_samples)
        batch_distance_matrix = rerank_dist[start:end, start:end]
        # Chuyển đổi sang double
        batch_distance_matrix = batch_distance_matrix.astype('double')

        batch_labels = cluster.fit_predict(batch_distance_matrix)
        pseudo_labels[start:end] = batch_labels
    # pseudo_labels = cluster.fit_predict(rerank_dist)#numbel label
    pl = pseudo_labels.copy()
    num_outliers = sum(pseudo_labels == -1)
    num_cluster_label = len(set(pl))-1 if num_outliers != 0 else len(set(pl))
    return target_features, rerank_dist, pseudo_labels, num_outliers, num_cluster_label

# Buoc giua


class Dataset(Dataset):

    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        image_path, label, cam_angle = self.records[idx]
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations if any

        if self.transform:

            image = self.transform(image)

        # Convert label and cam_angle to tensors

        label = torch.tensor(label, dtype=torch.long)

        cam_angle = torch.tensor(cam_angle, dtype=torch.long)

        return image, label, cam_angle


def to_numpy(tensor):
    return tensor.cpu().numpy() if torch.is_tensor(tensor) else tensor


def to_torch(ndarray):
    return torch.from_numpy(ndarray) if isinstance(ndarray, np.ndarray) else ndarray


def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    with torch.no_grad():
        inputs = to_torch(inputs).to('cuda')  # Ensure inputs are on GPU
        if modules is None:
            outputs = model(inputs)
            outputs = outputs.data.cpu()  # Move to CPU after processing
            return outputs


def cosine_similarity(tensor1, tensor2):

    # Ensure the tensors are flattened if they're not 1-dimensional

    tensor1 = tensor1.view(-1)

    tensor2 = tensor2.view(-1)

    # Calculate the cosine similarity using torch's functional API
    cos_sim = F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0))
    return cos_sim.item()


def k_nearest_neighbors(similarity_matrix, k):
    # Ensure similarity_matrix is a square matrix
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "Matrix must be square"
    # Create a tensor to hold the indices of the k-nearest neighbors
    neighbors = torch.zeros((similarity_matrix.shape[0], k), dtype=torch.long)
    # For each row (instance) in the similarity matrix
    for i in range(similarity_matrix.shape[0]):

        # Get the similarity scores for instance i, and exclude self-similarity
        similarities = similarity_matrix[i]

        # Set self-similarity to -inf to exclude it
        similarities[i] = -float('inf')
        # Get the indices of the k largest similarity values (k-nearest neighbors)

        _, indices = torch.topk(similarities, k)
        # Store the indices of the k-nearest neighbors

        neighbors[i] = indices
    return neighbors


def cce_sim(cam_1, cam_2):

    return 1 if cam_1 == cam_2 else 0


def extract_and_save_features(model, dataloader, save_path):
    all_features = []
    all_labels = []
    all_cam_angles = []

    with torch.no_grad():
        for images, labels, cam_angles in tqdm(dataloader):
            images = images.to('cuda')
            # Extract features
            features = model(images)
            # Move to CPU and convert to numpy
            features = features.cpu().numpy()
            labels = labels.cpu().numpy()
            cam_angles = cam_angles.cpu().numpy()

            # Append to lists
            all_features.append(features)
            all_labels.append(labels)
            all_cam_angles.append(cam_angles)

    # Stack features and labels
    all_features = np.vstack(all_features)
    all_labels = np.hstack(all_labels)
    all_cam_angles = np.hstack(all_cam_angles)

    # Save features and labels as numpy arrays
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.save(save_path, {'features': all_features,
            'labels': all_labels, 'cam_angles': all_cam_angles})


class TripletDataset(Dataset):

    def __init__(self, dataset):
        """

        Initialize the dataset with records containing transformed images, labels, and camera angles.



        Args:

            dataset (Dataset): Original dataset with images, labels, and camera information.

        """

        self.dataset = dataset

        self.label_to_indices = self._create_label_dict()

    def _create_label_dict(self):
        """

        Creates a dictionary to map each label to the indices of all images with that label.

        """

        label_to_indices = {}

        for idx, (_, label, _) in enumerate(self.dataset):

            if label.item() not in label_to_indices:

                label_to_indices[label.item()] = []

            label_to_indices[label.item()].append(idx)

        return label_to_indices

    def __getitem__(self, index):
        """

        Retrieves an anchor, positive, and negative sample for triplet loss training.

        """

        # Anchor sample

        anchor_img, anchor_label, anchor_camera = self.dataset[index]

        # Positive sample (same label as anchor)

        positive_indices = self.label_to_indices[anchor_label.item()]

        # Exclude the anchor's index
        positive_indices = [i for i in positive_indices if i != index]

        if len(positive_indices) < 1:

            # If only one instance, skip this anchor by returning None or handle differently

            positive_indices.append(index)

        positive_index = random.choice(positive_indices)

        positive_img, positive_label, positive_camera = self.dataset[positive_index]

        # Negative sample (different label than anchor)

        negative_label = random.choice(
            [label for label in self.label_to_indices.keys() if label != anchor_label.item()])

        negative_index = random.choice(self.label_to_indices[negative_label])

        negative_img, negative_label, negative_camera = self.dataset[negative_index]

        return (index, positive_index, negative_index), (anchor_label, positive_label, negative_label), (anchor_camera, positive_camera, negative_camera)

    def __len__(self):

        return len(self.dataset)


def refined_features(feature_dataset, sim_matrix, neighbors):

    refined_feats = torch.zeros(
        len(feature_dataset), len(feature_dataset[0][0]))

    for i in range(len(feature_dataset)):

        for j in neighbors[i]:

            weight = sim_matrix[i][j.item()] / sum(neighbors[i])

            refined_feats[i] += weight * feature_dataset[j.item()][0]

    return refined_feats
# Buoc 2


class FeatureDataset(Dataset):
    def __init__(self, features_file):
        """
        Args:
            features_file (str): Path to the .npy file containing the features and labels.
        """
        # Load the saved features and labels
        data = np.load(features_file, allow_pickle=True).item()
        self.features = data['features']
        self.labels = data['labels']
        self.cam_angles = data['cam_angles']

    def __len__(self):
        # Return the number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Get feature and label at the specified index
        feature = self.features[idx]
        label = self.labels[idx]
        cam_angle = self.cam_angles[idx]

        # Convert to torch tensors
        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        cam_angle = torch.tensor(cam_angle, dtype=torch.long)

        return feature, label, cam_angle


def compute_cosine_similarity_between_datasets(dataset1, dataset2, batch_size=256):
    # Create DataLoaders for batch processing
    dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False)
    dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)

    # Accumulate features from both datasets
    features1 = []
    features2 = []

    for batch in dataloader1:
        features1.append(batch[0])  # batch[0] contains features
    for batch in dataloader2:
        features2.append(batch[0])

    # Stack all batches to form full feature tensors
    features1 = torch.cat(features1, dim=0)  # Shape (N, D)
    features2 = torch.cat(features2, dim=0)  # Shape (M, D)

    # Normalize features to unit length
    features1 = features1 / features1.norm(dim=1, keepdim=True)
    features2 = features2 / features2.norm(dim=1, keepdim=True)

    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(features1, features2.T)  # Shape (N, M)
    return similarity_matrix

# def compute_cosine_similarity_between_datasets(dataset1, dataset2, batch_size):
#     """
#     Tính toán ma trận tương đồng cosine giữa hai tập dữ liệu.

#     Args:
#       dataset1: Tập dữ liệu đầu tiên.
#       dataset2: Tập dữ liệu thứ hai.
#       batch_size: Kích thước batch.

#     Returns:
#       Ma trận tương đồng cosine.
#     """

#     # Tạo DataLoader cho xử lý theo batch
#     dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False)
#     print(23423)
#     dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)
#     print(22343423)

#     # Khởi tạo ma trận tương đồng 
#     similarity_matrix = torch.zeros((len(dataset1), len(dataset2)))
#     print(2342323)

#     # Tính toán tương đồng theo từng batch
#     with torch.no_grad():  # Vô hiệu hóa tính toán gradient để tiết kiệm bộ nhớ
#         for i, batch1 in enumerate(dataloader1):
#             features1 = batch1[0]  # batch[0] chứa các đặc trưng
#             features1 = features1 / features1.norm(dim=1, keepdim=True)  # Chuẩn hóa

#             for j, batch2 in enumerate(dataloader2):
#                 features2 = batch2[0]
#                 features2 = features2 / features2.norm(dim=1, keepdim=True)

#                 # Tính toán tương đồng cosine và lưu vào ma trận
#                 similarity_matrix[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size] = torch.mm(features1, features2.T)

#     return similarity_matrix

class TensorRowDataset(Dataset):
    def __init__(self, data_tensor):
        """
        Args:
            data_tensor (torch.Tensor): Tensor of shape (19000, 1000).
        """
        self.data_tensor = data_tensor
        self.filler = 1

    def __len__(self):
        # Return the number of rows
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        # Return the idx-th row as a tensor of size (1000,)
        return self.data_tensor[idx], self.filler


def create_dataset(records, transform, model1, k):
    dataset = Dataset(records, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32,
                             shuffle=True, num_workers=4)

    for i, item in enumerate(data_loader):
        imgs, labels = item[0], item[1]
        outputs = extract_cnn_feature(model1, imgs)
        break
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    features = []
    for i, item in enumerate(data_loader):
        imgs, labels = item[0], item[1]
        outputs = extract_cnn_feature(model1, imgs)
        features = outputs
        tensor1 = outputs[0]
        tensor2 = outputs[1]
        similarity = cosine_similarity(tensor1, tensor2)
        print(f"Cosine Similarity: {similarity}")
        break
    sim_matrix = torch.zeros(len(features), len(features))
    for i in range(len(features)):
        for j in range(i, len(features)):
            sim_matrix[i][j] = cosine_similarity(features[i], features[j])

            sim_matrix[j][i] = sim_matrix[i][j]
    # neighbors = k_nearest_neighbors(sim_matrix, k)
    sorted_data = sorted([(img, label, cam)
                         for img, label, cam in dataset], key=lambda x: x[1])
    dataset = sorted_data
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    extract_and_save_features(
        model1, dataloader, save_path='/hgst/longdn/UCF-main/logs/WDA/duke2msmt/ExtractData/features.npy')
    data = np.load(
        '/hgst/longdn/UCF-main/logs/WDA/duke2msmt/ExtractData/features.npy', allow_pickle=True).item()
    
    features, labels, cam_angles = data['features'], data['labels'], data['cam_angles']

    feature_dataset = FeatureDataset(
        '/hgst/longdn/UCF-main/logs/WDA/duke2msmt/ExtractData/features.npy')
    
    triplet_dataset = TripletDataset(feature_dataset)
    return feature_dataset, triplet_dataset


def cal_weights_bias(feature_dataset, triplet_dataset, k):
    # Now you can use this dataset with a DataLoader for further processing or model training
    dataset1 = feature_dataset
    dataset2 = feature_dataset
    sim_matrix = compute_cosine_similarity_between_datasets(
        dataset1, dataset2, batch_size=256)    
    neighbors = k_nearest_neighbors(sim_matrix, k)
    refined_feats = refined_features(
        feature_dataset=feature_dataset, sim_matrix=sim_matrix, neighbors=neighbors)
    
    positive_instances = []
    negative_instances = []
    for i, item in enumerate(triplet_dataset):

        # print(item)

        anchor_index = item[0][0]

        pos_index = item[0][1]

        neg_index = item[0][2]

        # print(anchor_index, pos_index, neg_index)

        anchor_feat = feature_dataset[anchor_index][0]

        pos_sim = sim_matrix[anchor_index, pos_index].item()

        neg_sim = sim_matrix[anchor_index, neg_index].item()

        # print(pos_sim, neg_sim)

        pos_urf = refined_feats[pos_index]

        neg_urf = refined_feats[neg_index]

        pos_urf_sim = cosine_similarity(anchor_feat, pos_urf)

        neg_urf_sim = cosine_similarity(anchor_feat, neg_urf)

        # print(pos_urf_sim, neg_urf_sim)

        pos_cce = float(item[2][0].item() == item[2][1].item())

        neg_cce = float(item[2][0].item() == item[2][2].item())

        # print(pos_cce, neg_cce)

        positive_instances.append((pos_sim, pos_urf_sim, pos_cce, 1))

        negative_instances.append((neg_sim, neg_urf_sim, neg_cce, -1))

        # positive_instances.append((pos_sim, pos_urf_sim, 1))

        # negative_instances.append((neg_sim, neg_urf_sim, -1))

    positive_input = np.array(positive_instances)

    negative_input = np.array(negative_instances)

    input = np.concatenate((positive_input, negative_input), axis=0)

    # Separate inputs (x, y, z) and outputs (label)
    X3 = input[:, :3]

    y3 = input[:, 3]

    X = input[:, :2]  # Features (x, y, z)
    y = input[:, 2]   # Labels (1 or -1)
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    lin_model3 = LinearRegression()
    lin_model3.fit(X3, y3)
    # Extract and print weights (coefficients) and bias (intercept)
    weights = lin_model.coef_ 
    bias = lin_model.intercept_
    weights3 = lin_model3.coef_
    bias3 = lin_model3.intercept_ 
    print("Weights:", weights)
    print("Bias:", bias)
    return weights, bias, weights3, bias3


def cal_gal_querry_sim_maxtrix(query_feature_dataset, gallery_feature_dataset, k, weights, bias, weights3, bias3):
    dataset1 = query_feature_dataset
    dataset2 = gallery_feature_dataset

    query_to_gallery_sim_matrix = compute_cosine_similarity_between_datasets(
        dataset1, dataset2, batch_size=256)
    dataset1 = gallery_feature_dataset
    dataset2 = gallery_feature_dataset

    gallery_sim_matrix = compute_cosine_similarity_between_datasets(
        dataset1, dataset2, batch_size=256)

    # gallery_dist_matrix = 1 - gallery_sim_matrix

    # dataset1 = query_feature_dataset
    # dataset2 = query_feature_dataset

    # query_sim_matrix = compute_cosine_similarity_between_datasets(dataset1, dataset2, batch_size=256)
    # query_dist_matrix = 1 - query_sim_matrix

    gallery_neighbors = k_nearest_neighbors(gallery_sim_matrix, k)
    gallery_refined_feats = refined_features(
        feature_dataset=gallery_feature_dataset, sim_matrix=gallery_sim_matrix, neighbors=gallery_neighbors)
    data_tensor = gallery_refined_feats  # Replace with your actual tensor
#    Create the dataset
    refined_feat_dataset = TensorRowDataset(data_tensor)
    dataset1 = query_feature_dataset
    dataset2 = refined_feat_dataset
    query_to_refined_gallery_sim_matrix = compute_cosine_similarity_between_datasets(
        dataset1, dataset2, batch_size=256)
    
    query_cam_angles = torch.tensor([sample[2]
                                    for sample in query_feature_dataset])
    gallery_cam_angles = torch.tensor(
        [sample[2] for sample in gallery_feature_dataset])

    # Use broadcasting to create the binary matrix

    # NoCCE
    # Use the shape of the similarity matrix
    bias_matrix = torch.full(query_to_gallery_sim_matrix.shape, bias)

    num_parts = 8 # Chia thành 4 phần
    part_size = query_to_gallery_sim_matrix.shape[0] // num_parts

    query_to_gallery_find_matrix2 = torch.zeros_like(query_to_gallery_sim_matrix)

    for i in range(num_parts):
        start = i * part_size
        end = (i + 1) * part_size
        query_to_gallery_find_matrix2[start:end] = weights[0] * query_to_gallery_sim_matrix[start:end] + \
                                                weights[1] * query_to_refined_gallery_sim_matrix[start:end] + \
                                                bias_matrix[start:end]

    # Xử lý phần còn lại nếu có
    if query_to_gallery_sim_matrix.shape[0] % num_parts != 0:
        start = num_parts * part_size
        query_to_gallery_find_matrix2[start:] = weights[0] * query_to_gallery_sim_matrix[start:] + \
                                            weights[1] * query_to_refined_gallery_sim_matrix[start:] + \
                                            bias_matrix[start:]
    # CCE
    num_parts = 4  # Chia thành 4 phần
    part_size = len(query_cam_angles) // num_parts

    query_cam_angles = query_cam_angles.cuda()
    gallery_cam_angles = gallery_cam_angles.cuda()

    num_parts = 8  # Chia thành 4 phần
    part_size = len(query_cam_angles) // num_parts

    cce_matrix = torch.zeros((0, len(gallery_cam_angles)), dtype=torch.float).cuda()  # Khởi tạo tensor rỗng trên GPU
    bias_matrix = torch.zeros((0, len(gallery_cam_angles)), dtype=torch.float).cuda()  # Khởi tạo tensor rỗng trên GPU

    for i in range(num_parts):
        start = i * part_size
        end = (i + 1) * part_size

        # Tính toán cce_matrix cho phần hiện tại
        part_cce_matrix = (query_cam_angles[start:end, None] == gallery_cam_angles[None, :]).float().cuda()

        # Tạo bias_matrix cho phần hiện tại
        part_bias_matrix = torch.full(part_cce_matrix.shape, bias3).cuda()

        # Nối phần cce_matrix và bias_matrix vừa tính vào ma trận tổng
        cce_matrix = torch.cat((cce_matrix, part_cce_matrix), dim=0)
        bias_matrix = torch.cat((bias_matrix, part_bias_matrix), dim=0)

    # Xử lý phần còn lại nếu có
    if len(query_cam_angles) % num_parts != 0:
        start = num_parts * part_size

        # Tính toán cce_matrix cho phần còn lại
        part_cce_matrix = (query_cam_angles[start:, None] == gallery_cam_angles[None, :]).float().cuda()

        # Tạo bias_matrix cho phần còn lại
        part_bias_matrix = torch.full(part_cce_matrix.shape, bias3).cuda()

        # Nối phần cce_matrix và bias_matrix vừa tính vào ma trận tổng
        cce_matrix = torch.cat((cce_matrix, part_cce_matrix), dim=0)
        bias_matrix = torch.cat((bias_matrix, part_bias_matrix), dim=0)

    # query_to_gallery_find_matrix3 = weights3[0] * query_to_gallery_sim_matrix + weights3[1] * \
    #     query_to_refined_gallery_sim_matrix + \
    #     weights3[2] * cce_matrix + bias_matrix
    torch.cuda.empty_cache()
    # query_to_gallery_find_matrix3 = torch.zeros_like(query_to_gallery_sim_matrix).cuda()

    

    # num_parts_inner = 20  # Chia thành 4 phần nhỏ hơn
    # part_size_inner = query_to_gallery_sim_matrix.shape[0] // num_parts_inner

    # for j in range(num_parts_inner):
    #     start_inner = j * part_size_inner
    #     end_inner = (j + 1) * part_size_inner
    #     query_to_gallery_find_matrix3[start_inner:end_inner] = weights3[0] * query_to_gallery_sim_matrix[start_inner:end_inner] + \
    #                                                             weights3[1] * query_to_refined_gallery_sim_matrix[start_inner:end_inner] + \
    #                                                             weights3[2] * cce_matrix[start_inner:end_inner] + \
    #                                                             bias_matrix[start_inner:end_inner]
        

    # # Xử lý phần còn lại nếu có
    # if query_to_gallery_sim_matrix.shape[0] % num_parts_inner != 0:
    #     start_inner = num_parts_inner * part_size_inner
    #     query_to_gallery_find_matrix3[start_inner:] = weights3[0] * query_to_gallery_sim_matrix[start_inner:] + \
    #                                                     weights3[1] * query_to_refined_gallery_sim_matrix[start_inner:] + \
    #                                                     weights3[2] * cce_matrix[start_inner:] + \
    #                                                     bias_matrix[start_inner:]
    # # Xóa các biến trung gian
    # torch.cuda.empty_cache()
    # print(7)
    
    # query_to_gallery_find_dist_matrix = 1 - query_to_gallery_find_matrix

    # scaler = MinMaxScaler()
    # query_to_gallery_find_dist_matrix = scaler.fit_transform(query_to_gallery_find_dist_matrix.cpu().detach().numpy())
    # query_dist_matrix = scaler.fit_transform(query_dist_matrix.cpu().detach().numpy())
    # gallery_dist_matrix = scaler.fit_transform(gallery_dist_matrix.cpu().detach().numpy())
    # reranked_dist_matrix = re_ranking(query_to_gallery_find_dist_matrix,  # Sử dụng query_to_gallery_find_dist_matrix
    #                                  query_dist_matrix,
    #                                  gallery_dist_matrix,
    #                                  k1=args.k1, k2=args.k2, lambda_value=0.3)

    # return query_to_gallery_sim_matrix, query_to_refined_gallery_sim_matrix, query_to_gallery_find_matrix2, query_to_gallery_find_matrix3
    return query_to_gallery_sim_matrix, query_to_refined_gallery_sim_matrix, query_to_gallery_find_matrix2



def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    batch_size = 1024  # Kích thước batch
    ret = np.zeros(topk)
    num_valid_queries = 0

    # Compute CMC for each query in batches
    for i in range(0, m, batch_size):
        start = i
        end = min(i + batch_size, m)

        indices = np.argsort(distmat[start:end], axis=1)
        matches = (gallery_ids[indices] == query_ids[start:end, np.newaxis])

        for j in range(end - start):
            # Filter out the same id and same camera
            valid = ((gallery_ids[indices[j]] != query_ids[start + j]) |
                     (gallery_cams[indices[j]] != query_cams[start + j]))
            if separate_camera_set:
                # Filter out samples from same camera
                valid &= (gallery_cams[indices[j]] != query_cams[start + j])
            if not np.any(matches[j, valid]):
                continue
            if single_gallery_shot:
                repeat = 10
                gids = gallery_ids[indices[j][valid]]
                inds = np.where(valid)[0]
                ids_dict = defaultdict(list)
                for k, x in zip(inds, gids):
                    ids_dict[x].append(k)
            else:
                repeat = 1
            for _ in range(repeat):
                if single_gallery_shot:
                    # Randomly choose one instance for each id
                    sampled = (valid & _unique_sample(ids_dict, len(valid)))
                    index = np.nonzero(matches[j, sampled])[0]
                else:
                    index = np.nonzero(matches[j, valid])[0]
                delta = 1. / (len(index) * repeat)
                for k, l in enumerate(index):
                    if l - k >= topk:
                        break
                    if first_match_break:
                        ret[l - k] += 1
                        break
                    ret[l - k] += delta
            num_valid_queries += 1

    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    distmat = to_numpy(distmat)

    m, n = distmat.shape

    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array

    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)

    # Sort and find correct matches
    batch_size = 1024  # Kích thước batch
    aps = []

    # Compute AP for each query in batches
    for i in range(0, m, batch_size):
        start = i
        end = min(i + batch_size, m)

        indices = np.argsort(distmat[start:end], axis=1)
        matches = (gallery_ids[indices] == query_ids[start:end, np.newaxis])

        for j in range(end - start):
            # Filter out the same id and same camera
            valid = ((gallery_ids[indices[j]] != query_ids[start + j]) |
                     (gallery_cams[indices[j]] != query_cams[start + j]))
            y_true = matches[j, valid]
            y_score = -distmat[start + j][indices[j]][valid]
            if np.any(y_true):
                aps.append(average_precision_score(y_true, y_score))

    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)


def evaluate_all(sim_matrix, args,  query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    sim_matrix = sim_matrix.cuda()

    if query is not None and gallery is not None:
        query_ids = [item[1] for item in query]
        gallery_ids = [item[1] for item in gallery]
        query_cams = [item[-1] for item in query]
        gallery_cams = [item[-1] for item in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Tính toán mAP
    mAP = mean_ap(sim_matrix, query_ids, gallery_ids, query_cams, gallery_cams)

    print('Mean AP: {:4.1%}'.format(mAP))
    torch.cuda.empty_cache()
    cmc_configs = {
        args.dataset_target: dict(separate_camera_set=False,
                                  single_gallery_shot=False,
                                  first_match_break=True)
    }
    torch.cuda.empty_cache()
    # Tính toán CMC scores
    cmc_scores = {name: cmc(sim_matrix, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
    torch.cuda.empty_cache()
    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores[args.dataset_target][k - 1]))

    if cmc_flag:
        return cmc_scores[args.dataset_target][0], mAP
    return mAP

# final
def main_worker(args):
    print("==========\nArgs:{}\n==========".format(args))
#     fc_len = 32621
#     ncs = [int(x) for x in args.ncs.split(',')]
#     model, model_ema, model1 = create_model2(
#         args, [fc_len for _ in range(len(ncs))])

#     dataset_target, label_dict, ground_label_list = get_data(
#         args.dataset_target, args.data_dir, len(ncs))

#     tar_cluster_loader = get_test_loader(
#         dataset_target, args.height, args.width, args.batch_size, args.workers, testset=dataset_target.train)

#     cluster = hdbscan.HDBSCAN(metric='precomputed')
#     target_features, rerank_dist, pseudo_labels, num_outliers, num_cluster_label = prepare_inf(
#         model, tar_cluster_loader, args, cluster)
#     torch.cuda.empty_cache()
#     target_features_ema, rerank_dist_ema, pseudo_labels_ema, num_outliers_ema, num_cluster_label_ema = prepare_inf(
#         model_ema, tar_cluster_loader, args, cluster)
#     del tar_cluster_loader
#     torch.cuda.empty_cache()
#     print(1)
#     if args.HC:
#         pseudo_labels, num_outliers, num_cluster_label = hierarchical_clustering(
#             pseudo_labels, rerank_dist, target_features, args)
#         del rerank_dist
#         pseudo_labels_ema, num_outliers_ema, num_cluster_label_ema = hierarchical_clustering(
#             pseudo_labels_ema, rerank_dist_ema, target_features_ema, args)
#         del rerank_dist_ema
#         print(
#             f'The HC Re-cluster result: num cluster = {num_cluster_label}(model) // {num_cluster_label_ema}(model_ema) \t num outliers = {num_outliers}(model) // {num_outliers_ema}(model_ema)')

#     if args.UCIS:
#         pseudo_labels, num_clean_cluster_label, num_noisy_outliers = apply_UCIS(
#             pseudo_labels, pseudo_labels_ema, dataset_target, num_cluster_label, num_cluster_label_ema, args)
#         print(
#             f'The UCIS result: num clean cluster = {num_clean_cluster_label}(model) \t num outliers = {num_noisy_outliers}(model)')

#     cl = list(set(pseudo_labels))
#     p1 = []
#     new_dataset = []
#     cluster_centers = collections.defaultdict(list)
#     for i, (item, label) in enumerate(zip(dataset_target.train, pseudo_labels)):
#         if label == -1:
#             continue
#         label = cl.index(label)
#         p1.append(label)
#         new_dataset.append((item[0], label, item[-1]))
#         cluster_centers[label].append(target_features[i])
#     cluster_features, c_labels = get_cluster_item(cluster_centers)
#     del cluster_centers, dataset_target 

#     model.module.classifier0_32621.weight.data[:len(c_labels)].copy_(
#         torch.cat(cluster_features, dim=0).float().cuda())
#     model_ema.module.classifier0_32621.weight.data[:len(c_labels)].copy_(
#         torch.cat(cluster_features, dim=0).float().cuda())
    
#     del cluster_features
#     ncs = [len(set(p1)) + 1]
#     print('mdoel:new class are {}, length of new dataset is {}'.format(
#         ncs, len(new_dataset)))
#     del target_features, target_features_ema
#     records = new_dataset

# # # #     # Buoc2
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    if (args.dataset_target == 'msmt17'):
        # Replace with the path to your folder
        gallery_folder = '/hgst/longdn/UCF-main/data/MSMT17_V1/bounding_box_test'
        # Replace with the path to your folder
        query_folder = '/hgst/longdn/UCF-main/data/MSMT17_V1/query'
    elif (args.dataset_target == 'duke'):
        gallery_folder = '/hgst/longdn/UCF-main/data/dukemtmc/bounding_box_test'
        query_folder = '/hgst/longdn/UCF-main/data/dukemtmc/query'
    elif (args.dataset_target == 'market1501'):
        gallery_folder = '/hgst/longdn/UCF-main/data/market1501/Market-1501-v15.09.15/bounding_box_test'
        query_folder = '/hgst/longdn/UCF-main/data/market1501/Market-1501-v15.09.15/query'
    gallery_jpg_files = get_jpg_files(gallery_folder)
    gallery_dataset = Dataset(gallery_jpg_files, transform=transform)
    query_jpg_files = get_jpg_files(query_folder)
    query_dataset = Dataset(query_jpg_files, transform=transform)
#     batch_size = 256
#     gallery_dataloader = DataLoader(
#         gallery_dataset, batch_size=batch_size, shuffle=False)
#     query_dataloader = DataLoader(
#         query_dataset, batch_size=batch_size, shuffle=False)
#     # Example usage
#     # extract_and_save_features(model1, gallery_dataloader, save_path='/hgst/longdn/UCF-main/logs/WDA/duke2msmt/ExtractData/gallery_features.npy')

#     # extract_and_save_features(model1, query_dataloader, save_path='/hgst/longdn/UCF-main/logs/WDA/duke2msmt/ExtractData/query_features.npy')

    gallery_feature_dataset = FeatureDataset(
        '/hgst/longdn/UCF-main/logs/WDA/duke2msmt/ExtractData/gallery_features.npy')

    query_feature_dataset = FeatureDataset(
        '/hgst/longdn/UCF-main/logs/WDA/duke2msmt/ExtractData/query_features.npy')
    del  query_jpg_files, gallery_jpg_files
    # torch.cuda.empty_cache()
#     all_weights_and_bias = {}  # Khởi tạo dictionary để lưu trữ

#     for k in range(2, 20):
#         feature_dataset, triplet_dataset = create_dataset(records, transform, model1, k) 

#         weights, bias, weights3, bias3 = cal_weights_bias(feature_dataset, triplet_dataset, k)

#         # Lưu trữ k, weights và bias vào dictionary
#         all_weights_and_bias[k] = {
#             "weights": weights, 
#             "bias": bias, 
#             "weights3": weights3, 
#             "bias3": bias3
#         }

#         del feature_dataset, triplet_dataset
#         torch.cuda.empty_cache()

#     # Lưu dictionary ra file
    save_dir = "/hgst/longdn/UCF-main/logs/WDA/duke2msmt/ExtractData"
#     os.makedirs(save_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
#     save_path = os.path.join(save_dir, "all_weights_and_bias.pth")
#     torch.save(all_weights_and_bias, save_path)

    save_path = os.path.join(save_dir, "all_weights_and_bias.pth")
    all_weights_and_bias = torch.load(save_path)
    for k in range(2, 20):

        weights = all_weights_and_bias[k]["weights"]
        bias = all_weights_and_bias[k]["bias"]
        weights3 = all_weights_and_bias[k]["weights3"]
        bias3 = all_weights_and_bias[k]["bias3"]
        # # Buoc2
        query_to_gallery_sim_matrix, query_to_refined_gallery_sim_matrix, query_to_gallery_find_matrix2 = cal_gal_querry_sim_maxtrix(
            query_feature_dataset, gallery_feature_dataset, k, weights, bias, weights3, bias3)
        del weights, bias, weights3, bias3


        torch.cuda.empty_cache()
        # Use broadcasting to create the binary matrix
        Sim_cosin_time = time.time()
        print("Sim Cosin")
        print(evaluate_all(1-query_to_gallery_sim_matrix, args,
              query=query_dataset, gallery=gallery_dataset))
        print(f'Cost time={time.time() - Sim_cosin_time}s')
        torch.cuda.empty_cache()

        Urf_time = time.time()
        print("urf")
        print(evaluate_all(1-query_to_refined_gallery_sim_matrix,
              args, query=query_dataset, gallery=gallery_dataset))
        print(f'Cost time={time.time() - Urf_time}s')
        
        torch.cuda.empty_cache()

        # times = time.time()
        # print("Cosin+urf+cce")
        # print(evaluate_all(1-query_to_gallery_find_matrix3, args,
        #       query=query_dataset, gallery=gallery_dataset))
        # torch.cuda.empty_cache()
        # print(f'Cost time={time.time() - times}s')

        times = time.time()

        print("Cosin+urf")
        print(evaluate_all(1-query_to_gallery_find_matrix2, args,
              query=query_dataset, gallery=gallery_dataset))
        print(f'Cost time={time.time() - times}s')
        print("k: ", k)


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    now = datetime.now()
    log_filename = now.strftime("%m%d%y_%H%M_log.txt")
    log_filepath = osp.join(args.logs_dir, log_filename)

    # Sử dụng Logger để ghi log
    sys.stdout = Logger(log_filepath) 

    # In ra thông báo để kiểm tra
    print(f"Logging to {log_filepath}") 
    main_worker(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="COMMON ")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('-tt', '--dataset-target', type=str, default='msmt17',
                        choices=datasets.names())
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir,
                                         '/hgst/longdn/UCF-main/logs/WDA/duke2msmt/'))

    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--ncs', type=str, default='60')
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--features', type=int, default=0)

    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=6)
    parser.add_argument('-a', '--arch', type=str,
                        default='resnet50', choices=models.names())
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--init-1', type=str,
                        default='/hgst/longdn/UCF-main/logs/dbscan/duke2msmt/model_best.pth.tar',
                        metavar='PATH')
    parser.add_argument('--init-2', type=str,
                        default='/hgst/longdn/UCF-main/logs/dbscan/duke2msmt/model_best.pth.tar',
                        metavar='PATH')
    parser.add_argument('--init-3', type=str,
                        default='/hgst/longdn/UCF-main/logs/dbscan/duke2msmt/model_best.pth.tar',
                        metavar='PATH')
    parser.add_argument('--k1', type=int, default=20,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--HC', action='store_true',
                        help="active the hierarchical clustering (HC) method")
    parser.add_argument('--UCIS', action='store_true',
                        help="active the uncertainty-aware collaborative instance selection (UCIS) method")
    main()
