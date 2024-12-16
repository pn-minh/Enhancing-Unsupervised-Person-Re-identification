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
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import numpy as np
from torch.backends import cudnn
from UDAsbs import datasets
import argparse
import os.path as osp
import random
import numpy as np
import sys
import torch
import torchvision.models as models
import time
import torch
from torch import nn
from torch.backends import cudnn
from UDAsbs import datasets
from UDAsbs import models
from UDAsbs.utils.serialization import load_checkpoint  # , copy_state_dict
from sbs_traindbscan import *
from sklearn.linear_model import LinearRegression
from UDAsbs.evaluators1 import extract_features, pairwise_distance


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
    # Nạp trọng số từ checkpoint
    if args.init_1:
        initial_weights = load_checkpoint(args.init_1)
        copy_state_dict(initial_weights['state_dict'], model)
        print('load pretrain model:{}'.format(args.init_1))

    # Chuyển mô hình lên GPU
    model.cuda()

    # Sử dụng DataParallel
    model = nn.DataParallel(model)

    return model

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

def k_nearest_neighbors(dist_matrix, k):

    assert dist_matrix.shape[0] == dist_matrix.shape[1], "Matrix must be square"

    neighbors = torch.zeros((dist_matrix.shape[0], k), dtype=torch.long)

    for i in range(dist_matrix.shape[0]):
        dist = dist_matrix[i]
        dist[i] = float('inf')  # Set self-similarity to -inf to exclude it
        _, indices = torch.topk(dist, k, largest= False)
        neighbors[i] = indices

    return neighbors

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

def refined_features(feature_dataset, dist_matrix, neighbors):

    refined_feats = torch.zeros(
        len(feature_dataset), len(feature_dataset[0][0]))

    for i in range(len(feature_dataset)):

        for j in neighbors[i]:

            refined_feats[i] += feature_dataset[j.item()][0] / len(neighbors[i])

    return refined_feats

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
    
class FeatureDataset1(Dataset): 
    def __init__(self, features, labels, cam_angles):
        # features, labels, cam_angles 
        self.features = features
        self.labels = labels 
        self.cam_angles = cam_angles 

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

def pairwise_distance1(tensor1, tensor2, metric=None):

    tensor1 = tensor1.view(1, -1)
    tensor2 = tensor2.view(1, -1)

    # Optionally apply the metric transformation if provided
    if metric is not None:
        tensor1 = metric.transform(tensor1)
        tensor2 = metric.transform(tensor2)

    # Compute the pairwise squared distances
    dist_m = torch.pow(tensor1, 2).sum(dim=1, keepdim=True) + \
             torch.pow(tensor2, 2).sum(dim=1, keepdim=True).t()
    dist_m -= 2 * torch.mm(tensor1, tensor2.t())

    return dist_m[0][0]

def cal_weights_bias(feature_dataset, triplet_dataset, k, dist_matrix= None):
    # Now you can use this dataset with a DataLoader for further processing or model training
    # dataset1 = feature_dataset
    # dataset2 = feature_dataset
    if dist_matrix== None:
        features = [record[0] for record in feature_dataset]
        dist_matrix = pairwise_distance(features)
    neighbors = k_nearest_neighbors(dist_matrix, k)
    refined_feats = refined_features(
        feature_dataset, dist_matrix=dist_matrix, neighbors=neighbors)
    
    positive_instances = []
    negative_instances = []
    positive_instances3 = []
    negative_instances3 = []
    for i, item in enumerate(triplet_dataset):

        anchor_index = item[0][0]
        pos_index = item[0][1]
        neg_index = item[0][2]
        anchor_feat = feature_dataset[anchor_index][0]

        pos_dist = dist_matrix[anchor_index, pos_index].item()
        neg_dist = dist_matrix[anchor_index, neg_index].item()

        pos_urf = refined_feats[pos_index]
        neg_urf = refined_feats[neg_index]

        pos_urf_dist = pairwise_distance1(anchor_feat, pos_urf)
        neg_urf_dist = pairwise_distance1(anchor_feat, neg_urf)

        pos_cce = float(item[2][0].item() == item[2][1].item())
        neg_cce = float(item[2][0].item() == item[2][2].item())

        positive_instances3.append((pos_dist, pos_urf_dist, pos_cce, -100))
        negative_instances3.append((neg_dist, neg_urf_dist, neg_cce, 100))
        positive_instances.append((pos_dist, pos_urf_dist, -100))
        negative_instances.append((neg_dist, neg_urf_dist, 100))

    positive_input = np.array(positive_instances)
    negative_input = np.array(negative_instances)
    
    positive_input3 = np.array(positive_instances3)
    negative_input3 = np.array(negative_instances3)
    input = np.concatenate((positive_input, negative_input), axis=0)
    input3 = np.concatenate((positive_input3, negative_input3), axis=0)
    # Separate inputs (x, y, z) and outputs (label)
    X3 = input3[:, :3]

    y3 = input3[:, 3]

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

def cal_gal_query_dist_maxtrix(query_feature_dataset, gallery_feature_dataset, k, weights, bias, weights3, bias3):
    query_features= [record[0] for record in query_feature_dataset]
    gallery_features= [record[0] for record in gallery_feature_dataset]
    query_to_gallery_dist_matrix, _, _ = pairwise_distance(query= query_features, gallery= gallery_features)
    gallery_dist_matrix = pairwise_distance(features= gallery_features)
    gallery_neighbors = k_nearest_neighbors(gallery_dist_matrix, k)
    gallery_refined_feats = refined_features(feature_dataset=gallery_feature_dataset, dist_matrix=gallery_dist_matrix, neighbors=gallery_neighbors)
    refined_gallery_features = [gallery_refined_feats[i] for i in range(len(gallery_refined_feats))]
#    Create the dataset
    query_to_refined_gallery_dist_matrix, _, _ = pairwise_distance(query= query_features, gallery= refined_gallery_features)
    
    query_cam_angles = torch.tensor([sample[2] for sample in query_feature_dataset])
    gallery_cam_angles = torch.tensor([sample[2] for sample in gallery_feature_dataset])

    cce_matrix = (query_cam_angles[:, None] == gallery_cam_angles[None, :]).float()
    
    bias_matrix = torch.full(cce_matrix.shape, bias)
    query_to_gallery_find_matrix2 = weights[0] * query_to_gallery_dist_matrix + weights[1] * query_to_refined_gallery_dist_matrix  -  bias_matrix
    
    bias_matrix3 = torch.full(cce_matrix.shape, bias3)
    query_to_gallery_find_matrix3 = weights3[0] * query_to_gallery_dist_matrix + weights3[1] * query_to_refined_gallery_dist_matrix + weights3[2] * cce_matrix + bias_matrix3

    return query_to_gallery_dist_matrix, query_to_refined_gallery_dist_matrix, query_to_gallery_find_matrix2, query_to_gallery_find_matrix3

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

def evaluate_all(dist_matrix, args,  query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    dist_matrix = dist_matrix.cuda()

    if query is not None and gallery is not None:
        query_ids = [item[1] for item in query]
        gallery_ids = [item[1] for item in gallery]
        query_cams = [item[-1] for item in query]
        gallery_cams = [item[-1] for item in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Tính toán mAP
    mAP = mean_ap(dist_matrix, query_ids, gallery_ids, query_cams, gallery_cams)

    print('Mean AP: {:4.1%}'.format(mAP))
    torch.cuda.empty_cache()
    cmc_configs = {
        args.dataset_target: dict(separate_camera_set=False,
                                  single_gallery_shot=False,
                                  first_match_break=True)
    }
    torch.cuda.empty_cache()
    # Tính toán CMC scores
    cmc_scores = {name: cmc(dist_matrix, query_ids, gallery_ids,
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
    fc_len = 32621
    ncs = [int(x) for x in args.ncs.split(',')]
    model = create_model2(args, [fc_len for _ in range(len(ncs))])

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if (args.dataset_target == 'msmt17'):
        # Replace with the path to your folder
        train_folder = '/hgst/longdn/UCF-main/data/MSMT17_V1/bounding_box_train'
        gallery_folder = '/hgst/longdn/UCF-main/data/MSMT17_V1/bounding_box_test'
        # Replace with the path to your folder
        query_folder = '/hgst/longdn/UCF-main/data/MSMT17_V1/query'
    elif (args.dataset_target == 'duke'):
        train_folder = '/hgst/longdn/UCF-main/data/dukemtmc/bounding_box_train'
        gallery_folder = '/hgst/longdn/UCF-main/data/dukemtmc/bounding_box_test'
        query_folder = '/hgst/longdn/UCF-main/data/dukemtmc/query'
    elif (args.dataset_target == 'market1501'):
        train_folder = '/hgst/longdn/UCF-main/data/market1501/Market-1501-v15.09.15/bounding_box_train'
        gallery_folder = '/hgst/longdn/UCF-main/data/market1501/Market-1501-v15.09.15/bounding_box_test'
        query_folder = '/hgst/longdn/UCF-main/data/market1501/Market-1501-v15.09.15/query'
    gallery_jpg_files = get_jpg_files(gallery_folder)
    gallery_dataset = Dataset(records= gallery_jpg_files, transform=transform)
    query_jpg_files = get_jpg_files(query_folder)
    query_dataset = Dataset(records= query_jpg_files, transform=transform)


    batch_size = 256
    gallery_dataloader = DataLoader(
        gallery_dataset, batch_size=batch_size, shuffle=False)
    query_dataloader = DataLoader(
        query_dataset, batch_size=batch_size, shuffle=False)
    

    train_jpg_files = get_jpg_files(train_folder)
    train_dataset = Dataset(train_jpg_files, transform=transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)

    features, labels, cam_angles, images = extract_features(model= model, data_loader= train_dataloader)

    train_feature_dataset = FeatureDataset1(features= list(features.values()), labels= list(labels.values()), cam_angles= list(cam_angles.values()))



    # Example usage
    extract_and_save_features(model, gallery_dataloader, save_path='/hgst/longdn/UCF-main/logs/WDA1/market2duke/ExtractData/gallery_features.npy')

    extract_and_save_features(model, query_dataloader, save_path='/hgst/longdn/UCF-main/logs/WDA1/market2duke/ExtractData/query_features.npy')



    gallery_feature_dataset = FeatureDataset(
        '/hgst/longdn/UCF-main/logs/WDA1/market2duke/ExtractData/gallery_features.npy')
    query_feature_dataset = FeatureDataset(
        '/hgst/longdn/UCF-main/logs/WDA1/market2duke/ExtractData/query_features.npy')
    gallery_jpg_files = get_jpg_files(gallery_folder)
    gallery_dataset = Dataset(gallery_jpg_files, transform=transform)
    gallery_dataloader = DataLoader(
        gallery_dataset, batch_size=batch_size, shuffle=False)

    gallery_features, gallery_labels, gallery_cam_angles, gallery_images = extract_features(model= model, data_loader= gallery_dataloader)

    gallery_feature_dataset = FeatureDataset1(features= list(gallery_features.values()), labels= list(gallery_labels.values()), cam_angles= list(gallery_cam_angles.values()))

    query_jpg_files = get_jpg_files(query_folder)
    query_dataset = Dataset(query_jpg_files, transform=transform)
    query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)

    query_features, query_labels, query_cam_angles, images = extract_features(model= model, data_loader= query_dataloader)

    query_feature_dataset = FeatureDataset1(features= list(query_features.values()), labels= list(query_labels.values()), cam_angles= list(query_cam_angles.values()))

    del  query_jpg_files, gallery_jpg_files
    torch.cuda.empty_cache()
    all_weights_and_bias = {}  # Khởi tạo dictionary để lưu trữ

    train_triplet_dataset = TripletDataset(train_dataset)

    for k in range(8,9):

        weights, bias, weights3, bias3 = cal_weights_bias(train_feature_dataset, train_triplet_dataset, k)

        # Lưu trữ k, weights và bias vào dictionary
        all_weights_and_bias[k] = {
            "weights": weights, 
            "bias": bias, 
            "weights3": weights3, 
            "bias3": bias3
        }
        torch.cuda.empty_cache()

# #     # Lưu dictionary ra file
    save_dir = "/hgst/longdn/UCF-main/logs/WDA1/market2duke/ExtractData"
    os.makedirs(save_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    save_path = os.path.join(save_dir, "all_weights_and_bias.pth")
    torch.save(all_weights_and_bias, save_path)

    save_path = os.path.join(save_dir, "all_weights_and_bias.pth")
    all_weights_and_bias = torch.load(save_path)
    for k in range(8,9):

        weights = all_weights_and_bias[k]["weights"]
        bias = all_weights_and_bias[k]["bias"]
        weights3 = all_weights_and_bias[k]["weights3"]
        bias3 = all_weights_and_bias[k]["bias3"]
        # # Buoc2
        query_to_gallery_dist_matrix, query_to_refined_gallery_dist_matrix, query_to_gallery_find_matrix2, query_to_gallery_find_matrix3 = cal_gal_query_dist_maxtrix(
            query_feature_dataset, gallery_feature_dataset, k, weights, bias, weights3, bias3)
        del weights, bias, weights3, bias3


        torch.cuda.empty_cache()
        # Use broadcasting to create the binary matrix
        print("Euclidean distance")
        euclidean_dist_time = time.time()
        print(evaluate_all(query_to_gallery_dist_matrix, args,
              query=query_dataset, gallery=gallery_dataset))
        print(f'Cost time={time.time() - euclidean_dist_time}s')
        torch.cuda.empty_cache()

        print("urf")
        Urf_time = time.time()
        print(evaluate_all(query_to_refined_gallery_dist_matrix,
              args, query=query_dataset, gallery=gallery_dataset))
        print(f'Cost time={time.time() - Urf_time}s')
        torch.cuda.empty_cache()

        times = time.time()
        print("Euclidean distance +urf+cce")
        print(evaluate_all(query_to_gallery_find_matrix3, args, query=query_dataset, gallery=gallery_dataset))
        torch.cuda.empty_cache()
        print(f'Cost time={time.time() - times}s')

        print("Euclidean distance + urf")
        times = time.time()
        print(evaluate_all(query_to_gallery_find_matrix2, args, query=query_dataset, gallery=gallery_dataset))
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
    parser.add_argument('-dt', '--dataset-target', type=str, default='duke',
                        choices=datasets.names())
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir,
                                         '/hgst/longdn/UCF-main/logs/WDA1/market2duke/'))

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
                        default='/hgst/longdn/UCF-main/logs/dbscan/market2duke/model_best.pth.tar',
                        metavar='PATH')
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--HC', action='store_true',
                        help="active the hierarchical clustering (HC) method")
    parser.add_argument('--UCIS', action='store_true',
                        help="active the uncertainty-aware collaborative instance selection (UCIS) method")
    main()