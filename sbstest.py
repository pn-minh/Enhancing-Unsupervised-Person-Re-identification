from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

import torch
from torch.backends import cudnn

from UDAsbs import datasets
from UDAsbs import models
from UDAsbs.evaluators import Evaluator
from UDAsbs.utils.logging import Logger
from UDAsbs.utils.serialization import load_checkpoint  # , copy_state_dict
from sbs_traindbscan import *
start_epoch = best_mAP = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):

    global start_epoch, best_mAP
    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    # Create data loaders
    ncs = [int(x) for x in args.ncs.split(',')]
    dataset_target, label_dict,ground_label_list = get_data(args.dataset_target, args.data_dir, len(ncs))
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    # Create model
    fc_len = 32621
    model, _, model_ema, _ = create_model(args, [fc_len for _ in range(len(ncs))])
    eps = 0.6
    print('Clustering criterion: eps: {:.3f}'.format(eps))
    evaluator = Evaluator(model)
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UCF Training")
    # data
    parser.add_argument('-st', '--dataset-source', type=str, default='duke',
                        choices=datasets.names())
    parser.add_argument('-tt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=6)
    parser.add_argument('--choice_c', type=int, default=0)
    parser.add_argument('--num-clusters', type=int, default=32621)
    parser.add_argument('--ncs', type=str, default='60')
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.000035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--moving-avg-momentum', type=float, default=0)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--soft-ce-weight', type=float, default=0.5)
    parser.add_argument('--soft-tri-weight', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)

    parser.add_argument('--lambda-value', type=float, default=0)
    # training configs

    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    parser.add_argument('--init-1', type=str,
                        default='/hgst/longdn/UCF-main/logs/dbscan/duke2market/model_best.pth.tar',
                        metavar='PATH')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=8)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PAT H',
                        default=osp.join(working_dir,
                                         '/hgst/longdn/UCF-main/logs/dbscan/duke2market/'))
    parser.add_argument('--log-name',type=str,default='')

    # UCF setting
    parser.add_argument('--HC', action='store_true',
                        help="active the hierarchical clustering (HC) method")
    parser.add_argument('--UCIS', action='store_true',
                        help="active the uncertainty-aware collaborative instance selection (UCIS) method")

    main()