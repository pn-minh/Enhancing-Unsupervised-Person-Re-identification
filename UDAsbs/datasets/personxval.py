from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class personXval(BaseImageDataset):

    dataset_dir = '.'

    def __init__(self, root, ncl=1, verbose=True, **kwargs):
        super(personXval, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'challenge_datasets/target_training/image_train/')
        self.query_dir = osp.join(self.dataset_dir, 'challenge_datasets/target_validation/image_query/')
        self.gallery_dir = osp.join(self.dataset_dir, 'challenge_datasets/target_validation/image_gallery/')


        self._check_before_run()

        self.ncl=ncl
        self.num_cam= 6
        # camstytrain = self._process_dir(self.camstylegallery_dir, relabel=True)
        self.name2camera_path=osp.join(self.dataset_dir, 'challenge_datasets/target_training/label_target_training.txt')
        with open(self.name2camera_path,'r') as f:
            self.name2camera=f.readlines()

        train = self._process_dir_train(self.train_dir,self.name2camera)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> personXval loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
    def _process_dir_train(self, dir_path, name2cam):
        #img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')

        dataset = []
        for item_pandc in name2cam:
            img_path,camid=item_pandc.strip('\n').split(' ')
            img_path=osp.join(dir_path,img_path)
            pid = 0#map(int, pattern.search(img_path).groups())
            camid = int(camid)
            pids = ()
            for _ in range(self.ncl):
                pids = (pid,) + pids
            item = (img_path,) + pids + (camid,)
            dataset.append(item)

        return dataset

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            pids=()
            for _ in range(self.ncl):
                pids=(pid,)+pids
            item=(img_path,) + pids + (camid,)
            dataset.append(item)

        return dataset
