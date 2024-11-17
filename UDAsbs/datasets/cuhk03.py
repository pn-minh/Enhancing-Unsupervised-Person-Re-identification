from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

class CUHK03(BaseImageDataset):
    dataset_dir = '/hgst/longdn/UCF-main/data/CUHK03/CUHK03/'
    def __init__(self, root, ncl=1, verbose=True, **kwargs):
        super(CUHK03, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self.ncl = ncl
        self._check_before_run()

        train = self._process_dir1(self.train_dir, relabel=True)
        query = self._process_dir1(self.query_dir, relabel=False)
        gallery = self._process_dir1(self.gallery_dir, relabel=False)
        print(self.gallery_dir)
        if verbose:
            print("=> CUHK03 loaded")
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

    def _process_dir1(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        # Adjust regex pattern for new naming format: camid_pid_viewcam_pictureid
        pattern = re.compile(r'\d+_(\d+)_(\d+)_\d+')

        # Collect unique person IDs
        pid_container = set()
        for img_path in img_paths:
            pid, camid  = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # ignore junk images
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid  = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # ignore junk images
            assert 0 <= pid <= 1467  # pid == 0 means background
            assert 1 <= camid <= 2  # valid camid range
            camid -= 1  # make camid zero-indexed

            # Relabel pid if necessary
            if relabel:
                pid = pid2label[pid]

            # Create a tuple of repeated pids
            pids = ()
            for _ in range(self.ncl):
                pids = (pid,) + pids
            item = (img_path,) + pids + (camid,)
            dataset.append(item)

        return dataset