# from __future__ import print_function, absolute_import
# import os.path as osp
# import tarfile

# import glob
# import re
# import urllib
# import zipfile

# from ..utils.osutils import mkdir_if_missing
# from ..utils.serialization import write_json

# style='MSMT17_V1'
# def _pluck_msmt(list_file, subdir, ncl, pattern=re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')):
#     with open(list_file, 'r') as f:
#         lines = f.readlines()
#     ret = []
#     pids_ = []
#     for line in lines:
#         line = line.strip()
#         fname = line.split(' ')[0]
#         pid, _, cam = map(int, pattern.search(osp.basename(fname)).groups())
#         if pid not in pids_:
#             pids_.append(pid)

#         img_path=osp.join(subdir,fname)
#         pids = ()
#         for _ in range(ncl):
#             pids = (pid,) + pids
#         item = (img_path,) + pids + (cam,)

#         ret.append(item)
#         # ret.append((osp.join(subdir,fname), pid, cam))
#     return ret, pids_

# class Dataset_MSMT(object):
#     def __init__(self, root, ncl):
#         self.root = root
#         self.train, self.val, self.trainval = [], [], []
#         self.query, self.gallery = [], []
#         self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0
#         self.ncl=ncl
#     @property
#     def images_dir(self):
#         return osp.join(self.root, style)

#     def load(self, verbose=True):
#         exdir = osp.join(self.root, style)
#         nametrain= 'train'#'mask_train_v2'
#         nametest = 'test'#'mask_test_v2'
#         self.train, train_pids = _pluck_msmt(osp.join(exdir, 'list_train.txt'), nametrain, self.ncl)
#         self.val, val_pids = _pluck_msmt(osp.join(exdir, 'list_val.txt'), nametrain, self.ncl)
#         self.train = self.train + self.val
#         self.query, query_pids = _pluck_msmt(osp.join(exdir, 'list_query.txt'), nametest, self.ncl)
#         self.gallery, gallery_pids = _pluck_msmt(osp.join(exdir, 'list_gallery.txt'), nametest, self.ncl)
#         self.num_train_pids = len(list(set(train_pids).union(set(val_pids))))


#         if verbose:
#             print(self.__class__.__name__, "v1~~~ dataset loaded")
#             print("  ---------------------------")
#             print("  subset   | # ids | # images")
#             print("  ---------------------------")
#             print("  train    | {:5d} | {:8d}"
#                   .format(self.num_train_pids, len(self.train)))
#             print("  query    | {:5d} | {:8d}"
#                   .format(len(query_pids), len(self.query)))
#             print("  gallery  | {:5d} | {:8d}"
#                   .format(len(gallery_pids), len(self.gallery)))
#             print("  ---------------------------")

# class MSMT17(Dataset_MSMT):

#     def __init__(self, root, ncl=1, split_id=0, download=True):
#         super(MSMT17, self).__init__(root,ncl)

#         if download:
#             self.download()

#         self.load()

#     def download(self):

#         import re
#         import hashlib
#         import shutil
#         from glob import glob
#         from zipfile import ZipFile

#         raw_dir = osp.join(self.root)
#         mkdir_if_missing(raw_dir)

#         # Download the raw zip file
#         fpath = osp.join(raw_dir, style)
#         if osp.isdir(fpath):
#             print("Using downloaded file: " + fpath)
#         else:
#             raise RuntimeError("Please download the dataset manually to {}".format(fpath))
from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

class MSMT17(BaseImageDataset):
    dataset_dir = '/hgst/longdn/UCF-main/data/MSMT17_V1'
    def __init__(self, root, ncl=1, verbose=True, **kwargs):
        super(MSMT17, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self.ncl = ncl
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> MSMT17 loaded")
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

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'^(\d{4})_c(\d)_\d{4}')
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 1404  # pid == 0 means background
            # assert 1 <= camid <= 8
            camid -= 1  # index starts from 0

            if relabel: pid = pid2label[pid]
            pids=()
            for _ in range(self.ncl):
                pids=(pid,)+pids
            item=(img_path,) + pids + (camid,)
            dataset.append(item)

            # if relabel: pid = pid2label[pid]
            # dataset.append((img_path, pid, camid))

        return dataset