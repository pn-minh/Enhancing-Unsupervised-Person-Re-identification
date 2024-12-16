# Enhancing unsupervised person re-identification using camera angle aware distance aggregation

## Requirements

### Prepare Datasets

Download the person datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [DukeMTMC](https://arxiv.org/abs/1609.01775).
Then unzip them under the directory like
```
data
├── market1501
│   └── Market-1501-v15.09.15
└── dukemtmc
    └── dukemtmc
```

## Training

We utilize 2 GPUs for training. **Note that**

+ use `--width 128 --height 256` (default) for datasets.
+ use `-a resnet50` (default) for the backbone of ResNet-50.

### Unsupervised Domain Adaptation
To train the model(s) in the paper, run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1 \
python source_pretrained.py \
  -ds $SOURCE_DATASET -dt $TARGET_DATASET --logs-dir $PATH_OF_LOGS
```

**Some examples:**
```shell
### Market-1501 -> Duke ###
# use all default settings is ok
CUDA_VISIBLE_DEVICES=0,1 \
python source_pretrained.py \
  -ds market1501 -dt duke --logs-dir logs/pretrained/market2msmt
# after pretraining , to train a baseline:
CUDA_VISIBLE_DEVICES=0,1 \
python sbs_traindbscan.py \
  -ds market1501 -dt duke --logs-dir logs/dbscan/market2duke \
  --init-1 logs/pretrained/market2duke/model_best.pth.tar
# after pretraining , to train a baseline + HC(Hierarchical clustering) + UCIS(uncertainty-aware collaborative instance selection) :
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/sbs_traindbscan.py \
  -ds market1501 -dt duke --logs-dir logs/dbscan/market2duke \
  --init-1 logs/pretrained/market2msmt/model_best.pth.tar \
  --HC --UCIS
```

## Evaluation

We utilize 2 GPUs for testing. **Note that**

+ use `--width 128 --height 256` (default) for person datasets.
+ use `-a resnet50` (default) for the backbone of ResNet-50.

### Unsupervised Domain Adaptation

To evaluate the domain adaptive model on the **target-domain** dataset, run:
```shell
CUDA_VISIBLE_DEVICES=0,1 \
python sbstest.py  \
  -dt $DATASET --init-1 $PATH_OF_MODEL
```

**Some examples:**
```shell
### Market-1501 -> Duke ###
# test on the target domain
CUDA_VISIBLE_DEVICES=0,1 \
python sbstest.py --evaluate \
  -dt duke --init-1 logs/dbscan/market2duke/model_best.pth.tar
```

### Wise Distance Aggregation
To evaluate WDA on the **target-domain** dataset, run:
```shell
CUDA_VISIBLE_DEVICES=0,1 \
python sbstest.py  \
  -dt $DATASET --init-1 $PATH_OF_MODEL
```

**Some examples:**
```shell
### Market-1501 -> Duke ###
# test on the target domain
CUDA_VISIBLE_DEVICES=0,1 \
python3 WDA.py --dt duke --init-1 logs/dbscan/market2duke/model_best.pth.tar
```




