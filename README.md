# AutoTransition: Learning to Recommend Video Transition Effects

This is an official implementation of the ECCV 2022 paper [AutoTransition: Learning to Recommend Video Transition Effects](https://arxiv.org/abs/2207.13479).

## Transition Dataset

We release the videos with annotated transitions extracted from the video editing template on
online video editing platforms.
The dataset can be downloaded from here: [HF Dataset](https://huggingface.co/datasets/yaojie-shen/AutoTransition)

## Usage

### Prepare Data

To speed up the training, we convert videos to JPEG image and extract audio features before training.
Run the following commands to finish these steps:

```shell
python3 tools/convert_video_folder.py ./path/to/template_root
python3 tools/extract_audio_features.py ./path/to/template_root path/to/annotation.json --model_path /path/to/audio/model.pth --cuda
```

The pretrained Harmonic CNN model could be downloaded
from [this link](https://drive.google.com/file/d/1PwtNen0LJDp7werYPCoMAVDnipPOykrc/view?usp=sharing).

### Train & Test

To train transition embeddings:

```shell
python3 tools/run_net.py --cfg configs/base/train_transition_embedding.yaml \
  DATASET.TRANSITION_CLASSIFICATION.JSON_ANNOTATION /path/to/annotation.json \
  DATASET.TRANSITION_CLASSIFICATION.TEMPLATE_ROOT /path/to/template_root
```

The transition embeddings can be found in `./log` directory after training.

To train transition recommendation:

```shell
python3 tools/run_net.py --cfg configs/base/train_transition_recommendation.yaml \
  MODEL.TRANSITION_TRANSFORMER.EMBEDDING.PRETRAINED_EMBEDDING /path/to/pretrained/transition/embedding.pth \
  DATASET.TRANSITION_DATASET.JSON_ANNOTATION /path/to/annotation.json \
  DATASET.TRANSITION_DATASET.TEMPLATE_ROOT /path/to/template_root

tensorboard --logdir=./log
```
