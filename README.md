# AutoTransition: Learning to Recommend Video Transition Effects

This is an official implementation of AutoTransition: Learning to Recommend Video Transition Effects.

## Transition Dataset

We release the videos with annotated transitions extracted from the video editing template on
online video editing platforms. Due to the privacy policy, we only release the link to the videos.
The dataset can be downloaded from
here: [Download Link](https://drive.google.com/file/d/179r-bRu9trSgqh4ejhyplfFKO2ta98BT/view?usp=sharing)

Use the following command to download the source video in the dataset:

```shell
python3 tools/download_videos.py annotation.json ./template_download
```

The videos will be downloaded to `./template_download`.

## Usage

### Prepare Data

To speed up the training, we convert videos to JPEG image and extract audio features before training.
Run the following commands to finish these steps:

```shell
python3 preprocess/convert_video_folder.py ./path/to/template_root
python3 preprocess/extract_audio_features.py ./path/to/template_root path/to/annotation.json --model_path /path/to/audio/model.pth --cuda
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
