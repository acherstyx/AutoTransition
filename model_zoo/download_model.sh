#!/bin/bash

# see: https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md
mkdir slowfast
(
  cd slowfast || exit
  wget "https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_4x16_R50.pkl"
  wget "https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl"
)
