#!/bin/bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TAG=${1:-"22.04"}
CMD=${2:-/bin/bash}
NV_VISIBLE_DEVICES=${3:-"all"}
DOCKER_BRIDGE=${4:-"host"}

docker run -it \
  --restart always \
  --gpus device=$NV_VISIBLE_DEVICES \
  --net=$DOCKER_BRIDGE \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  -e LD_LIBRARY_PATH='/workspace/install/lib/' \
  -v $PWD:/workspace/electra \
  --name pretrain \
  poli-electra:$TAG $CMD
