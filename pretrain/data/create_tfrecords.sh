#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

CORPUS=${1}

if test -n "${CORPUS}"
then
    for length in 128 512
    do
        # Create tfrecords files Phase 1, 2
        python3 /workspace/electra/data/dataPrep.py \
            --dataset ${CORPUS} \
            --max_seq_length ${length} \
            --vocab_file=vocab/vocab.txt \
            --n_training_shards 2048 \
            --n_processes=32
    done
else
    echo "Need argument: Corpus Dataset NAME (pretrain/data/shard/<Corpus Dataset NAME>)"
fi