#!/usr/bin/docker bash
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

ckpt_dir="result/backup"

for ckpt_index in ${ckpt_dir}/*.index; do
    ckpt_name=${ckpt_index%.*}  # remove '.index' extension, leave only ckpt file name
    echo "==================================== START ${ckpt_name} ===================================="
    python postprocess_pretrained_ckpt.py --pretrained_checkpoint=${ckpt_name} --output_dir=${ckpt_dir}/$(basename "${ckpt_name}") --amp
    echo "====================================  END ${ckpt_name}  ====================================";
done