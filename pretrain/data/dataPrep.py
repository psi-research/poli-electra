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

import argparse
import os
import pprint
import subprocess


def main(args):
    working_dir = os.environ['DATA_PREP_WORKING_DIR']

    print('Working Directory:', working_dir)
    print('Dataset Name:', args.dataset)

    directory_structure = {
        'sharded' : working_dir + '/shard',
        'tfrecord' : working_dir + '/tfrecord'+ "_len" + str(args.max_seq_length),
    }

    print('\nDirectory Structure:')
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(directory_structure)
    print('')

    if not os.path.exists(directory_structure['tfrecord'] + "/" + args.dataset):
        os.makedirs(directory_structure['tfrecord'] + "/" + args.dataset)

    if args.vocab_file is None:
        args.vocab_file = os.path.join(working_dir, "vocab.txt")

    # Create TFrecords
    electra_preprocessing_command = 'python /workspace/electra/build_pretraining_dataset.py'
    electra_preprocessing_command += ' --corpus-dir=' + directory_structure['sharded'] + '/' + args.dataset
    electra_preprocessing_command += ' --output-dir=' + directory_structure['tfrecord'] + '/' + args.dataset
    electra_preprocessing_command += ' --vocab-file=' + args.vocab_file
    electra_preprocessing_command += ' --no-lower-case'
    electra_preprocessing_command += ' --max-seq-length=' + str(args.max_seq_length)
    electra_preprocessing_command += ' --num-processes=' + str(args.n_processes)
    electra_preprocessing_command += ' --num-out-files=' + str(args.n_training_shards)
    electra_preprocessing_process = subprocess.Popen(electra_preprocessing_command, shell=True)

    electra_preprocessing_process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing Application for Everything BERT-related'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Specify the dataset to perform --action on',
    )

    parser.add_argument(
        '--n_training_shards',
        type=int,
        help='Specify the number of training shards to generate',
        default=2048
    )

    parser.add_argument(
        '--n_processes',
        type=int,
        help='Specify the max number of processes to allow at one time',
        default=4
    )

    parser.add_argument(
        '--max_seq_length',
        type=int,
        help='Specify the maximum sequence length',
        default=512
    )

    parser.add_argument(
        '--vocab_file',
        type=str,
        help='Specify absolute path to vocab file to use)'
    )

    args = parser.parse_args()
    main(args)
