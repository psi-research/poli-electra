# Poli-ELECTRA
ELECTRA 데이터 전처리, 사전학습, 미세조정 Pipeline

## Requirements
- OS: Linux Ubuntu
- GPU: NVIDIA `Volta|Turing|Ampere` Architectures
- Docker

## Environments
### 1. 기본 개발환경 구축
#### Online
```commandline
pip install -r requirements.txt
```

#### Offline
```commandline
pip install --no-index --find-links=wheelhouse -r requirements.txt
```


### 2. 사전학습 개발환경 구축
#### Docker image 로딩
```commandline
docker load -i poli-electra_tf-pretrain.tar
```

#### Docker Container 최초 실행
```commandline
cd pretrain && bash scripts/docker/launch.sh
```

#### Docker Container 재접속
```commandline
cd pretrain && bash scripts/docker/exec.sh
```

컨테이너에서 나오기: `Ctrl` + `p`, `q`


### 3. 미세조정 개발환경 구축
#### Docker image 로딩
```commandline
$ docker load -i poli-electra_tf-finetune.tar
```

#### Docker Container 최초 실행
```commandline
cd finetune && bash docker/launch.sh
```

#### Docker Container 재접속
```commandline
cd finetune && bash docker/exec.sh
```

컨테이너에서 나오기: `Ctrl` + `p`, `q`  


## Usage
### 0. Vocab 만들기
`vocab` 폴더로 이동합니다.
```commandline
cd vocab
```

`WPM.py` 파일을 실행합니다.
```commandline
python WPM.py --corpus path/to/text1.txt path/to/text2.txt --size 35000
```


### 1. 데이터 전처리
전처리 된 말뭉치들을 `pretrain/data/corpus` 폴더에 복사합니다.

앞서 만든 `vocab.txt`를 `pretrain/vocab` 폴더에 복사합니다.
```commandline
cp vocab/vocab.txt pretrain/vocab
```

사전학습을 위한 개발 환경이 구축되어 있는 Docker Container 내부로 진입합니다.
```commandline
cd pretrain && bash scripts/docker/exec.sh
```

#### 1) 말뭉치 통합
Sharding & Shuffling 작업을 위해 말뭉치를 통합합니다.
```commandline
cd data/corpus && cat text1.txt text2.txt text3.txt > corpus.txt
```

#### 2) Shard
앞서 만든 통합 말뭉치의 파일 이름만 인자로 사용합니다.
```commandline
cd .. && bash split_corpus.sh corpus
```

#### 3) TFrecord 변환
앞서 만든 통합 말뭉치의 파일 이름만 인자로 사용합니다.
```commandline
cd .. && bash data/create_tfrecords.sh corpus
```
결과는 `data/tfrecord_len128/corpus`, `data/tfrecord_len512/corpus`에 저장됩니다.


### 2. 사전학습
`run_pretraining.py`의 `96 Line`에서 `vocab size`를 확인합니다. (`self.vocab_size`)
```commandline
vi run_pretraining.py 
```

`Hyper-parameter`를 확인합니다.
```commandline
vi scripts/run_pretraining.sh 
```

사전학습을 시작합니다.
```commandline
bash scripts/run_pretraining.sh
```

##### Output Structure (ex. # of GPU: 2)
```
results/
├── 20221019-051449
│   ├── train_0_of_4
│   │   └── events.out.tfevents.1666156489.pvoice-ESC4000-G4.1550.0.v2
│   ├── train_1_of_4
│   │   └── events.out.tfevents.1666156489.pvoice-ESC4000-G4.1551.0.v2
├── backup
├── dllogger_rank0.log
├── dllogger_rank1.log
├── electra_lamb_pretraining.electra_pretraining_phase1_amp.221019051441.log
├── electra_lamb_pretraining.electra_pretraining_phase2_amp.221019052443.log
└── models
    └── base
        └── checkpoints
            ├── checkpoint
            ├── ckpt-0.data-00000-of-00001
            ├── ckpt-0.index
            ├── iter_ckpt_rank_00
            │   ├── checkpoint
            │   ├── iter_ckpt_rank_00-0.data-00000-of-00001
            │   └── iter_ckpt_rank_00-0.index
            ├── iter_ckpt_rank_01
            │   ├── checkpoint
            │   ├── iter_ckpt_rank_01-0.data-00000-of-00001
            │   └── iter_ckpt_rank_01-0.index
            └── pretrain_config.json
```

#### 모니터링
아래 명령어 수행 후, http://localhost:6006으로 접속합니다.
```commandline
tensorboard --logdir results --bind_all
```

#### 후처리
앞서 만들어진 `ckpt` 파일을 `backup` 폴더로 복사합니다.
```commandline
cp results/models/base/checkpoints/ckpt-0.* results/backup
```

Discriminator를 분리합니다.
```commandline
bash scripts/split_disc_gen.sh
```
##### Output Structure
```
results/backup/
├── ckpt-0
│   ├── discriminator
│   │   ├── config.json
│   │   └── tf_model.h5
│   └── generator
│       ├── config.json
│       └── tf_model.h5
├── ckpt-0.data-00000-of-00001
└── ckpt-0.index
```

사전학습이 모두 끝나면 컨테이너에서 나옵니다: `Ctrl` + `p`, `q`


### 3. 미세조정
앞서 만든 사전학습 모델(`discriminator/tf_model.h5`)을 `finetune/model` 폴더에 복사합니다.
```commandline
cp path/to/discriminator/tf_model.h5 finetune/model
```

미세조정을 위한 개발 환경이 구축되어 있는 Docker Container 내부로 진입합니다.
```commandline
cd finetune && bash docker/exec.sh
```

Benchmark 테스트를 진행합니다.
```commandline
bash run.sh
```

Benchmark 테스트가 끝나면 컨테이너에서 나옵니다: `Ctrl` + `p`, `q`


### 4. 추론
미세조정이 끝난 모델로 추론을 진행합니다.
```commandline
python inference.py --model path/to/model
```


## Reference
### 사전학습
- [WordPiece Vocab 생성](https://huggingface.co/docs/tokenizers/quicktour)
- [사전학습 개발환경 도커 이미지 상세 스펙](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
- [사전학습 개발환경 도커 이미지 목록](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow/tags)
- [사전학습 소스](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/ELECTRA)

### 미세조정
- [파인튜닝 개발환경 도커 이미지 목록](https://hub.docker.com/r/tensorflow/tensorflow/tags)
- [Text Classification 소스](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb)
- [Token Classification 소스](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb)
- [Question Answering 소스](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)
- [Benchmark dataset - NSMC](https://huggingface.co/datasets/nsmc)
- [Benchmark dataset - Naver NER](https://github.com/monologg/KoELECTRA/tree/master/finetune/data/naver-ner)
- [Benchmark dataset - KorQuAD v1.0](https://huggingface.co/datasets/squad_kor_v1)
- [Benchmark dataset - PAWS-X](https://huggingface.co/datasets/paws-x)
- [Benchmark dataset - KorNLU: STS, NLI](https://huggingface.co/datasets/kor_nlu)
- [Benchmark dataset - Question Pair](https://huggingface.co/datasets/kor_qpair)
- [Benchmark dataset - Korean Hate Speech](https://huggingface.co/datasets/kor_hate)
- [Dataload를 위한 JSON 서식 만들기](https://huggingface.co/docs/datasets/loading#json)
