WORKSPACE=$(pwd)

docker run \
    --restart always \
    --gpus all \
    -it \
    --name finetune \
    -v ${WORKSPACE}:/workspace \
    -w /workspace \
    -e TOKENIZERS_PARALLELISM=True \
    poli-electra:2.9.1 \
    bash
