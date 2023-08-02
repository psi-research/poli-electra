import os
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers.normalizers import Lowercase, StripAccents
import argparse
import json


def make_vocab(corpus, size):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )
    trainer = WordPieceTrainer(
        vocab_size=size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.train(corpus, trainer)
    tokenizer.save("vocab.json")


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    return result


def parse_vocab():
    vocab = load_json("vocab.json")
    parsed_vocab = list(vocab["model"]["vocab"].keys())
    save("vocab.txt", "\n".join(parsed_vocab))


def save(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        f.write(obj)


def main(corpus, size):
    make_vocab(corpus, size)  # vocab.json 파일 생성
    parse_vocab()  # vocab.json 파일로부터 필요한 부분만 추출해 vocab.txt로 저장
    os.remove("vocab.json")  # vocab.json 파일 삭제


if __name__ == '__main__':
    """
    Vocab의 최소 크기가 존재합니다. size=10,000으로 설정해도 30,000 크기의 vocab이 만들어질 수도 있습니다.
    """
    parser = argparse.ArgumentParser(description="Make WordPiece Tokenizer")
    parser.add_argument("-c", "--corpus", nargs="+", type=str, required=True, help="path/to/corpus.txt")
    parser.add_argument("-s", "--size", type=int, required=True, help="# of Tokens")
    args = parser.parse_args()

    main(**vars(args))