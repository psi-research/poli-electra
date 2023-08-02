from transformers import pipeline
from pprint import pprint
import argparse


def main(model):
    pipe = pipeline(task="text-classification", model=model, top_k=3)
    while True:
        text = input("입력: ")
        result = pipe(text)[0]
        pprint(result)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path/to/model")
    args = parser.parse_args()

    main(args.model)  # models/tf/pvo-3mstep/finetune/police
