"""
Reference: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb
"""
import json
from transformers import ElectraTokenizerFast, TFAutoModelForTokenClassification, create_optimizer,\
    DataCollatorForTokenClassification, PreTrainedTokenizerFast
from datasets import load_dataset, load_metric
import numpy as np
from transformers.keras_callbacks import KerasMetricCallback
from tensorflow.keras.callbacks import TensorBoard
import argparse


label_all_tokens = True
metric = load_metric("seqeval")
tokenizer = ElectraTokenizerFast.from_pretrained("model")
assert isinstance(tokenizer, PreTrainedTokenizerFast)
data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors="tf")


def load_label(path):
    global label_list
    with open(path, "r", encoding="utf-8") as f:
        label_list = json.load(f)["label_list"]

    id2label = {str(i): label for i, label in enumerate(label_list)}
    label2id = {v: k for k, v in id2label.items()}

    return id2label, label2id


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],  # 어절 단위 리스트
        truncation=True,  # max_seq_length에 맞춰 전처리
        is_split_into_words=True  # 어절 단위 리스트를 토큰화 할 경우 적용
    )

    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main():
    # Load
    datasets = load_dataset("json", data_files={"train": "data/naver-ner/train.json", "test": "data/naver-ner/test.json"}, field="data")
    id2label, label2id = load_label("data/naver-ner/label_list.json")

    # Encode Data
    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

    # Create Optimizer
    batch_size = 8
    num_train_epochs = 3
    num_train_steps = (len(tokenized_datasets["train"]) // batch_size) * num_train_epochs
    optimizer, lr_schedule = create_optimizer(
        init_lr=2e-5,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
        num_warmup_steps=0,
    )

    # Build Model & Compile
    model = TFAutoModelForTokenClassification.from_pretrained("model", num_labels=len(label_list), id2label=id2label, label2id=label2id)
    model.compile(optimizer=optimizer)  # input length가 다 다르기 때문에 jit_compile은 오히려 낭비이므로 사용하지 않는다.

    # Convert TF format
    train_set = model.prepare_tf_dataset(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    validation_set = model.prepare_tf_dataset(
        tokenized_datasets["test"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    # Train & Eval
    metric_callback = KerasMetricCallback(
        metric_fn=compute_metrics, eval_dataset=validation_set
    )

    # tensorboard_callback = TensorBoard(log_dir="./tc_model_save/logs")
    # callbacks = [metric_callback, tensorboard_callback, push_to_hub_callback]
    callbacks = [metric_callback]

    model.fit(
        train_set,
        validation_data=validation_set,
        epochs=num_train_epochs,
        callbacks=callbacks,
    )

    model.save_pretrained("result/tuned_tokenizer/ner")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True, help="path/to/model")
    # parser.add_argument("--data", type=str, required=True, help="path/to/data")
    # args = parser.parse_args()
    #
    # main(args.model, args.data)
    main()
