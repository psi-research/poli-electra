"""
Reference: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb#scrollTo=P_qHpojde5gp
"""
from datasets import load_dataset
import evaluate
from transformers import ElectraTokenizer, TFAutoModelForSequenceClassification, create_optimizer
from transformers.keras_callbacks import KerasMetricCallback
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import argparse
import os


metric = evaluate.load("accuracy")
tokenizer = ElectraTokenizer.from_pretrained("model")


def check_tuned_tokenizer():
    text = "자오바반몐산업경젱력이수됭"
    tokens = tokenizer(text)["input_ids"]
    print(tokens)
    if len(tokens) - 2 == 1:
        raise ValueError("Tokenizer Not Tuned")
    else:
        print("This Tokenizer Tuned!")


def preprocess_function(examples):
    return tokenizer(examples["document"], truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    # Load Data
    dataset = load_dataset("nsmc")

    # Encode Data
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # Create Optimizer
    batch_size = 32
    num_epochs = 2
    batches_per_epoch = len(encoded_dataset["train"]) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

    # Build Model & Compile
    id2label = {0: "Neg", 1: "Pos"}
    label2id = {val: key for key, val in id2label.items()}
    model = TFAutoModelForSequenceClassification.from_pretrained("model", num_labels=2, id2label=id2label, label2id=label2id)
    model.compile(optimizer=optimizer)

    # Preprocess Data
    tf_train_dataset = model.prepare_tf_dataset(
        encoded_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        tokenizer=tokenizer
    )

    tf_validation_dataset = model.prepare_tf_dataset(
        encoded_dataset["test"],
        shuffle=False,
        batch_size=batch_size,
        tokenizer=tokenizer,
    )

    # Train
    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_dataset)
    # filepath = os.path.join(model_ckpt, "results", "nsmc", "nsmc-epoch{epoch:02d}-acc{accuracy:.4f}")
    # cp_callback = ModelCheckpoint(filepath=filepath, save_weights_only=True, verbose=1)
    # tensorboard_callback = TensorBoard(log_dir=f"{model_ckpt}/logs/nsmc")
    # callbacks = [metric_callback, tensorboard_callback]
    callbacks = [metric_callback]

    model.fit(
        tf_train_dataset,
        validation_data=tf_validation_dataset,
        epochs=num_epochs,
        callbacks=callbacks,
    )

    model.save_pretrained("result/tuned_tokenizer/nsmc")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True, help="path/to/model")
    # args = parser.parse_args()
    #
    # main(args.model)
    # check_tuned_tokenizer()
    main()
