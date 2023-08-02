"""
Reference: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb#scrollTo=P_qHpojde5gp
"""
from datasets import load_dataset
import evaluate
from transformers import ElectraTokenizer, TFAutoModelForSequenceClassification, create_optimizer
from transformers.keras_callbacks import KerasMetricCallback
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import argparse


metric = evaluate.load("accuracy")
tokenizer = ElectraTokenizer.from_pretrained("model")


def preprocess_function(examples):
    return tokenizer(examples["question1"], examples["question2"], truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    # Load
    dataset = load_dataset("kor_qpair")

    # Encode Data
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset = encoded_dataset.remove_columns(["question1", "question2"])  # unuse column
    encoded_dataset = encoded_dataset.rename_column("is_duplicate", "labels")  # model expects 'labels'

    # Create Optimizer
    batch_size = 32
    num_epochs = 2  # 3부터 오버피팅
    batches_per_epoch = len(encoded_dataset["train"]) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

    # Build Model & Compile
    model = TFAutoModelForSequenceClassification.from_pretrained("model", num_labels=2)
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

    # Make callbacks
    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_dataset)
    # tensorboard_callback = TensorBoard(log_dir=f"{model_ckpt}/logs/nsmc")
    # callbacks = [metric_callback, tensorboard_callback]
    callbacks = [metric_callback]

    # Train
    model.fit(
        tf_train_dataset,
        validation_data=tf_validation_dataset,
        epochs=num_epochs,
        callbacks=callbacks,
    )

    model.save_pretrained("result/tuned_tokenizer/qpair")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True, help="path/to/model")
    # args = parser.parse_args()
    #
    # main(args.model)
    main()
