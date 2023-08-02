"""
Reference: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb#scrollTo=P_qHpojde5gp
"""
from datasets import load_dataset
import evaluate
from transformers import ElectraTokenizer, TFAutoModelForSequenceClassification, create_optimizer
from transformers.keras_callbacks import KerasMetricCallback
from tensorflow.keras.callbacks import TensorBoard
import argparse


metric = evaluate.load("spearmanr")
tokenizer = ElectraTokenizer.from_pretrained("model")


def preprocess_function(examples):
    # (Debug) TypeError: TextInputSequence must be str
    examples["sentence1"] = list(map(str, examples["sentence1"]))
    examples["sentence2"] = list(map(str, examples["sentence2"]))

    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # predictions = np.argmax(logits, axis=0)
    # predictions = list(map(lambda x: x[0], logits))
    predictions = logits[:, 0]
    return metric.compute(predictions=predictions, references=labels)


def main():
    # Load
    dataset = load_dataset("kor_nlu", "sts")

    # Encode Data
    encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["genre", "filename", "year", "id"])
    encoded_dataset = encoded_dataset.remove_columns(["sentence1", "sentence2"])  # train에 사용하지 않는 column 제거
    encoded_dataset = encoded_dataset.rename_column("score", "label")

    # Create Optimizer
    batch_size = 8
    num_epochs = 5
    batches_per_epoch = len(encoded_dataset["train"]) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

    # Build Model & Compile
    model = TFAutoModelForSequenceClassification.from_pretrained("model", num_labels=1)
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
    # tensorboard_callback = TensorBoard(log_dir=f"{model_ckpt}/logs/nsmc")
    # callbacks = [metric_callback, tensorboard_callback]
    callbacks = [metric_callback]

    model.fit(
        tf_train_dataset,
        validation_data=tf_validation_dataset,
        epochs=num_epochs,
        callbacks=callbacks,
    )

    model.save_pretrained("result/tuned_tokenizer/sts")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True, help="path/to/model")
    # args = parser.parse_args()
    #
    # main(args.model)
    main()
