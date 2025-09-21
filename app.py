import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import logging
from dataclasses import dataclass
from typing import Any, Dict, MutableMapping
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import IntervalStrategy
from transformers.trainer_utils import EvaluationStrategy


MODEL_ID = "distilbert-base-uncased"
OUTPUT_DIR = "distilbert-finetuned-imdb"
LABELS = ("NEGATIVE", "POSITIVE")
ID2LABEL = {index: label for index, label in enumerate(LABELS)}
LABEL2ID = {label: index for index, label in ID2LABEL.items()}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def _resolve_interval_strategy(name: str):
    #Return the correct interval/evaluation strategy token for the installed transformers

    for enum in (IntervalStrategy, EvaluationStrategy):
        if enum is not None and hasattr(enum, name.upper()):
            return getattr(enum, name.upper())
    return name


def _strategy_to_name(value: Any) -> str:
    #Normalize enum/string values to their lowercase string representation

    if value is None:
        return "no"
    if isinstance(value, str):
        return value.lower()
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value.lower()
    return str(value).lower()

def prepare_dataset(seed: int = 42) -> DatasetDict:
    #Load and shrink the IMDb dataset to manageable train/val/test splits
    LOGGER.info("Loading the IMDb dataset…")
    dataset = load_dataset("imdb")

    LOGGER.info("Creating small train/validation/test splits…")
    shuffled_train = dataset["train"].shuffle(seed=seed)
    train_valid_split = shuffled_train.train_test_split(test_size=200, seed=seed)

    def clamp_select(ds, num_samples: int):
        return ds.select(range(min(len(ds), num_samples)))

    small_train = clamp_select(train_valid_split["train"], 1000)
    validation = train_valid_split["test"]
    test = dataset["test"].shuffle(seed=seed).select(range(min(len(dataset["test"]), 200)))

    small_dataset = DatasetDict({
        "train": small_train,
        "validation": validation,
        "test": test,
    })
    LOGGER.info(
        "Dataset sizes -> train: %s, validation: %s, test: %s",
        len(small_dataset["train"]),
        len(small_dataset["validation"]),
        len(small_dataset["test"]),
    )
    return small_dataset


@dataclass
class TokenizedDataset:
    dataset: DatasetDict
    tokenizer_name: str


def tokenize_dataset(dataset: DatasetDict, tokenizer_name: str = MODEL_ID) -> TokenizedDataset:
    #Tokenize all splits using the provided tokenizer
    LOGGER.info("Loading tokenizer '%s'…", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def tokenize_batch(batch: Dict[str, list]) -> Dict[str, list]:
        return tokenizer(batch["text"], truncation=True)

    LOGGER.info("Tokenizing dataset…")
    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    return TokenizedDataset(dataset=tokenized, tokenizer_name=tokenizer_name)


def compute_metrics(predictions) -> Dict[str, float]:
    logits = predictions.predictions
    labels = predictions.label_ids

    if np is not None:
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).astype(np.float32).mean().item()
        return {"accuracy": float(accuracy)}

    try:  #exercised only when numpy is unavailable at runtime.
        import torch
    except ImportError as exc:  #surfaced as a clear runtime error.
        raise RuntimeError(
            "Computing metrics requires either numpy or torch to be installed."
        ) from exc

    logits_tensor = torch.tensor(logits)
    labels_tensor = torch.tensor(labels)
    preds = torch.argmax(logits_tensor, dim=-1)
    accuracy = (preds == labels_tensor).float().mean().item()
    return {"accuracy": float(accuracy)}

def _populate_optional_training_args_fields(
    base_kwargs: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    #Populate compatibility kwargs based on the installed transformers version

    fields = getattr(TrainingArguments, "__dataclass_fields__", {})

    evaluation_field = next(
        (candidate for candidate in ("evaluation_strategy", "eval_strategy") if candidate in fields),
        None,
    )
    evaluation_value: Any = None

    if evaluation_field is not None:
        evaluation_value = base_kwargs.setdefault(
            evaluation_field,
            _resolve_interval_strategy("epoch"),
        )
    elif "evaluate_during_training" in fields:
        base_kwargs.setdefault("evaluate_during_training", True)

    evaluation_name = _strategy_to_name(evaluation_value)

    if "save_strategy" in fields:
        if evaluation_name in {"epoch", "steps"}:
            save_default = _resolve_interval_strategy(evaluation_name)
        else:
            save_default = _resolve_interval_strategy("no")
        base_kwargs.setdefault("save_strategy", save_default)

    if "load_best_model_at_end" in fields and evaluation_name in {"epoch", "steps"}:
        base_kwargs.setdefault("load_best_model_at_end", True)

        if "metric_for_best_model" in fields:
            base_kwargs.setdefault("metric_for_best_model", "accuracy")

        if "greater_is_better" in fields:
            base_kwargs.setdefault("greater_is_better", True)

    if "report_to" in fields:
        #Support both string and list inputs.
        base_kwargs.setdefault("report_to", "none")

    return base_kwargs


def train_and_evaluate(tokenized_dataset: TokenizedDataset) -> Dict[str, float]:
    LOGGER.info("Loading model '%s'…", MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenized_dataset.tokenizer_name, use_fast=True)
    if DataCollatorWithPadding is not None:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:  #this path is only taken for very old transformers releases.
        try:
            import torch
        except ImportError as exc:  #surfaced as a runtime error for clarity.
            raise RuntimeError(
                "DataCollatorWithPadding is unavailable and PyTorch could not be imported."
            ) from exc

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        def data_collator(features):
            batch = {}
            label_keys = [key for key in ("label", "labels") if key in features[0]]

            for key in features[0].keys():
                values = [feature[key] for feature in features]
                if key in label_keys:
                    batch[key] = torch.tensor(values)
                    continue

                if isinstance(values[0], list):
                    max_length = max(len(v) for v in values)
                    padded = [v + [pad_token_id] * (max_length - len(v)) for v in values]
                    batch[key] = torch.tensor(padded)
                else:
                    batch[key] = torch.tensor(values)
            return batch

    LOGGER.info("Configuring training arguments…")
    training_args_kwargs = _populate_optional_training_args_fields(
        {
            "output_dir": OUTPUT_DIR,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "logging_steps": 10,
            "seed": 42,
        }
    )

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset.dataset["train"],
        eval_dataset=tokenized_dataset.dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    LOGGER.info("Starting training…")
    trainer.train()

    LOGGER.info("Saving the fine-tuned model and tokenizer to '%s'…", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    LOGGER.info("Evaluating on the test split…")
    metrics = trainer.evaluate(tokenized_dataset.dataset["test"])
    LOGGER.info("Evaluation metrics: %s", metrics)
    return metrics

def main() -> None:
    dataset = prepare_dataset()
    tokenized = tokenize_dataset(dataset)
    metrics = train_and_evaluate(tokenized)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()