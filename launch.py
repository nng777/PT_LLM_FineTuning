import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_dir = "distilbert-finetuned-imdb"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

clf = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print(clf("What a nice movie!"))
