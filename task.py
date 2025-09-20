"""

Task Objective
Write a Python script that fine-tunes a pre-trained model to classify text as having positive or negative sentiment.


1. Find a small (e.g., 1000 training samples, 200 validation samples, and 200 test samples), publicly available dataset for sentiment analysis. A great option is the imdb dataset, which is readily available through the Hugging Face datasets library.
2. Select distilbert-base-uncased pre-trained model that is well-suited for text classification.
3. Python script should perform the following actions:
  3.1. Load Dependencies: Import necessary libraries (transformers, datasets, torch, etc.).
  3.2. Load Dataset: Load the chosen sentiment analysis dataset. Select a subset of the data for training, validation, and testing.
  3.3. Load Model and Tokenizer: Load the distilbert-base-uncased model and its corresponding tokenizer.
  3.4. Preprocess and Tokenize Data: Write a function to tokenize the text data from the dataset, preparing it for the model.
  3.5. Define Training Arguments: Set up the TrainingArguments for the Hugging Face Trainer. This includes parameters like the output directory, number of epochs, batch size, and evaluation strategy.
  3.6. Instantiate the Trainer: Create an instance of the Trainer, providing it with your model, training arguments, training dataset, and validation dataset.
  3.7. Fine-Tune the Model: Call the train() method on your Trainer instance to start the fine-tuning process.
  3.8. Evaluate the Model: After training is complete, call the evaluate() method to measure the performance of your fine-tuned model on the test set.
4. Create a short REPORT.md file that includes:
  4.1. The name of the dataset and model which are used.
  4.2. The final evaluation results (e.g., accuracy, loss) on the test set.
  4.3. A brief discussion of challenges faced and what should be learned during the process.

"""