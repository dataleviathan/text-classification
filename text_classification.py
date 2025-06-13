# Installing Required Hugging Face Libraries

!pip install evaluate
!pip install datasets

# Setting Up Environment

from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import DataCollatorWithPadding

# Prevent WandB from interfering
import os
os.environ["WANDB_DISABLED"] = "true"

# Uploading Local Files to Google Colab

from google.colab import files
uploaded_1 = files.upload()
uploaded_2 = files.upload()
uploaded_3 = files.upload()

# Converting Excel Data to Hugging Face Datasets

# Load Data
dataset_dict_training = pd.read_excel('training_data.xlsx')
dataset_dict_testing = pd.read_excel('testing_data.xlsx')
dataset_dict_validating = pd.read_excel('validating_data.xlsx')

# Convert DataFrames to Hugging Face Datasets
dataset_dict_training = Dataset.from_pandas(dataset_dict_training.rename(columns={"Label": "label"}))
dataset_dict_testing = Dataset.from_pandas(dataset_dict_testing.rename(columns={"Label": "label"}))
dataset_dict_validating = Dataset.from_pandas(dataset_dict_validating.rename(columns={"Label": "label"}))

print(dataset_dict_training)
print(dataset_dict_testing)
print(dataset_dict_validating)

'''
Dataset({
    features: ['Links', 'label'],
    num_rows: 2100
})
Dataset({
    features: ['Links', 'label'],
    num_rows: 450
})
Dataset({
    features: ['Links', 'label'],
    num_rows: 450
})
'''

# Load Pre-Trained Model

# Define pre-trained model path
model_path = "google-bert/bert-base-uncased"

# Load model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model with binary classification head
id2label = {0: "Safe", 1: "Not Safe"}
label2id = {"Safe": 0, "Not Safe": 1}
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# Set Trainable Parameters

# Freeze all base model parameters
for name, param in model.base_model.named_parameters():
    param.requires_grad = False

# Unfreeze base model pooling layers
for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True

# Combine datasets into a DatasetDict

dataset_dict = DatasetDict({
    "train": dataset_dict_training,
    "test": dataset_dict_testing,
    "validation": dataset_dict_validating
})

# Define text preprocessing
def preprocess_function(examples):
    return tokenizer(examples["Links"], truncation=True, padding=True)

# Apply preprocessing to each split
tokenized_data = dataset_dict.map(preprocess_function, batched=True)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define Evaluation Metrics

from scipy.special import softmax

accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # softmax for probabilities
    probabilities = softmax(predictions, axis=1)
    positive_class_probs = probabilities[:, 1]

    # Compute AUC
    auc = np.round(auc_score.compute(
        prediction_scores=positive_class_probs,
        references=labels
    )['roc_auc'], 3)

    # Compute Accuracy
    predicted_classes = np.argmax(predictions, axis=1)
    acc = np.round(accuracy.compute(
        predictions=predicted_classes,
        references=labels
    )['accuracy'], 3)

    return {"Accuracy": acc, "AUC": auc}

# Set Hyperparameters

lr = 2e-4
batch_size = 8
num_epochs = 10

training_args = TrainingArguments(
    output_dir="bert-phishing-classifier_teacher",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Fine-Tune Model

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

'''
Epoch	Training Loss	Validation Loss	Accuracy	Auc
1	0.504900	0.383961	0.833000	0.926000
2	0.406000	0.330444	0.867000	0.937000
3	0.371100	0.326133	0.873000	0.941000
4	0.365800	0.352483	0.849000	0.941000
5	0.333600	0.306263	0.887000	0.942000
6	0.326300	0.300184	0.873000	0.947000
7	0.323700	0.296756	0.878000	0.946000
8	0.327300	0.294262	0.878000	0.947000
9	0.323000	0.295207	0.884000	0.947000
10	0.313300	0.295451	0.884000	0.947000
'''

# Validation Data

# Apply model to validation dataset
predictions = trainer.predict(tokenized_data["test"])

# Extract the logits and labels from the predictions object
logits = predictions.predictions
labels = predictions.label_ids

# Use compute_metrics function
metrics = compute_metrics((logits, labels))
print(metrics)

'''
{'Accuracy': np.float64(0.871), 'AUC': np.float64(0.953)}
'''
