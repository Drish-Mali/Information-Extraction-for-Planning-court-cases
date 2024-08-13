from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, Features, Value, Array2D
from sklearn.model_selection import train_test_split
from datasets import load_dataset, load_metric
from transformers import TFAutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import evaluate
from transformers import AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments,pipeline
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
import os
from transformers import EarlyStoppingCallback
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import TrainerCallback
from accelerate import Accelerator

import os
task = "ner"  # Should be one of "ner", "pos" or "chunk"
model_checkpoint = "google/bigbird-roberta-base"
label_all_tokens = True
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token
ner_tag_mapping = {
    'O': 0,
    'B-CITATION': 1,
    'I-CITATION': 2,
    'B-DATE': 3,
    'I-DATE': 4,
    'B-JUDGE': 5,
    'I-JUDGE': 6,
    'B-LOCATION': 7,
    'I-LOCATION': 8,
    'B-COURT': 9,
    'I-COURT': 10
    }
metric = evaluate.load('seqeval')


def compute_metrics(eval_preds):
    logits, labels = eval_preds

    predictions = np.argmax(logits, axis=-1)
    label_names=list(ner_tag_mapping.keys())

    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]

    true_predictions = [[label_names[p] for p, l in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]

    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

    # Flatten lists for precision-recall and AUC-PR computation
    flat_predictions = [item for sublist in true_predictions for item in sublist]
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    # Create a binary format for precision-recall computation
    flat_true_labels_binary = [1 if label != 'O' else 0 for label in flat_true_labels]
    flat_predictions_binary = [1 if label != 'O' else 0 for label in flat_predictions]

    precision, recall, _ = precision_recall_curve(flat_true_labels_binary, flat_predictions_binary)
    auc_pr = auc(recall, precision)

    return {
        "precision": all_metrics['overall_precision'],
        "recall": all_metrics['overall_recall'],
        "f1": all_metrics['overall_f1'],
        "accuracy": all_metrics['overall_accuracy'],
        "auc_pr": auc_pr
    }


def convert_to_dataset(tsv_file):

    data = []
    current_tokens = []
    current_ner_tags = []
  
    with open(tsv_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                # Check for start of new entry
                if line.startswith("-DOCSTART- -X-"):
                    if current_tokens:  # If there are tokens in the current entry
                        # Append current entry to the data list
                        data.append({
                            'id': str(len(data)),
                            'tokens': current_tokens,
                            'ner_tags': current_ner_tags
                        })
                        # Reset current tokens and ner_tags lists for the next entry
                        current_tokens = []
                        current_ner_tags = []
                elif line:  # If line is not empty
                    parts = line.split("\t")
                    token_text = parts[0]
                    ner_tag = parts[-1]
                    # Convert NER tag label to numerical value using the mapping
                    ner_tag_num = ner_tag_mapping.get(ner_tag, 0)  # Default to 0 if label not found
                    # Append token and numerical ner_tag to current entry
                    current_tokens.append(token_text)
                    current_ner_tags.append(ner_tag_num)
    if current_tokens:
         data.append({
            'id': str(len(data)),
            'tokens': current_tokens,
            'ner_tags': current_ner_tags
        })
    # Append the last entry to the data list (if any)
    return Dataset.from_dict({
        'id': [entry['id'] for entry in data],
        'tokens': [entry['tokens'] for entry in data],
        'ner_tags': [entry['ner_tags'] for entry in data],
    })

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, max_length=2048,padding='max_length',is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
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

def create_data_dict(tsv_file):
     # Convert TSV data to dataset
    dataset = convert_to_dataset(tsv_file)

    # Set the random seed for reproducibility
    random_seed = 42

    # Split the dataset into train (70%), test (15%), and validation (15%)
    train_test_ratio = 0.70
    val_test_ratio = 0.15 / (0.15 + 0.15)

    train_dataset, temp_dataset = dataset.train_test_split(test_size=(1 - train_test_ratio), seed=random_seed).values()
    val_dataset, test_dataset = temp_dataset.train_test_split(test_size=val_test_ratio, seed=random_seed).values()

    # Combine datasets into a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    tokenized_train_datasets = dataset_dict['train'].map(tokenize_and_align_labels, batched=True)
    tokenized_val_datasets = dataset_dict['validation'].map(tokenize_and_align_labels, batched=True)
    tokenized_test_datasets = dataset_dict['test'].map(tokenize_and_align_labels, batched=True)
    return tokenized_train_datasets,tokenized_val_datasets,tokenized_test_datasets


def model(tsv_file):
    accelerator = Accelerator()
    tokenized_train_datasets,tokenized_val_datasets,tokenized_test_datasets=create_data_dict(tsv_file)
    label_list=list(ner_tag_mapping.keys())
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    # model = TFAutoModelForTokenClassification.from_pretrained(
    #     model_checkpoint, num_labels=len(label_list), id2label=id2label, label2id=label2id)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = evaluate.load('seqeval')
    label_names=label_list
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=30)
    checkpoint_dir = os.path.join('/home/results_ner_big_bird', model_checkpoint.replace("/", "-"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = AutoModelForTokenClassification.from_pretrained(
                                                    model_checkpoint,
                                                    id2label=id2label,
                                                    label2id=label2id)
    model, data_collator = accelerator.prepare(model, data_collator)
    args = TrainingArguments(output_dir=checkpoint_dir,
                         evaluation_strategy = "epoch",
                         save_strategy="epoch",
                         learning_rate = 1e-5,
                         num_train_epochs=200,
                         weight_decay=0.01,
                         save_total_limit=2,
                         per_device_train_batch_size=16,
                         per_device_eval_batch_size=16,
                         lr_scheduler_type="cosine",
                         warmup_ratio=0.1,
                         load_best_model_at_end=True )# Metric to monitor (adjust as needed)
                           # Save the model with the lowest monitored metric ) # Only save the best model)
    trainer = Trainer(model=model,
                  args=args,
                  train_dataset = tokenized_train_datasets,
                  eval_dataset = tokenized_val_datasets,
                  data_collator=data_collator,
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer,
                  callbacks=[early_stopping_callback])

    trainer.train()
    test_results = trainer.evaluate(tokenized_test_datasets)

    # Print the test results
    print(f"Test set results: {test_results}")
    # Add the method name to the results
    method_name = "legal-bert-base-uncased"

    # Create a DataFrame from the test results
    results_df = pd.DataFrame([test_results])

    # Add the method name to the DataFrame
    results_df['method'] = model_checkpoint

    # Save the DataFrame to a CSV file
    result_csv_name= model_checkpoint.replace("/", "_") + '_test_results.csv'
    result_csv_name = './data/results_ner_big_bird/' + result_csv_name

    results_df.to_csv(result_csv_name, index=False)

    # Print the DataFrame
    print(results_df)
    predictions, labels, _ = trainer.predict(tokenized_test_datasets)

    # Convert predictions and labels to a flat list of tags
    true_labels = [label_names[label] for label_list in labels for label in label_list if label != -100]
    predicted_labels = [label_names[np.argmax(pred)] for pred_list, label_list in zip(predictions, labels) for pred, label in zip(pred_list, label_list) if label != -100]
    report = classification_report(true_labels, predicted_labels, labels=label_list)
    print("Classification Report:\n", report)
    # Splitting the classification report into lines
    report_lines = report.split('\n')

    # Extracting headers and data
    headers = ['class'] + report_lines[0].split()
    data = [line.split() for line in report_lines[2:-5]]  # Exclude first and last few lines

    # Creating a DataFrame
    try:
        df = pd.DataFrame(data, columns=headers)
        print(df)
        report_csv_name = model_checkpoint.replace("/", "_") + '_classification_report.csv'
        report_csv_path = '/home/results_ner_big_bird/' + report_csv_name
        df.to_csv(report_csv_path, index=False)
        print(f"Classification report saved to {report_csv_path}")
    except ValueError as e:
        print(f"Error occurred while creating DataFrame: {e}")
        print("Headers:", headers)
        print("Data:", data)
        # Ensure the lengths of losses and epochs are the same


def eval():
    checkpoint = "/content/nlpaueb/legal-bert-base-uncased/"
    token_classifier = pipeline(
        "token-classification", model=checkpoint, aggregation_strategy="simple"
    )
    sample_sentence = "Neutral Citation Number: [2020] EWHC 872 (Admin) Case No: CO/4505/2019 IN THE HIGH COURT OF JUSTICE QUEEN'S BENCH DIVISION PLANNING COURT IN BIRMINGHAM Birmingham Civil Justice Centre 33 Bull Street, Birmingham Date: 09/04/2020 Before: THE HONOURABLE MRS JUSTICE ANDREWS DBE Between: SOUTH DERBYSHIRE DISTRICT COUNCIL Claimant - and - SECRETARY OF STATE FOR HOUSING COMMUNITIES AND LOCAL GOVERNMENT -and- Defendant D COOPER CONSTRUCTION LTD Interested Party Timothy Jones (instructed by Geldards LLP) for the Claimant Killian Garvey (instructed by the Government Legal Department) for the Defendant Christian Hawley instructed by Howes Percival LLP for the Interested Party (written submissions only) Hearing dates: 31 May 2022 Approved Judgment Covid-19 Protocol: This judgment was handed down by the Judge remotely by circulation to the parties’ representatives by email and release to BAILII. The date and time for hand-down is deemed to be 10 am on Thursday 9th April 2020. The deemed hearing has been adjourned to enable the parties to make written submissions on any consequential matters by 4.30pm on Friday 17 April 2020 and for the order on judgment to be drawn up."
    output=token_classifier(sample_sentence)
    # Evaluate the model on the test dataset


if __name__=="__main__":
    
    tsv_file = "./data/output-iob.tsv"
    
    # Assuming model is a function that takes the full file path as an argument
    model(tsv_file)