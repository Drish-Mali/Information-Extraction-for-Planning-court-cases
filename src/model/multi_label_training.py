import pandas as pd 
import torch
from transformers import AutoTokenizer
from transformers import  AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss
from transformers import EvalPrediction
from transformers import TrainingArguments, Trainer
import os
from transformers import EarlyStoppingCallback
from accelerate import Accelerator

# from optimum.quanto import qint8

# from quanto import quantize, freeze

model_checkpoint = "lexlms/legal-longformer-large"
def get_data(file_path):
    df=pd.read_csv(file_path)
    df['adjusted_paragraph']=df['Paragraph Number']+' '+df['paragraph']

    # Step 1: Get unique file names
    file_names = df['file_name'].unique()

    # Step 2: Shuffle the file names
    shuffled_files = pd.Series(file_names).sample(frac=1, random_state=42).tolist()

    # Step 3: Compute split indices
    train_size = int(0.7 * len(shuffled_files))
    val_size = int(0.15 * len(shuffled_files))
    test_size = len(shuffled_files) - train_size - val_size

    # Step 4: Split file names into train, validation, and test sets
    train_files = shuffled_files[:train_size]
    val_files = shuffled_files[train_size:train_size + val_size]
    test_files = shuffled_files[train_size + val_size:]

    # Step 5: Create train, validation, and test dataframes
    train_df = df[df['file_name'].isin(train_files)]
    val_df = df[df['file_name'].isin(val_files)]
    test_df = df[df['file_name'].isin(test_files)]

    # Extract texts and labels for train set
    train_texts = train_df['adjusted_paragraph'].tolist()
    train_labels = train_df[['introduction', 'fact', 'citation', 'judgment']].values

    # Extract texts and labels for validation set
    val_texts = val_df['adjusted_paragraph'].tolist()
    val_labels = val_df[['introduction', 'fact', 'citation', 'judgment']].values

    # Extract texts and labels for test set
    test_texts = test_df['adjusted_paragraph'].tolist()
    test_labels = test_df[['introduction', 'fact', 'citation', 'judgment']].values

    # Optional: Print sizes of each set to verify
    print(f"Train set: {len(train_texts)} texts, {train_labels.shape[0]} labels")
    print(f"Validation set: {len(val_texts)} texts, {val_labels.shape[0]} labels")
    print(f"Test set: {len(test_texts)} texts, {test_labels.shape[0]} labels")

    return train_texts, train_labels, val_texts, val_labels, test_texts,test_labels

# build custom dataset
class CustomDataset(Dataset):
  def __init__(self, texts, labels, tokenizer, max_len=1532):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts[idx])
    label = torch.tensor(self.labels[idx])

    encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors='pt')

    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': label
    }




def multi_labels_metrics(predictions, labels, threshold=0.3):
  sigmoid = torch.nn.Sigmoid()
  probs = sigmoid(torch.Tensor(predictions))

  y_pred = np.zeros(probs.shape)
  y_pred[np.where(probs>=threshold)] = 1
  y_true = labels

  f1 = f1_score(y_true, y_pred, average = 'macro')
  roc_auc = roc_auc_score(y_true, y_pred, average = 'macro')
  hamming = hamming_loss(y_true, y_pred)

  metrics = {
      "roc_auc": roc_auc,
      "hamming_loss": hamming,
      "f1": f1
  }

  return metrics

def compute_metrics(p:EvalPrediction):
  preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

  result = multi_labels_metrics(predictions=preds,
                                labels=p.label_ids)

  return result

def model(file_path):
    train_texts, train_labels, val_texts, val_labels, test_texts,test_labels=get_data(file_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,trust_remote_code=True,padding=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(train_labels[0]),trust_remote_code=True,
                                                            problem_type="multi_label_classification")
    model.config.pad_token_id = tokenizer.eos_token_id 
    # quantize(model, weights=qint8, activations=qint8)

    # freeze(model)
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=15)
    checkpoint_dir = os.path.join('./data/multilabel_reannotation', model_checkpoint.replace("/", "-"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    args = TrainingArguments(output_dir=checkpoint_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=30,
        evaluation_strategy = "epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True 
        
        )
    accelerator = Accelerator()
    train_dataset, val_dataset, test_dataset, model, args = accelerator.prepare(train_dataset, val_dataset, test_dataset, model, args)

    trainer = Trainer(model=model,
                    args=args,
                    train_dataset=train_dataset,
                    eval_dataset = val_dataset,
                    compute_metrics=compute_metrics,
                    callbacks=[early_stopping_callback])
    trainer.train()
    trainer.evaluate()
    test_results = trainer.evaluate(test_dataset)
     # Print the test results
    print(f"Test set results: {test_results}")

    # Create a DataFrame from the test results
    results_df = pd.DataFrame([test_results])

    # Add the method name to the DataFrame
    results_df['method'] = model_checkpoint

    # Save the DataFrame to a CSV file
    result_csv_name= model_checkpoint.replace("/", "_") + '_test_results.csv'
    result_csv_name = './data/multilabel_reannotation/' + result_csv_name

    results_df.to_csv(result_csv_name, index=False)

    # Print the DataFrame
    print(results_df)
   


if __name__=='__main__':
    file_path="./data/reannotated_multi_label_data.csv"
    model(file_path)


