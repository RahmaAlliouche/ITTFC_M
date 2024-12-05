import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os
import numpy as np

data_path = "c:/Users/rahma/Documents/S2/ITFFC_M/model/text_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"CSV file not found at {data_path}")

data = pd.read_csv(data_path, on_bad_lines='warn')


if 'text' not in data.columns or 'label' not in data.columns:
    raise ValueError("The dataset must contain 'text' and 'label' columns.")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['text'].tolist(), data['label'].tolist(), test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")




class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, texts):
        self.encodings = encodings
        self.labels = labels
        self.texts = texts  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['text'] = self.texts[idx]  
        return item

train_dataset = TextDataset(train_encodings, train_labels, train_texts)
val_dataset = TextDataset(val_encodings, val_labels, val_texts)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics 
)

def get_uncertain_predictions(model, dataset, threshold=0.6):
    model.eval()
    uncertainties = []
    texts = []
    with torch.no_grad():
        for item in dataset:
            inputs = {key: val.unsqueeze(0) for key, val in item.items() if key != 'labels'}
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            max_prob, pred = torch.max(probs, dim=-1)
            uncertainties.append(max_prob.item())
            texts.append(item['text'])  

    uncertain_indices = [i for i, prob in enumerate(uncertainties) if prob < threshold]
    return [(texts[i], dataset[i]['labels']) for i in uncertain_indices]

#  Learning Loop
for iteration in range(3):  
    print(f"Training Iteration: {iteration + 1}")
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    uncertain_data = get_uncertain_predictions(model, val_dataset)
    
    if uncertain_data:
        print("Uncertain Predictions:")
        for text, true_label in uncertain_data:
            print(f"Text: {text}, True Label: {true_label}")
            
            corrected_label = int(input("Please enter the correct label (0 for human, 1 for AI): "))
            train_texts.append(text)
            train_labels.append(corrected_label)
            
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        train_dataset = TextDataset(train_encodings, train_labels, train_texts)

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Training complete! Model and tokenizer saved.")
