import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load your dataset (ensure it's in the same directory or provide the full path)
df = pd.read_csv('chatbot_data.csv')

# Ensure you have the correct column names in your dataset
assert 'user_input' in df.columns and 'assistant_response' in df.columns

# List of possible responses
responses = df['assistant_response'].unique().tolist()
user_inputs = df['user_input'].tolist()
assistant_responses = df['assistant_response'].tolist()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocessing function to convert responses to labels
def preprocess_function(examples):
    inputs = tokenizer(examples['user_input'], truncation=True, padding='max_length', max_length=128)
    # Map 'assistant_response' to its corresponding index in the 'responses' list
    inputs['labels'] = [responses.index(resp) for resp in examples['assistant_response']]
    return inputs

# Convert DataFrame to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Train-test split
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

# Preprocess the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(responses))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./mental_health_model')
tokenizer.save_pretrained('./mental_health_model')
