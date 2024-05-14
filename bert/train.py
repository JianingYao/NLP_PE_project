# train.py
import torch
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
from model import CustomBertModel
from utils import compute_metrics  # Make sure to import compute_metrics
from data_preparation import load_and_preprocess_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_embedding_type = 'sinusoidal'  # Change to 'alibi' or 'rope' as needed

    # Load data
    datasets = load_and_preprocess_data()
    datasets = datasets.map(lambda examples: {'labels': examples['label']}, batched=True)  # Ensure label field is added
    
    model = CustomBertModel(pos_embedding_type=pos_embedding_type).to(device)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",  # Ensure evaluation happens at least every epoch
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['test'],
        compute_metrics=compute_metrics  # Add the metric computation
    )
    
    # Train and evaluate
    trainer.train()

if __name__ == "__main__":
    main()
