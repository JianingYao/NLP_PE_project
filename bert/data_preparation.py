# data_preparation.py
from datasets import load_dataset
from transformers import BertTokenizer

def load_and_preprocess_data(tokenizer_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    
    dataset = load_dataset('imdb')
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

#if __name__ == "__main__":
    #data = load_and_preprocess_data()
    #data.save_to_disk('tokenized_imdb')  # Save the preprocessed data for later use
