# predict.py
import torch
from model import CustomBertModel

def load_model(path):
    model = CustomBertModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs)
    return torch.argmax(logits, dim=-1)

# Example usage
model = load_model('path_to_model.pth')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(predict(model, tokenizer, "This movie is fantastic!"))
