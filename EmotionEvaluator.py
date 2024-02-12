import torch
import numpy as np
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader
from TextCleaner import TextCleaner
from main import BERT_layers, tokenizer, bert, device, train_labels

model = BERT_layers(bert)
model = model.to(device)

model.load_state_dict(torch.load('saved_weights.pt', map_location=device))
model.eval()
labels = train_labels.unique()

def classify_emotion(text):
    cleaned_text = TextCleaner(text).cleaned_text
    tokenized_text = tokenizer.encode_plus(
        cleaned_text,
        max_length=18,
        padding='longest',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = tokenized_text['input_ids'].to(device)
    attention_mask = tokenized_text['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = int(np.argmax(probabilities.cpu().numpy()))

    return predicted_class


def main():
    while True:
        user_input = input("Enter text (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        emotion_class = classify_emotion(user_input)

        print("Predicted emotion class:", emotion_class, type(emotion_class))
        print("Predicted emotion type:", labels[emotion_class], labels)


def evaluator(text):
    emotion_class = classify_emotion(text)
    return labels[emotion_class]


if __name__ == "__main__":
    main()
