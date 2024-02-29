import torch
from emotion_evaluator import BERT_layers, tokenizer, bert, device
from torch.utils.data import DataLoader, TensorDataset

model = BERT_layers(bert).to(device)
model.load_state_dict(torch.load('saved_weights.pt', map_location=device))
model.eval()


def preprocess_data(sentences, emotions):
    tokenized_sentences = tokenizer.batch_encode_plus(sentences, max_length=18, padding='longest', truncation=True,
                                                      return_tensors='pt')
    return tokenized_sentences, torch.tensor(emotions)


def fine_tune_model(sentences, emotions):
    tokenized_data, encoded_emotions = preprocess_data(sentences, emotions)
    data = TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'], encoded_emotions)
    dataloader = DataLoader(data, batch_size=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.NLLLoss()
    model.train()

    for _ in range(3):
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids, attention_mask), labels)
            loss.backward()
            optimizer.step()


def update_model():
    emotion_to_int = {'sadness': 0, 'anger': 1, 'love': 2, 'surprise': 3, 'fear': 4, 'joy': 5}
    with open("datasets/feedback.txt", 'r') as file:
        sentences, emotions = zip(*[line.strip().split(';') for line in file])
    fine_tune_model(list(sentences), [emotion_to_int[e] for e in emotions])
    torch.save(model.state_dict(), 'saved_weights.pt')


if __name__ == "__main__":
    update_model()
    print("Model updated successfully")
