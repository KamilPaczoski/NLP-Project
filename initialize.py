import pandas as pd
import numpy as np
from transformers import AutoModel, BertTokenizerFast
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from text_cleaner import TextCleaner

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
epochs = 40
batch_size = 64
pad_len = 60


def load_data(file_name):
    with open(file_name, 'r') as file:
        data = file.readlines()
    data = [line.strip().split(';') for line in data]
    df = pd.DataFrame(data, columns=['sentence', 'label'])
    df = TextCleaner(df).cleaned_text
    return df['sentence'], df['label']


train_text, train_labels = load_data('datasets/train.txt')
test_text, test_labels = load_data('datasets/test.txt')
val_text, val_labels = load_data('datasets/val.txt')

bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def tokenize_data(text):
    return tokenizer.batch_encode_plus(
        text.tolist(),
        max_length=pad_len,
        padding='longest',
        truncation=True
    )


tokens_train = tokenize_data(train_text)
tokens_val = tokenize_data(val_text)
tokens_test = tokenize_data(test_text)

label_encoder = LabelEncoder()

train_y = torch.tensor(label_encoder.fit_transform(train_labels.tolist()))
val_y = torch.tensor(label_encoder.fit_transform(val_labels.tolist()))
test_y = torch.tensor(label_encoder.fit_transform(test_labels.tolist()))

train_seq = torch.tensor(tokens_train['input_ids'])
val_seq = torch.tensor(tokens_val['input_ids'])
test_seq = torch.tensor(tokens_test['input_ids'])

train_mask = torch.tensor(tokens_train['attention_mask'])
val_mask = torch.tensor(tokens_val['attention_mask'])
test_mask = torch.tensor(tokens_test['attention_mask'])

train_data = TensorDataset(train_seq, train_mask, train_y)
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
val_data = TensorDataset(val_seq, val_mask, val_y)
val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

for param in bert.parameters():
    param.requires_grad = True


class BERT_layers(nn.Module):
    def __init__(self, bert):
        super(BERT_layers, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 6)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


model = BERT_layers(bert)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
cross_entropy = nn.NLLLoss(weight=torch.tensor(compute_class_weight(class_weight="balanced",
                                                                    classes=np.unique(train_labels),
                                                                    y=train_labels
                                                                    ), dtype=torch.float).to(device))


def train():
    model.train()
    total_loss, total_preds = 0, []
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and step != 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        model.zero_grad()
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)
    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


def evaluate():
    print("\nEvaluating...")
    model.eval()
    total_loss, total_preds = 0, []
    for step, batch in enumerate(val_dataloader):
        if step % 50 == 0 and step != 0:
            print('Batch {:>5,} of {:>5,}.'.format(step, len(val_dataloader)))
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
    avg_loss = total_loss / len(val_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


# sequence_lengths = [len(sequence) for sequence in df_train['sentence'].tolist()]
# average_length = sum(sequence_lengths) / len(sequence_lengths) # to get the average length of the sequences ergo padding length

if __name__ == "__main__":
    best_valid_loss = float('inf')
    for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        train_loss, _ = train()
        valid_loss, _ = evaluate()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')
        print('\nTraining Loss: {}'.format(train_loss))
        print('Validation Loss: {}'.format(valid_loss))
    model.load_state_dict(torch.load('saved_weights.pt'))
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
    pred = np.argmax(preds, axis=1)
    print(classification_report(test_y, pred))
    print("train labels: ", train_labels.unique())
    print("accuracy:", (pred == test_y.numpy()).mean())
