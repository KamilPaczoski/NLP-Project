import pandas as pd
import numpy as np
from transformers import AutoModel, BertTokenizerFast
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from TextCleaner import TextMaid


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
epochs = 10
batch_size = 64
pad_len = 18

with open('train.txt', 'r') as file:
    data = file.readlines()
data = [line.strip().split(';') for line in data]
df_train = pd.DataFrame(data, columns=['sentence', 'label'])
df_train = TextMaid(df_train).cleaned_text

train_text = df_train['sentence']
train_labels = df_train['label']

with open('test.txt', 'r') as file:
    data = file.readlines()
data = [line.strip().split(';') for line in data]
df_test = pd.DataFrame(data, columns=['sentence', 'label'])
df_test = TextMaid(df_test).cleaned_text

test_text = df_test['sentence']
test_labels = df_test['label']

with open('val.txt', 'r') as file:
    data = file.readlines()
data = [line.strip().split(';') for line in data]
df_val = pd.DataFrame(data, columns=['sentence', 'label'])
df_val = TextMaid(df_val).cleaned_text

val_text = df_val['sentence']
val_labels = df_val['label']

bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')



tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length=pad_len,
    padding='longest',
    truncation=True
)

tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length=pad_len,
    padding='longest',
    truncation=True
)

tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length=pad_len,
    padding='longest',
    truncation=True
)

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
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
for param in bert.parameters():
    param.requires_grad = True #true unfreezes the bert model


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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # learning rate
class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(train_labels),
                                     y=train_labels
                                     )

weights = torch.tensor(class_weights, dtype=torch.float)
weights = weights.to(device)
cross_entropy = nn.NLLLoss(weight=weights)
def train():
    model.train()
    total_loss, _ = 0, 0
    total_preds = []
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
        preds = preds.detach().cpu().numpy()  # znowu gpu się buntuje
        total_preds.append(preds)
    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


def evaluate():
    print("\nEvaluating...")
    model.eval()
    total_loss, total_accuracy = 0, 0
    total_preds = []
    for step, batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
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


best_valid_loss = float('inf')
train_losses = []
valid_losses = []
for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    train_loss, _ = train()
    valid_loss, _ = evaluate()
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print('\nTraining Loss: {}'.format(train_loss))
    print('Validation Loss: {}'.format(valid_loss))
path = 'saved_weights.pt'
model.load_state_dict(torch.load(path))
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()
pred = np.argmax(preds, axis=1)
print(classification_report(test_y, pred))
print("train labels: ", train_labels.unique())
print("accuracy:", (pred == test_y.numpy()).mean())

