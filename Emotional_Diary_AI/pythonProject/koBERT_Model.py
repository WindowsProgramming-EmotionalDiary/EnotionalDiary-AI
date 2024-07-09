import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

excel_file = '/data/한국어_단발성_대화_데이터셋.xlsx'
chatbot_data = pd.read_excel(excel_file)

emotion_dict = {
    "공포": 0,
    "놀람": 1,
    "분노": 2,
    "슬픔": 3,
    "중립": 4,
    "행복": 5,
    "혐오": 6
}
chatbot_data['Emotion'] = chatbot_data['Emotion'].apply(lambda x: emotion_dict.get(x))

data_list = []
for q, label in zip(chatbot_data['Sentence'], chatbot_data['Emotion']):
    data_list.append([q, label])

dataset_train, dataset_test = train_test_split(data_list, test_size=0.25, random_state=0)

class BERTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=64):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentence, label = self.dataset[idx]
        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        return input_ids, attention_mask, label

class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=7, dr_rate=0.5):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=dr_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bertmodel = BertModel.from_pretrained('bert-base-multilingual-cased')

train_dataset = BERTDataset(dataset_train, tokenizer)
test_dataset = BERTDataset(dataset_test, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = BERTClassifier(bertmodel).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 10
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_dataloader) * num_epochs)

def calc_accuracy(preds, labels):
    pred_class = preds.argmax(dim=1)
    accuracy = (pred_class == labels).float().mean()
    return accuracy

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for batch in tqdm(train_dataloader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        train_acc += calc_accuracy(outputs, labels)

    avg_train_loss = train_loss / len(train_dataloader)
    avg_train_acc = train_acc / len(train_dataloader)

    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}')

    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            eval_loss += loss.item()
            eval_acc += calc_accuracy(outputs, labels)

    avg_eval_loss = eval_loss / len(test_dataloader)
    avg_eval_acc = eval_acc / len(test_dataloader)

torch.save(model.state_dict(), 'bert_emotion_classification2.pth')
print("Model saved.")
