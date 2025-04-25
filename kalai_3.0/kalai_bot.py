# kalai_bot.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math, random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ------------ Tokenizer ------------
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.idx = 3

    def build_vocab(self, lines):
        for line in lines:
            for word in line.strip().lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.idx
                    self.idx2word[self.idx] = word
                    self.idx += 1

    def encode(self, text, max_len=20):
        tokens = [self.word2idx.get(w, 0) for w in text.lower().split()]
        return [1] + tokens[:max_len] + [2]

    def decode(self, ids):
        return ' '.join([self.idx2word.get(i, "<UNK>") for i in ids if i not in [0, 1, 2]])

    def vocab_size(self):
        return len(self.word2idx)

# ------------ Dataset ------------
class ChatDataset(Dataset):
    def __init__(self, file, tokenizer, max_len=20):
        with open(file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.pairs = [(lines[i], lines[i+1]) for i in range(0, len(lines)-1, 2)]
        tokenizer.build_vocab([q for q, a in self.pairs] + [a for q, a in self.pairs])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, a = self.pairs[idx]
        src = self.tokenizer.encode(q, self.max_len)
        tgt = self.tokenizer.encode(a, self.max_len)
        src = torch.tensor(src + [0]*(self.max_len+2 - len(src)))
        tgt = torch.tensor(tgt + [0]*(self.max_len+2 - len(tgt)))
        return src, tgt

# ------------ Transformer Model ------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerBot(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.pos(self.embed(src))
        tgt = self.pos(self.embed(tgt))
        memory = self.encoder(src.permute(1, 0, 2))
        out = self.decoder(tgt.permute(1, 0, 2), memory)
        return self.fc(out.permute(1, 0, 2))

# ------------ Intent Classifier ------------
def train_intent_model(pairs):
    questions = [q.lower() for q, a in pairs]
    labels = []
    intent_map = {}
    fixed = {
        "greeting": ["hi", "hello", "how are you"],
        "thanks": ["thank you", "thanks"],
        "goodbye": ["bye", "see you"],
        "identity": ["who are you", "who created you"]
    }
    for q in questions:
        found = False
        for intent, examples in fixed.items():
            if any(ex in q for ex in examples):
                labels.append(intent)
                found = True
                break
        if not found:
            labels.append("default")

    vec = TfidfVectorizer()
    X = vec.fit_transform(questions)
    clf = LogisticRegression()
    clf.fit(X, labels)

    joblib.dump((vec, clf), "intent_classifier.joblib")
    return vec, clf

def predict_intent(text, vec, clf):
    return clf.predict(vec.transform([text]))[0]

# ------------ Generate Response ------------
def generate_response(model, tokenizer, prompt, max_len=20):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt, max_len)).unsqueeze(0)
    output = torch.tensor([[1]])
    for _ in range(max_len):
        with torch.no_grad():
            out = model(input_ids, output)
        probs = F.softmax(out[0, -1], dim=0)
        next_token = torch.multinomial(probs, 1).item()
        if next_token == 2:
            break
        output = torch.cat([output, torch.tensor([[next_token]])], dim=1)
    return tokenizer.decode(output[0][1:].tolist())

# ------------ Main ------------
if __name__ == "__main__":
    DATA_FILE = "kalai_dataset.txt"
    MODEL_FILE = "transformer_model.pth"
    tokenizer = SimpleTokenizer()
    dataset = ChatDataset(DATA_FILE, tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Intent Classifier
    if not os.path.exists("intent_classifier.joblib"):
        print("Training intent classifier...")
        train_intent_model(dataset.pairs)

    vec, clf = joblib.load("intent_classifier.joblib")

    # Load / Train Transformer
    model = TransformerBot(tokenizer.vocab_size())
    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))
        print("Transformer model loaded.")
    else:
        print("Training transformer model...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        model.train()
        for epoch in range(10):
            total_loss = 0
            for src, tgt in loader:
                optimizer.zero_grad()
                output = model(src, tgt[:, :-1])
                loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), MODEL_FILE)
        print("Model saved.")

    # Chat Loop
    intent_responses = {
        "greeting": ["Hello!", "Hi there!", "Hey!"],
        "thanks": ["You're welcome!", "Anytime, Kalai!"],
        "goodbye": ["Goodbye!", "See you later!"],
        "identity": ["I'm KalaiBot!", "Kalai created me!"]
    }

    print("\n💬 KalaiBot is ready! Type your message (or 'quit'):\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in ['quit', 'exit']:
            break
        intent = predict_intent(msg, vec, clf)
        if intent in intent_responses:
            print("Bot:", random.choice(intent_responses[intent]))
        else:
            print("Bot:", generate_response(model, tokenizer, msg))

