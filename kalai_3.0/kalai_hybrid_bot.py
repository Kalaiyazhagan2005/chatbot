# kalai_hybrid_bot.py
# Step 3: Unified Hybrid KalaiBot Pipeline

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
import json
import math
import joblib

# ---------- Paths ----------
INTENT_MODEL_PATH = "intent_classifier.joblib"
TRANSFORMER_MODEL_PATH = "transformer_model.pth"
DATA_PATH = "input.txt"

# ---------- Intent Classifier (ML) ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load intent classifier
intent_pipeline = joblib.load(INTENT_MODEL_PATH)
# Predefined responses
intent_responses = {
    "greeting": ["Hello! How can I help you?", "Hi there!"],
    "thanks": ["You're welcome!", "Happy to help."],
    "goodbye": ["Goodbye! Take care.", "See you later!"],
}

def predict_intent(text):
    return intent_pipeline.predict([text])[0]

# ---------- Transformer Setup ----------
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.idx = 4

    def build_vocab(self, sentences):
        for line in sentences:
            for w in line.lower().split():
                if w not in self.word2idx:
                    self.word2idx[w] = self.idx
                    self.idx2word[self.idx] = w
                    self.idx += 1

    def encode(self, text):
        return [self.word2idx.get(w, 3) for w in text.lower().split()]

    def decode(self, ids):
        return ' '.join(self.idx2word.get(i, '<UNK>') for i in ids)

class TransformerResponder(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embed(src).permute(1,0,2)
        tgt = self.embed(tgt).permute(1,0,2)
        memory = self.encoder(src)
        out = self.decoder(tgt, memory)
        return self.fc(out).permute(1,0,2)

# Initialize tokenizer and model
# For demo, build vocab on DATA_PATH lines
raw_lines = []
with open(DATA_PATH,'r') as f:
    for i,line in enumerate(f):
        if line.strip(): raw_lines.append(line.strip())
# Build tokenizer
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(raw_lines)
# Load transformer model
transformer = TransformerResponder(tokenizer.idx, d_model=64, nhead=2, num_layers=2)
transformer.load_state_dict(torch.load(TRANSFORMER_MODEL_PATH))

def generate_transformer_reply(text):
    transformer.eval()
    in_ids = tokenizer.encode(text)
    src = torch.tensor([in_ids])
    tgt = torch.tensor([[tokenizer.word2idx['<SOS>']]])
    for _ in range(20):
        with torch.no_grad():
            out = transformer(src, tgt)
        nid = out[0,-1].argmax().item()
        if nid==tokenizer.word2idx['<EOS>']: break
        tgt = torch.cat([tgt, torch.tensor([[nid]])], dim=1)
    return tokenizer.decode(tgt[0].tolist()[1:])

# ---------- Main Chat Loop ----------
if __name__=='__main__':
    print("KalaiBot v3.0 (Hybrid) Ready! Type 'quit' to exit.")
    while True:
        query = input("You: ")
        if query.lower() in ['quit','exit']: break
        intent = predict_intent(query)
        if intent in intent_responses:
            reply = random.choice(intent_responses[intent])
        else:
            reply = generate_transformer_reply(query)
        print("Bot:", reply)
