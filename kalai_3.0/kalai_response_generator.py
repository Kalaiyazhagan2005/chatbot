# kalai_response_generator.py
# KalaiBot Step 2: Transformer-based response generation fallback

import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import random
from intent_model import predict_intent

# ----------- Tokenizer -----------
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.idx = 4

    def build_vocab(self, sentences):
        for line in sentences:
            for word in line.strip().lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.idx
                    self.idx2word[self.idx] = word
                    self.idx += 1

    def encode(self, sentence):
        return [self.word2idx.get(word, 3) for word in sentence.strip().lower().split()]

    def decode(self, indices):
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])

    def vocab_size(self):
        return len(self.word2idx)

# ----------- Tiny Transformer -----------
class TransformerResponder(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src).permute(1, 0, 2)
        tgt = self.embedding(tgt).permute(1, 0, 2)
        memory = self.encoder(src)
        out = self.decoder(tgt, memory)
        return self.fc(out.permute(1, 0, 2))

# ----------- Pretrained Responses -----------
intent_responses = {
    "greeting": ["Hello! How can I help you today?", "Hi there!"],
    "thanks": ["You're welcome!", "Glad I could help."],
    "goodbye": ["Goodbye! Have a great day!", "See you later!"]
}

# ----------- Chat Function -----------
def respond(user_input, model, tokenizer):
    intent = predict_intent(user_input)

    if intent in intent_responses:
        return random.choice(intent_responses[intent])

    # Else use transformer (for more complex queries)
    model.eval()
    input_ids = tokenizer.encode(user_input)
    input_tensor = torch.tensor([input_ids])
    tgt_tensor = torch.tensor([[tokenizer.word2idx["<SOS>"]]])

    for _ in range(20):
        with torch.no_grad():
            output = model(input_tensor, tgt_tensor)
        next_token = output[0, -1].argmax().item()
        if next_token == tokenizer.word2idx["<EOS>"]:
            break
        tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[next_token]])], dim=1)

    return tokenizer.decode(tgt_tensor[0][1:].tolist())

# ----------- Main -----------
if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer = SimpleTokenizer()
    data_samples = ["how are you", "what is ai", "explain neural networks"]
    tokenizer.build_vocab(data_samples)

    model = TransformerResponder(tokenizer.vocab_size())
    model.load_state_dict(torch.load("transformer_model.pth"))

    print("KalaiBot is ready! (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        response = respond(user_input, model, tokenizer)
        print("Bot:", response)
