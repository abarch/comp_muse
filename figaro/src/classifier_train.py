import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from remi_classifier import RemiClassifier
from input_representation import InputRepresentation
from vocab import RemiVocab
import glob
import os
import numpy as np

ROOT_DIR = os.getenv('ROOT_DIR', './data')

# Training data
vocab = RemiVocab()


class MidiFiles(Dataset):
    def __init__(self, directory, is_train_data=True, train_val_split=0.9):
        self.midi_files_q1 = glob.glob(os.path.join(directory, 'Q1/*.mid'), recursive=True)
        self.midi_files_q3 = glob.glob(os.path.join(directory, 'Q3/*.mid'), recursive=True)
        if is_train_data:
            self.midi_files_q1 = self.midi_files_q1[:int(len(self.midi_files_q1) * train_val_split)]
            self.midi_files_q3 = self.midi_files_q3[:int(len(self.midi_files_q3) * train_val_split)]
        else:
            self.midi_files_q1 = self.midi_files_q1[int(len(self.midi_files_q1) * train_val_split):]
            self.midi_files_q3 = self.midi_files_q3[int(len(self.midi_files_q3) * train_val_split):]

    def __len__(self):
        return len(self.midi_files_q1) + len(self.midi_files_q3)

    def __getitem__(self, item):
        if item < len(self.midi_files_q1):
            file = self.midi_files_q1[item]
            label = torch.tensor(0)
        else:
            file = self.midi_files_q3[item - len(self.midi_files_q1)]
            label = torch.tensor(1)
        rep = InputRepresentation(file, strict=True)
        events = rep.get_remi_events()
        event_ids = torch.tensor(vocab.encode(events), dtype=torch.long)
        return event_ids, label


# Hyperparameters
num_tokens = 1357  # Number of different tokens in the Remi format
hidden_size = 64  # Size of the hidden state in the LSTM and Transformer
num_classes = 2  # Number of classes (binary classification)
learning_rate = 0.001
num_epochs = 10
batch_size = 1

data_generator_train = MidiFiles(ROOT_DIR, is_train_data=True, train_val_split=0.9)
dataset_train = DataLoader(data_generator_train, batch_size=batch_size, shuffle=True)

data_generator_val = MidiFiles(ROOT_DIR, is_train_data=False)
dataset_val = DataLoader(data_generator_val, batch_size=batch_size, shuffle=True)

# Model initialization
model = RemiClassifier(num_tokens, hidden_size, num_classes)
model.load_state_dict(torch.load('checkpoints/classifier.pth'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

# Loss and optimization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for x, y in dataset_train:
        torch.cuda.empty_cache()
        # Forward pass
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        output = model(x)

        # Calculate the loss
        loss = criterion(output, y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Output the current loss
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    model.eval()
    current_classified = []
    with torch.no_grad():
        for x, y in dataset_val:

            optimizer.zero_grad()
            x = x.to(device)
            output = model(x).detach().cpu()
            current_classified += (np.argmax(output, axis=1) == y).tolist()
        print(f"Accuracy: {np.mean(current_classified)}")

torch.save(model.state_dict(), 'checkpoints/classifier.pth')
