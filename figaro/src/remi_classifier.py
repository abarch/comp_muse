import torch
import torch.nn as nn


class RemiClassifier(nn.Module):
    def __init__(self, num_tokens, hidden_size, num_classes):
        super(RemiClassifier, self).__init__()

        self.token_embedding = nn.Embedding(num_tokens, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=4)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_tokens):
        embedded = self.token_embedding(input_tokens)
        embedded = embedded.permute(1, 0, 2)  # Shape: (sequence_length, batch_size, hidden_size)

        lstm_output, _ = self.lstm(embedded)

        final_hidden = lstm_output[-1]  # Verwende den letzten LSTM Hidden State
        output = self.fc_layers(final_hidden)

        return output


# Beispielverwendung:

if __name__ == '__main__':
    # Hyperparameter
    num_tokens = 1357  # Anzahl der verschiedenen Token im Remi-Format
    hidden_size = 128  # Größe des versteckten Zustands im LSTM und Transformer
    num_classes = 2  # Anzahl der Klassen (binäre Klassifikation)

    # Modellinitialisierung
    model = RemiClassifier(num_tokens, hidden_size, num_classes)

    # Beispiel-Eingabedaten
    input_tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])

    # Forward-Pass
    output = model(input_tokens)
    print(output)
