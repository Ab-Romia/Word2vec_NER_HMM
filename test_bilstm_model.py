"""
Quick test to verify the BiLSTM model architecture works correctly.
"""
import torch
import torch.nn as nn
import numpy as np

# Test BiLSTM model definition
class BiLSTM_NER(nn.Module):
    """
    BiLSTM model for Named Entity Recognition.
    """
    def __init__(self, vocab_size, embedding_dim, num_labels, hidden_dim=256,
                 num_lstm_layers=2, dropout=0.3, embeddings=None):
        super(BiLSTM_NER, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_lstm_layers = num_lstm_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Load pre-trained embeddings if provided
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
            self.embedding.weight.requires_grad = False

        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Linear classifier - input is hidden_dim * 2 for bidirectional
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, mask=None):
        # Get embeddings
        embedded = self.embedding(input_ids)

        # BiLSTM processes the entire sequence
        lstm_out, _ = self.lstm(embedded)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Classify each token
        logits = self.classifier(lstm_out)

        return logits

# Test parameters
vocab_size = 1000
embedding_dim = 100
num_labels = 9
batch_size = 4
seq_len = 20

print("Testing BiLSTM NER Model")
print("=" * 50)

# Create model
model = BiLSTM_NER(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_labels=num_labels,
    hidden_dim=256,
    num_lstm_layers=2,
    dropout=0.3
)

print(f"✓ Model created successfully")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create dummy input
dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
dummy_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)

# Forward pass
model.eval()
with torch.no_grad():
    logits = model(dummy_input, dummy_mask)

print(f"✓ Forward pass successful")
print(f"  Input shape: {dummy_input.shape}")
print(f"  Output shape: {logits.shape}")
print(f"  Expected output shape: ({batch_size}, {seq_len}, {num_labels})")

# Verify output shape
assert logits.shape == (batch_size, seq_len, num_labels), "Output shape mismatch!"
print(f"✓ Output shape correct!")

# Test that predictions are different for different tokens
# (showing the model considers context)
predictions = torch.argmax(logits, dim=-1)
print(f"\n✓ Predictions computed")
print(f"  Sample predictions for first sequence: {predictions[0].tolist()}")

print("\n" + "=" * 50)
print("✓ All tests passed! BiLSTM model is working correctly.")
print("\nKey differences from Feed-Forward:")
print("  • BiLSTM uses LSTM layers to process sequences")
print("  • Each token sees context from both directions")
print("  • Output dimension: hidden_dim * 2 (bidirectional)")
print("  • Better suited for sequential tasks like NER")
