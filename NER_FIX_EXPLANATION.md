# NER Implementation Fix - Detailed Explanation

## Problem Summary

The original NER implementation had a **critical architectural flaw** that prevented it from properly learning named entity recognition patterns.

## Issues Identified

### 1. **No Sequential Context** ❌

**The Problem:**
```python
# Original Feed-Forward approach (WRONG)
embedded_flat = embedded.view(-1, emb_dim)    # Flattens all tokens
output = self.feedforward(embedded_flat)       # Processes independently
```

This approach treats each token **completely independently**:
- Token at position 5 has no information about tokens at positions 4 or 6
- The model can't learn sequence patterns like "B-PER → I-PER"
- Context is ignored, so "Washington" is always classified the same way regardless of surrounding words

### 2. **Misleading High Accuracy** 📊

Looking at your results:
```
Overall Accuracy: 90%  ✓ (seems good)
Macro F1-Score:   50%  ✗ (actually terrible!)
```

**Why this happens:**
- The 'O' (Outside) class represents 82.5% of all tokens (38,323 out of 46,435)
- A model that predicts 'O' most of the time gets ~90% accuracy
- But actual entity recognition is poor:
  - I-PER: **13% recall** (only finds 13% of person name continuations)
  - I-ORG: **30% recall** (misses 70% of organization continuations)
  - B-PER: **41% recall** (misses 59% of person names)

### 3. **The "11/20 Verification Pass" Pattern** 🎯

This happens because:
1. The model correctly identifies some easy cases (common locations, obvious patterns)
2. But fails on context-dependent cases:
   - "CHINA" tagged as B-LOC instead of B-PER (when it's a person name)
   - Can't maintain consistency (predicts B-PER then I-ORG, which is invalid)
   - Struggles with multi-word entities

## The Solution: BiLSTM Architecture ✓

### Why BiLSTM?

**BiLSTM (Bidirectional Long Short-Term Memory)** is the standard architecture for sequence labeling tasks like NER because:

1. **Sequential Processing:**
   - Processes tokens in sequence, maintaining hidden states
   - Each token's representation includes information from previous tokens

2. **Bidirectional Context:**
   - Forward LSTM: reads left-to-right (past → present)
   - Backward LSTM: reads right-to-left (future → present)
   - Each token gets context from BOTH directions

3. **Learned Patterns:**
   - Can learn valid tag sequences (B-PER → I-PER is valid, B-PER → I-ORG is not)
   - Understands entity boundaries
   - Recognizes multi-word entities

### Architecture Comparison

**Feed-Forward (WRONG):**
```
Input Token → Embedding → FeedForward Layers → Classification
                               ↓
                    No context from other tokens!
```

**BiLSTM (CORRECT):**
```
Input Sequence → Embeddings → BiLSTM Layers → Classification
                                   ↓
                     Forward LSTM (left context)
                     Backward LSTM (right context)
                     Combined for each token
```

### Code Changes

**Original (Feed-Forward):**
```python
class FeedForwardNER(nn.Module):
    def __init__(self, ...):
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            ...
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, mask=None):
        embedded = self.embedding(input_ids)
        embedded_flat = embedded.view(-1, emb_dim)  # ❌ Loses sequence structure
        output = self.feedforward(embedded_flat)     # ❌ Independent processing
        logits = self.classifier(output)
        return logits.view(batch_size, seq_len, num_labels)
```

**Fixed (BiLSTM):**
```python
class BiLSTM_NER(nn.Module):
    def __init__(self, ...):
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,              # ✓ Bidirectional context
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)  # *2 for bidirectional

    def forward(self, input_ids, mask=None):
        embedded = self.embedding(input_ids)  # (batch, seq_len, emb_dim)
        lstm_out, _ = self.lstm(embedded)     # ✓ Sequential processing
        # lstm_out shape: (batch, seq_len, hidden_dim * 2)
        # Each token now has context from entire sequence!
        logits = self.classifier(lstm_out)
        return logits
```

## Expected Improvements

With the BiLSTM architecture, you should see:

1. **Better Entity Recognition:**
   - I-PER recall: 13% → 70-80%
   - I-ORG recall: 30% → 65-75%
   - B-PER recall: 41% → 75-85%

2. **Higher Macro F1-Score:**
   - Previous: ~50%
   - Expected: 75-85%

3. **More Consistent Predictions:**
   - Valid tag sequences (B-PER → I-PER, not B-PER → I-ORG)
   - Better multi-word entity recognition
   - Context-aware predictions

4. **Better Verification Results:**
   - Should pass 18-20 out of 20 tests instead of 11/20

## How to Use the Fixed Implementation

The file `ner_neural_network.ipynb` has been updated with:
1. BiLSTM model class replacing FeedForward
2. Detailed comments explaining the architecture
3. Notes about the fix at the top of the notebook

Simply re-run the notebook cells to train with the new architecture. Training time will be similar, but results will be much better.

## Why This Matters for Your Assignment

The assignment requires a **Neural Network-based NER model**. While a feed-forward network is technically a neural network, it's:
- Not appropriate for sequence labeling tasks
- Will give poor results on actual entity recognition
- Doesn't demonstrate understanding of NER requirements

The BiLSTM model:
- ✓ Is the standard approach for NER
- ✓ Properly models sequential dependencies
- ✓ Will achieve much better performance
- ✓ Shows proper understanding of the task

## References

Standard NER architectures typically use:
1. **BiLSTM** (what we're using now) - Good baseline
2. **BiLSTM + CRF** - Adds constraint layer for valid tag sequences
3. **Transformer-based** (BERT, etc.) - State-of-the-art but more complex

For this assignment, BiLSTM alone should give excellent results (75-85% F1).

---

**Questions?** Check the updated notebook for implementation details and comments explaining each component.
