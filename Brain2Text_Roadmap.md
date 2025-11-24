# Brain-to-Text '25 Kaggle Competition Roadmap

A comprehensive learning and implementation roadmap for building a speech decoding model from intracortical neural signals.

---

## Phase 1: Foundation & Environment Setup (Week 1)

### 1.1 Prerequisites
- **Python proficiency**: Understanding of NumPy, Pandas, PyTorch basics
- **Deep learning background**: Familiarity with RNNs, LSTMs, transformers
- **Signal processing fundamentals**: Understanding of time-series data and filtering

### 1.2 Environment Configuration
```bash
# Create a virtual environment
python -m venv brain2text
source brain2text/bin/activate  # On Windows: brain2text\Scripts\activate

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas h5py matplotlib scikit-learn tensorboard wandb
pip install librosa scipy tqdm jupyter notebook
```

### 1.3 Download the Dataset
- Visit [Kaggle Brain-to-Text '25 competition page](https://www.kaggle.com/competitions/brain-to-text-25)
- Download dataset from either:
  - Kaggle Data tab (recommended for competition participants)
  - Dryad repository: https://doi.org/10.5061/dryad.dncjsxm85
- Extract to local directory (e.g., `~/data/brain2text/`)

### 1.4 Clone Baseline Repository
```bash
git clone https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text.git
cd nejm-brain-to-text
pip install -r requirements.txt  # Install competition dependencies
```

---

## Phase 2: Data Exploration & Understanding (Week 1-2)

### 2.1 Dataset Structure Analysis
**Learning Goal**: Understand HDF5 file format, data organization, and dimensions

```python
import h5py
import numpy as np

# Explore HDF5 structure
session_file = 't15.2023.08.13_data_train.hdf5'
with h5py.File(session_file, 'r') as f:
    print("Keys in file:", list(f.keys()))
    # Expected keys: neural_data, words, phonemes, word_times, phoneme_times
    
    # Neural data shape: (trials, timesteps, channels) or (timesteps, channels)
    neural = f['neural_data'][:]  # Shape likely (T, 256) for 256 electrodes
    words = f['words'][:]  # Text transcripts
    
    print(f"Neural data shape: {neural.shape}")
    print(f"Number of trials: {len(words)}")
    print(f"Sample transcript: {words[0]}")
```

### 2.2 Data Statistics
- **Number of sessions**: 45 sessions across 20 months
- **Total sentences**: 10,948 (train+val)
- **Test sentences**: 1,450
- **Electrodes**: 256 intracortical microelectrodes in speech motor cortex
- **Neural sampling rate**: ~30 kHz raw, typically binned to 20ms windows
- **Speaking speed**: ~30 wpm (attempted vocalized), ~50 wpm (attempted silent)

### 2.3 Exploratory Data Analysis Tasks
1. **Plot neural activity**: Visualize spike rates across electrodes and time
2. **Analyze sentence lengths**: Distribution of words per sentence
3. **Check data balance**: Are all sentence types equally represented?
4. **Identify missing blocks**: Note which corpora/strategies are in train vs. test

### 2.4 Create Data Visualization Script
```python
import matplotlib.pyplot as plt

# Visualize neural activity for a sample trial
trial_neural = neural_data[0]  # Shape: (timesteps, 256)
plt.figure(figsize=(12, 6))
plt.imshow(trial_neural.T, aspect='auto', cmap='viridis')
plt.xlabel('Time (20ms bins)')
plt.ylabel('Electrode index')
plt.title('Neural Activity Across 256 Electrodes')
plt.colorbar(label='Spike count')
plt.show()
```

---

## Phase 3: Data Preprocessing Pipeline (Week 2)

### 3.1 Neural Signal Processing
**Learning Goal**: Convert raw spike data to usable feature representations

```python
def preprocess_neural_data(neural_data, bin_size_ms=20, sampling_rate_hz=30000):
    """
    Convert raw neural signals to binned spike counts.
    
    Args:
        neural_data: Raw spike times/counts (T, 256)
        bin_size_ms: Time bin size in milliseconds
        sampling_rate_hz: Original sampling rate
    
    Returns:
        binned_data: Spike counts per bin (T', 256) where T' < T
    """
    bin_size_samples = int(bin_size_ms * sampling_rate_hz / 1000)
    n_bins = neural_data.shape[0] // bin_size_samples
    
    binned = neural_data[:n_bins * bin_size_samples].reshape(
        n_bins, bin_size_samples, -1
    ).sum(axis=1)
    
    return binned
```

### 3.2 Normalization & Standardization
```python
from sklearn.preprocessing import StandardScaler

def normalize_neural_data(train_data, val_data=None, test_data=None):
    """
    Standardize neural features using training set statistics.
    """
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
    
    if val_data is not None:
        val_norm = scaler.transform(val_data.reshape(-1, val_data.shape[-1])).reshape(val_data.shape)
    if test_data is not None:
        test_norm = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
    
    return train_norm, val_norm, test_norm, scaler
```

### 3.3 Data Augmentation (Optional but Recommended)
```python
def temporal_masking(neural_data, mask_probability=0.1, mask_size=5):
    """
    Randomly mask temporal windows in neural data.
    Improves robustness by simulating missing electrode recordings.
    """
    augmented = neural_data.copy()
    n_timesteps = neural_data.shape[0]
    
    for _ in range(int(n_timesteps * mask_probability / mask_size)):
        start_idx = np.random.randint(0, n_timesteps - mask_size)
        augmented[start_idx:start_idx + mask_size] = 0
    
    return augmented
```

### 3.4 Create PyTorch Dataset Class
```python
import torch
from torch.utils.data import Dataset, DataLoader

class BrainToTextDataset(Dataset):
    def __init__(self, neural_data, transcripts, phoneme_mappings):
        """
        Args:
            neural_data: List of (T, 256) neural arrays
            transcripts: List of text strings
            phoneme_mappings: Dict mapping phonemes to indices
        """
        self.neural_data = neural_data
        self.transcripts = transcripts
        self.phoneme_map = phoneme_mappings
        
    def __len__(self):
        return len(self.transcripts)
    
    def __getitem__(self, idx):
        neural = torch.FloatTensor(self.neural_data[idx])  # (T, 256)
        transcript = self.transcripts[idx]
        phonemes = self.text_to_phonemes(transcript)
        phoneme_ids = torch.LongTensor([self.phoneme_map[p] for p in phonemes])
        
        return {
            'neural': neural,
            'phonemes': phoneme_ids,
            'transcript': transcript
        }
    
    def text_to_phonemes(self, text):
        # Use g2p_en or similar library to convert text to phonemes
        # For now, return simplified version
        return list(text.lower().replace(' ', '|'))  # Placeholder

def collate_batch(batch):
    """Handle variable-length sequences."""
    neural_list = [item['neural'] for item in batch]
    phoneme_list = [item['phonemes'] for item in batch]
    
    # Pad to max length in batch
    max_neural_len = max(x.shape[0] for x in neural_list)
    max_phoneme_len = max(len(x) for x in phoneme_list)
    
    neural_padded = torch.zeros(len(batch), max_neural_len, 256)
    neural_lengths = []
    
    for i, neural in enumerate(neural_list):
        neural_padded[i, :neural.shape[0]] = neural
        neural_lengths.append(neural.shape[0])
    
    phoneme_padded = torch.zeros(len(batch), max_phoneme_len, dtype=torch.long)
    phoneme_lengths = []
    
    for i, phoneme in enumerate(phoneme_list):
        phoneme_padded[i, :len(phoneme)] = phoneme
        phoneme_lengths.append(len(phoneme))
    
    return {
        'neural': neural_padded,
        'neural_lengths': torch.tensor(neural_lengths),
        'phonemes': phoneme_padded,
        'phoneme_lengths': torch.tensor(phoneme_lengths)
    }

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_batch,
    num_workers=4,
    pin_memory=True
)
```

---

## Phase 4: Phoneme System Setup (Week 2)

### 4.1 Understanding Phonemes
**39 English Phonemes Used in Baseline**:
- Vowels: /aa/, /ae/, /ah/, /ao/, /aw/, /ay/, /eh/, /er/, /ih/, /iy/, /ow/, /oy/, /uh/, /uw/
- Consonants: /b/, /ch/, /d/, /dh/, /f/, /g/, /hh/, /jh/, /k/, /l/, /m/, /n/, /ng/, /p/, /r/, /s/, /sh/, /t/, /th/, /v/, /w/, /y/, /z/, /zh/

### 4.2 Text-to-Phoneme Conversion
```python
# Install g2p_en for grapheme-to-phoneme conversion
pip install g2p-en

from g2p_en.g2p import G2p

g2p = G2p()

def text_to_phonemes(text):
    """Convert text to phoneme sequence."""
    phonemes = g2p(text.lower())
    return phonemes  # Returns list like ['t', 'h', 'e', 'q', 's', 't']

# Create phoneme inventory
phoneme_set = set()
for transcript in all_transcripts:
    phonemes = text_to_phonemes(transcript)
    phoneme_set.update(phonemes)

phoneme_to_id = {p: i for i, p in enumerate(sorted(phoneme_set))}
id_to_phoneme = {i: p for p, i in phoneme_to_id.items()}

print(f"Total phonemes: {len(phoneme_to_id)}")
print(f"Phoneme mapping: {phoneme_to_id}")
```

---

## Phase 5: Baseline Model Implementation (Week 3-4)

### 5.1 Understanding CTC Loss
**Connectionist Temporal Classification**: Solves sequence-to-sequence problems without requiring alignment between input and output sequences.

**Key Concept**: 
\[
P(\text{phonemes} | \text{neural}) = \sum_{\text{all alignments}} P(\text{alignment} | \text{neural})
\]

**Why CTC for this task**:
- You don't know which neural timestep corresponds to which phoneme
- CTC automatically learns the alignment during training
- Handles variable-length inputs and outputs

### 5.2 RNN Encoder Architecture
```python
import torch
import torch.nn as nn

class NeuralEncoder(nn.Module):
    """RNN-based neural activity encoder."""
    
    def __init__(self, input_size=256, hidden_size=512, num_layers=3, 
                 output_size=39, dropout=0.3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Bidirectional GRU for better context
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Project to phoneme vocabulary size
        # Factor of 2 because GRU is bidirectional
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, neural_input, input_lengths):
        """
        Args:
            neural_input: (batch_size, max_timesteps, 256)
            input_lengths: (batch_size,) actual lengths before padding
        
        Returns:
            logits: (batch_size, max_timesteps, num_phonemes)
        """
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            neural_input, input_lengths.cpu(), 
            batch_first=True, enforce_sorted=False
        )
        
        # Pass through GRU
        gru_output, _ = self.gru(packed)
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True)
        # output shape: (batch_size, max_timesteps, hidden_size * 2)
        
        # Project to phoneme logits
        logits = self.fc(output)
        # logits shape: (batch_size, max_timesteps, num_phonemes)
        
        return logits
```

### 5.3 Training with CTC Loss
```python
import torch.nn.functional as F

class BrainToTextTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            neural = batch['neural'].to(self.device)
            phonemes = batch['phonemes'].to(self.device)
            neural_lengths = batch['neural_lengths'].to(self.device)
            phoneme_lengths = batch['phoneme_lengths'].to(self.device)
            
            # Forward pass
            logits = self.model(neural, neural_lengths)
            # logits: (batch_size, max_timesteps, num_phonemes)
            
            # Prepare for CTC loss
            # CTC expects: (max_timesteps, batch_size, num_phonemes)
            logits = logits.permute(1, 0, 2)
            log_probs = F.log_softmax(logits, dim=2)
            
            # Compute CTC loss
            loss = self.ctc_loss(log_probs, phonemes, neural_lengths, phoneme_lengths)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                neural = batch['neural'].to(self.device)
                phonemes = batch['phonemes'].to(self.device)
                neural_lengths = batch['neural_lengths'].to(self.device)
                phoneme_lengths = batch['phoneme_lengths'].to(self.device)
                
                logits = self.model(neural, neural_lengths)
                logits = logits.permute(1, 0, 2)
                log_probs = F.log_softmax(logits, dim=2)
                
                loss = self.ctc_loss(log_probs, phonemes, neural_lengths, phoneme_lengths)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=50):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
```

### 5.4 Inference with Beam Search
```python
def decode_phonemes_beam_search(logits, beam_width=10, blank_idx=0):
    """
    Decode CTC logits to phoneme sequence using beam search.
    
    Args:
        logits: (timesteps, num_phonemes)
        beam_width: Number of hypotheses to track
    
    Returns:
        Best phoneme sequence
    """
    from collections import defaultdict
    
    timesteps = logits.shape[0]
    num_classes = logits.shape[1]
    
    # Simplified beam search (full implementation is more complex)
    log_probs = F.log_softmax(logits, dim=1)  # (T, num_classes)
    
    # Greedy decoding for simplicity (replace with beam search for better results)
    predicted_indices = torch.argmax(log_probs, dim=1).cpu().numpy()
    
    # Remove blanks and consecutive duplicates
    phoneme_sequence = []
    for idx in predicted_indices:
        if idx != blank_idx and (not phoneme_sequence or phoneme_sequence[-1] != idx):
            phoneme_sequence.append(idx)
    
    return phoneme_sequence
```

---

## Phase 6: Language Model Integration (Week 4)

### 6.1 N-gram Language Model
```python
from collections import Counter, defaultdict
import math

class NGramLM:
    def __init__(self, n=5):
        self.n = n
        self.counts = defaultdict(Counter)
        self.total_grams = defaultdict(int)
    
    def train(self, phoneme_sequences):
        """Train n-gram model from phoneme sequences."""
        for seq in phoneme_sequences:
            for i in range(len(seq) - self.n + 1):
                context = tuple(seq[i:i+self.n-1])
                next_phoneme = seq[i+self.n-1]
                self.counts[context][next_phoneme] += 1
                self.total_grams[context] += 1
    
    def get_probability(self, context, phoneme):
        """Get P(phoneme | context)."""
        if context not in self.counts or len(self.counts[context]) == 0:
            return 1e-5  # Smoothing
        
        count = self.counts[context].get(phoneme, 0)
        total = self.total_grams[context]
        return (count + 1) / (total + len(self.counts))  # Add-one smoothing
```

### 6.2 LLM Rescoring (Advanced)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMRescorer:
    def __init__(self, model_name='facebook/opt-6.7b'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    
    def rescore_sentences(self, candidate_sentences, top_k=5):
        """Rescore candidates using LLM perplexity."""
        best_sentence = None
        best_score = float('-inf')
        
        for sentence in candidate_sentences[:top_k]:
            inputs = self.tokenizer(sentence, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
            
            # Lower perplexity (loss) is better
            score = -outputs.loss.item()
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        return best_sentence
```

---

## Phase 7: Advanced Techniques (Week 5+)

### 7.1 Ensemble Decoding (Top Strategy from 2024 Competition)
```python
class EnsembleDecoder:
    def __init__(self, num_models=10):
        self.models = [NeuralEncoder(...) for _ in range(num_models)]
    
    def decode(self, neural_input, neural_lengths):
        """Generate predictions from multiple models."""
        all_phoneme_probs = []
        
        for model in self.models:
            logits = model(neural_input, neural_lengths)
            probs = F.softmax(logits, dim=2)
            all_phoneme_probs.append(probs)
        
        # Average probabilities
        avg_probs = torch.stack(all_phoneme_probs).mean(dim=0)
        return avg_probs
```

### 7.2 Diphone-Based Decoding
```python
class Diphones:
    """Model transitions between phonemes rather than isolated phonemes."""
    
    def __init__(self, phoneme_list):
        self.phonemes = phoneme_list
        self.diphones = []
        for p1 in phoneme_list:
            for p2 in phoneme_list:
                self.diphones.append(f"{p1}_{p2}")
        self.diphone_to_id = {d: i for i, d in enumerate(self.diphones)}
    
    def text_to_diphones(self, phoneme_seq):
        """Convert phoneme sequence to diphone sequence."""
        diphone_seq = []
        for i in range(len(phoneme_seq) - 1):
            diphone = f"{phoneme_seq[i]}_{phoneme_seq[i+1]}"
            if diphone in self.diphone_to_id:
                diphone_seq.append(self.diphone_to_id[diphone])
        return diphone_seq
```

### 7.3 Transformer-Based Architecture
```python
class TransformerEncoder(nn.Module):
    """Alternative to RNN using Transformer blocks."""
    
    def __init__(self, input_size=256, d_model=512, nhead=8, 
                 num_layers=6, num_phonemes=39):
        super().__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_phonemes)
    
    def forward(self, neural_input, input_lengths):
        # Embed
        embedded = self.embedding(neural_input)  # (batch, T, d_model)
        
        # Create attention mask for padding
        mask = self.create_padding_mask(input_lengths, neural_input.device)
        
        # Transformer
        output = self.transformer(embedded, src_key_padding_mask=mask)
        
        # Project to phonemes
        logits = self.fc(output)
        return logits
    
    def create_padding_mask(self, lengths, device):
        max_len = lengths.max()
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        return mask
```

---

## Phase 8: Evaluation Metrics (Week 4)

### 8.1 Word Error Rate (WER)
```python
from difflib import SequenceMatcher

def calculate_wer(reference, hypothesis):
    """Calculate word error rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    matcher = SequenceMatcher(None, ref_words, hyp_words)
    
    # Calculate edit distance
    d = {}
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j], d[i, j-1], d[i-1, j-1]) + 1
    
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words) if ref_words else 0
    return wer

# Calculate aggregate WER
def calculate_aggregate_wer(references, hypotheses):
    total_edit_distance = 0
    total_words = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        
        # Edit distance calculation
        d = {}
        for i in range(len(ref_words) + 1):
            d[i, 0] = i
        for j in range(len(hyp_words) + 1):
            d[0, j] = j
        
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i, j] = d[i-1, j-1]
                else:
                    d[i, j] = min(d[i-1, j], d[i, j-1], d[i-1, j-1]) + 1
        
        total_edit_distance += d[len(ref_words), len(hyp_words)]
        total_words += len(ref_words)
    
    aggregate_wer = total_edit_distance / total_words if total_words > 0 else 0
    return aggregate_wer
```

---

## Phase 9: Competition Submission (Week 5+)

### 9.1 Create Submission CSV
```python
import pandas as pd

def create_submission(test_predictions, output_file='submission.csv'):
    """
    Create properly formatted submission file.
    
    Args:
        test_predictions: List of decoded text sentences in chronological order
        output_file: Output CSV filename
    """
    df = pd.DataFrame({
        'id': range(len(test_predictions)),
        'text': test_predictions
    })
    
    # Remove punctuation
    df['text'] = df['text'].apply(lambda x: x.replace('.', '').replace(',', '').replace('?', ''))
    
    df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    return df
```

### 9.2 Inference Pipeline
```python
def generate_predictions(model, test_loader, id_to_phoneme, lm, device='cuda'):
    """Generate predictions for test set."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            neural = batch['neural'].to(device)
            neural_lengths = batch['neural_lengths'].to(device)
            
            # Get phoneme predictions
            logits = model(neural, neural_lengths)
            
            for i in range(logits.shape[0]):
                # Decode single example
                logits_single = logits[i, :neural_lengths[i]]
                phoneme_ids = decode_phonemes_beam_search(logits_single)
                
                # Convert to phonemes
                phonemes = [id_to_phoneme[pid] for pid in phoneme_ids]
                phoneme_str = ''.join(phonemes)
                
                # Use language model to convert to text
                # This is simplified; in reality use phoneme_to_word mapping
                text = phoneme_str.replace('|', ' ')
                predictions.append(text)
    
    return predictions
```

---

## Phase 10: Optimization & Experimentation (Ongoing)

### 10.1 Hyperparameter Grid
```python
hyperparams = {
    'bin_size_ms': [10, 20, 30, 50],
    'hidden_size': [256, 512, 1024],
    'num_layers': [2, 3, 4, 5],
    'dropout': [0.1, 0.2, 0.3, 0.5],
    'learning_rate': [1e-4, 1e-3, 5e-4],
    'batch_size': [16, 32, 64],
    'ngram_size': [3, 4, 5, 6],
}
```

### 10.2 Experiment Tracking
```python
import wandb

wandb.init(project="brain-to-text-25")

# Log metrics
wandb.log({
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_wer': val_wer,
    'learning_rate': current_lr
})

# Log hyperparameters
wandb.config.update({
    'batch_size': 32,
    'hidden_size': 512,
    'num_layers': 3,
})
```

### 10.3 Strategies for Improvement
1. **Data augmentation**: Temporal masking, electrode dropout, time warping
2. **Model ensemble**: Combine 5-10 independently trained models
3. **Architecture search**: Experiment with Transformers, Conv1D, Attention layers
4. **Loss functions**: Try CTC variants (focal CTC), Transducer loss
5. **Transfer learning**: Fine-tune on ASR datasets or Brain-to-Text '24 T12 data
6. **Test-time adaptation**: Continuous fine-tuning on test distribution
7. **Corpus-aware decoding**: Detect speaking strategy and apply appropriate language model

---

## Timeline Summary

| Week | Phase | Key Milestones |
|------|-------|----------------|
| 1 | Setup & Exploration | Environment ready, dataset downloaded, EDA complete |
| 2 | Preprocessing | Data pipeline functional, PyTorch dataset working |
| 3 | Baseline Model | RNN+CTC model training, achieving 10-15% WER |
| 4 | Integration | Language model integrated, inference pipeline complete |
| 5+ | Advanced Techniques | Ensemble models, diphones, Transformers, optimization |

---

## Key Repositories & Resources

- **Official Baseline**: https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text
- **Research Paper**: https://www.nejm.org/doi/full/10.1056/NEJMoa2314132
- **2024 Lessons**: https://arxiv.org/abs/2412.17227
- **Kaggle Competition**: https://www.kaggle.com/competitions/brain-to-text-25
- **Related Research**: https://github.com/NeuSpeech/awesome-brain-decoding

---

## Expected Performance Milestones

- **Baseline (starter model)**: ~10-12% WER
- **With optimization**: ~7-9% WER
- **Advanced ensemble**: ~5-7% WER
- **Competition top 3**: 4-5% WER (target)

Good luck with the competition! Focus on solid engineering fundamentals first, then iterate on advanced techniques based on validation performance.
