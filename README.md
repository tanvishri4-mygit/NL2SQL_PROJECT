# Schema-Aware Natural Language Interface for Relational Databases
## Using Classical NLP, BERT Seq2Seq, TF-IDF Schema Linking, and Beam Search Decoding

> **M.Tech NLP Course Project - ECL545** | Tanvi Shrivastava (MT24AAC002) & Abhishek Gupte (MT24AAC021)

---

## NOTE  TO PROFESSOR
Download inferencing data and model from this drive link - https://drive.google.com/file/d/1CHXiY9UQE3cECcni5v5fdIrSnMTFKTcr/view?usp=sharing
This will download a zip file which will have a README_SETUP.txt file
Follow the steps written in README_SETUP.txt file to place the data and models in correct repository and then run the project based on this README.md file

## Overview

This system converts natural language questions into executable SQL queries against the **Chinook Digital Music Store** database. It combines classical NLP techniques with a trained BERT + LSTM Seq2Seq neural model.

**Database:** Chinook SQLite - 11 tables (Artist, Album, Track, Genre, Customer, Invoice, InvoiceLine, Employee, MediaType, Playlist, PlaylistTrack)

**Training Data:** WikiSQL (56,355 pairs) + Spider (8,659 pairs) + 481 Chinook-specific hand-written pairs

---

## Architecture

```
User Question
    ↓
Layer 1 - Classical NLP Preprocessing
          tokenize · POS tag · lemmatize · bigrams · trigrams
    ↓
Layer 2 - Naive Bayes Intent Classifier
          TF-IDF features · 8 intent classes
    ↓
Layer 3 - TF-IDF Schema Linker (RAG)
          table linking · column linking · value linking · join path detection
    ↓
Layer 4 - BERT Encoder + Bahdanau Attention + LSTM Decoder
          bert-base-uncased · trained on 64K pairs · beam search k=5
    ↓
Layer 5 - SQL Post-Processing Corrector
          alias resolution · column correction · value correction
    ↓
Execute on chinook_1.sqlite → Results
    ↓
User Feedback → saved for next fine-tuning cycle
```

---

## Syllabus Coverage (ECL545)

| Module | Requirement | Implementation |
|---|---|---|
| Module 2 | Tokenization | `engine_v2.py` - `NLPreprocessor` |
| Module 2 | POS Tagging | `engine_v2.py` - NLTK perceptron tagger |
| Module 2 | Lemmatization | `engine_v2.py` - WordNetLemmatizer |
| Module 2 | Bigrams / Trigrams / N-grams | `engine_v2.py` - NLTK ngrams |
| Module 2 | Entity Recognition | `schema_linker.py` - value linking |
| Module 3 | TF-IDF | `schema_linker.py` + `engine_v2.py` |
| Module 3 | Naive Bayes Classifier | `engine_v2.py` - `IntentClassifier` |
| Module 3 | Confusion Matrix | `engine_v2.py` - intent evaluation |
| Module 4 | Language Modeling | BERT + LSTM Seq2Seq (`seq2sql/`) |
| Module 4 | Sequence Tagging | Bahdanau attention alignment |
| Module 4 | Context-Free Grammar | SQL vocabulary + beam search decoding |
| Module 5 | AI Chatbot | `app_v2.py` - Streamlit interface |
| Module 5 | Recommendation Engine | `engine_v2.py` - query suggestions |

---

## Installation

### Step 1 - Create conda environment

```bash
conda create -n nl2sql python=3.10
conda activate nl2sql
```

### Step 2 - Install dependencies

```bash
pip install -r requirements_seq2sql.txt
```

### Step 3 - Download NLTK data

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```

### Step 4 - Place model files

The trained model file `best_bert.pt` and fine-tuned model `finetuned_bert.pt` are not stored in git due to size (~1.7GB each). Place them at:

```
models/best_bert.pt
models/finetuned_bert.pt
```

Also place vocabulary files at:

```
data/vocab/sql_vocab.pkl
data/vocab/nl_vocab.pkl
```

### Step 5 - Run the application

```bash
streamlit run app_v2.py
```

Open browser at `http://localhost:8501`

---

## Project Structure

```
nl2sql_project/
├── app_v2.py                      # Main Streamlit UI
├── app.py                         # Classical NLP demo UI (old system)
├── engine_v2.py                   # Main orchestrator - all layers
├── config.py                      # Paths, schemas, hyperparameters
├── schema_linker.py               # TF-IDF schema linker (RAG)
├── sql_corrector.py               # Post-processing SQL corrector
├── feedback_logger.py             # User feedback collection
├── requirements.txt               # Base dependencies
├── requirements_seq2sql.txt       # Neural model dependencies
│
├── seq2sql/
│   ├── model.py                   # Seq2Seq model + beam search
│   ├── decoder.py                 # LSTM decoder + Bahdanau attention
│   ├── encoder_bert.py            # BERT encoder wrapper
│   ├── inference.py               # Inference pipeline
│   ├── train.py                   # Original training script
│   ├── finetune.py                # Fine-tuning script
│   ├── dataset.py                 # DataLoader
│   ├── vocabulary.py              # SQL/NL vocabulary
│   └── preprocess.py             # Data preprocessing
│
├── rules/
│   └── sql_rules.py               # CFG rule patterns
│
├── src/                           # Classical NLP system (old)
│   ├── preprocessing/
│   ├── classification/
│   ├── retrieval/
│   ├── slot_tagging/
│   ├── sql_generation/
│   ├── agent/
│   └── conversation/
│
├── data/
│   ├── chinook_pairs.csv          # 481 hand-written Chinook NL-SQL pairs
│   ├── intent_dataset.csv         # Naive Bayes training data
│   ├── schema.json                # Ecommerce schema (old system)
│   ├── feedback_pairs.csv         # User-corrected SQL pairs
│   ├── feedback_positive.csv      # User-verified correct pairs
│   └── spider/database/chinook_1/ # Inference database
│
├── models/
│   ├── best_bert.pt               # Original trained model (not in git)
│   ├── finetuned_bert.pt          # Fine-tuned model (not in git)
│   └── log_bert.csv               # Training log
│
└── checks/
    ├── eval_queries.py            # 32-query evaluation script
    ├── eval_results_baseline.txt  # Baseline results (5/32)
    └── eval_results_finetuned.txt # Fine-tuned results (17/32)
```

---

## Model Training

### Original Training

Trained on WikiSQL + Spider (64K pairs) for 24 epochs on RTX 4070 GPU.

```bash
# Build vocabulary first
python seq2sql/vocabulary.py

# Preprocess training data
python seq2sql/preprocess.py

# Train
python seq2sql/train.py
```

### Fine-Tuning on Chinook Data

Fine-tuned on 481 hand-written Chinook-specific pairs.

```bash
python seq2sql/finetune.py
```

---

## Model Performance

| Model | Val Loss | Token Accuracy | Perplexity | Eval (32 queries) |
|---|---|---|---|---|
| `best_bert.pt` (baseline) | 2.762 | 66.7% | 11.61 | 5/32 |
| `finetuned_bert.pt` | 0.0038 | 100% (Chinook val) | 1.0 | 17/32 |

---

## Evaluation

Run the evaluation script to test model performance across 9 SQL categories:

```bash
conda activate nl2sql
python checks/eval_queries.py
```

Results saved to `checks/eval_results.txt`

---

## Example Queries

```
show all artists
→ SELECT * FROM Artist

show customers from usa
→ SELECT * FROM Customer WHERE Country = 'USA'

how many artists are there
→ SELECT COUNT(*) FROM Artist

total revenue by country
→ SELECT BillingCountry, SUM(Total) FROM Invoice GROUP BY BillingCountry

show albums with artist names
→ SELECT Album.Title, Artist.Name FROM Album JOIN Artist ON Album.ArtistId = Artist.ArtistId

top 5 artists by number of albums
→ SELECT Artist.Name, COUNT(Album.AlbumId) FROM Artist JOIN Album ON Artist.ArtistId = Album.ArtistId GROUP BY Artist.ArtistId ORDER BY COUNT(Album.AlbumId) DESC LIMIT 5
```

---

## Self-Learning Feedback Loop

The system collects user feedback after every query:

- **Correct** - saves the NL-SQL pair to `data/feedback_positive.csv`
- **Incorrect + correct SQL** - verifies SQL against the database, saves to `data/feedback_pairs.csv`

These pairs are used in the next fine-tuning cycle, allowing the model to improve from real usage.

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| Neural Model | PyTorch 2.0 |
| BERT | HuggingFace Transformers (bert-base-uncased) |
| Classical NLP | NLTK |
| UI | Streamlit |
| Database | SQLite3 |
| Schema Linking | scikit-learn TF-IDF |
| Training Hardware | NVIDIA RTX 4070 Laptop GPU (8GB VRAM) |

---

## Authors

**Tanvi Shrivastava** - MT24AAC002
**Abhishek Gupte** - MT24AAC021

M.Tech in Applied AI and Communications
Natural Language Processing - ECL545
