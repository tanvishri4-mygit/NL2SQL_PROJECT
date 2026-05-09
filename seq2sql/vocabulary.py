"""
vocabulary.py
=============
Builds SQL output vocabulary from training data.
For BERT encoder: SQL vocab only (BERT handles input tokenisation)
For BiLSTM encoder: NL vocab + SQL vocab

Satisfies syllabus Module 2:
    tokenization, lemmatization, n-grams, frequency distribution

Run from project root:
    python seq2sql/vocabulary.py
"""

import re
import pickle
import pandas as pd
import nltk
from collections import Counter
from pathlib import Path
import sys

# ensure project root in path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from config import (
    PROCESSED_DIR, VOCAB_DIR,
    PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN,
    PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX
)

nltk.download("punkt",                        quiet=True)
nltk.download("punkt_tab",                    quiet=True)
nltk.download("wordnet",                      quiet=True)
nltk.download("stopwords",                    quiet=True)
nltk.download("averaged_perceptron_tagger",   quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.stem     import WordNetLemmatizer
from nltk          import pos_tag
from nltk.util     import ngrams

lemmatizer = WordNetLemmatizer()

# ═══════════════════════════════════════════════════════
# Vocabulary class
# ═══════════════════════════════════════════════════════

class Vocabulary:
    def __init__(self, name: str):
        self.name       = name
        self.token2idx  = {}
        self.idx2token  = {}
        self.token_freq = Counter()
        for tok in [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]:
            self._add(tok)

    def _add(self, token: str):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx]   = token

    def build(self, token_lists: list, min_freq: int = 1):
        for tokens in token_lists:
            self.token_freq.update(tokens)
        for token, freq in self.token_freq.items():
            if freq >= min_freq and token not in self.token2idx:
                self._add(token)
        print(f"  [{self.name}] size={len(self)} "
              f"(min_freq={min_freq})")

    def encode(self, tokens: list) -> list:
        return [self.token2idx.get(t, UNK_IDX) for t in tokens]

    def decode(self, indices: list) -> list:
        return [self.idx2token.get(i, UNK_TOKEN) for i in indices]

    def __len__(self):
        return len(self.token2idx)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Saved {self.name} vocab → {path}")

    @staticmethod
    def load(path: Path) -> "Vocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)

# ═══════════════════════════════════════════════════════
# Tokenisers
# ═══════════════════════════════════════════════════════

def tokenize_question(text: str, lemmatize: bool = True) -> list:
    """
    NL question tokeniser (satisfies Module 2).
    Steps: lowercase → word_tokenize → lemmatize
    """
    tokens = word_tokenize(text.lower().strip())
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

_SQL_RE = re.compile(
    r"\d+\.\d+|\d+|'(?:[^']|'')*'|\"(?:[^\"|\"\"]*)\"" +
    r"|!=|<>|<=|>=|[<>=()]|,|\*" +
    r"|[A-Za-z_][A-Za-z0-9_.]*|\S",
    re.VERBOSE
)

SQL_KW = {
    "SELECT","FROM","WHERE","GROUP","BY","ORDER","HAVING","LIMIT",
    "JOIN","INNER","LEFT","RIGHT","OUTER","ON","AS","COUNT","SUM",
    "AVG","MIN","MAX","DISTINCT","AND","OR","NOT","IN","LIKE",
    "BETWEEN","IS","NULL","ASC","DESC","UNION","INTERSECT","EXCEPT",
    "EXISTS","ALL","CASE","WHEN","THEN","ELSE","END","INSERT",
    "UPDATE","DELETE","CREATE","DROP","TABLE","VALUES","SET","INTO",
    "INDEX","TRUE","FALSE","CROSS","NATURAL","FULL","OUTER"
}

def tokenize_sql(sql: str) -> list:
    """SQL tokeniser — preserves structure."""
    tokens = _SQL_RE.findall(sql.strip())
    return [t.upper() if t.upper() in SQL_KW else t for t in tokens]

def tokenize_schema(schema_ctx: str) -> list:
    """Schema context tokeniser."""
    if not schema_ctx or schema_ctx.strip().lower() in ("nan", "none", ""):
        return []
    tokens = re.split(r'[\s|;:]+', schema_ctx.lower())
    return [t for t in tokens if t and t.lower() not in ("nan", "none")]

# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    print("\n" + "="*55)
    print("  Vocabulary Builder")
    print("="*55)

    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            "Run preprocess.py first."
        )

    print(f"\n[1] Loading {train_path} ...")
    df = pd.read_csv(train_path)
    print(f"  {len(df)} training rows")

    print("\n[2] Tokenising ...")
    nl_lists  = []
    sql_lists = []

    for _, row in df.iterrows():
        q  = str(row.get("question", ""))
        sc = str(row.get("schema_context", ""))
        sql = str(row.get("sql", ""))

        q_tok  = tokenize_question(q)
        sc_tok = tokenize_schema(sc)
        nl_lists.append(q_tok + sc_tok)
        sql_lists.append(tokenize_sql(sql))

    print(f"  Sample NL  tokens : {nl_lists[0][:8]}")
    print(f"  Sample SQL tokens : {sql_lists[0][:8]}")

    # ── NL vocab (for BiLSTM fallback) ────────────────
    print("\n[3] Building NL vocab (min_freq=2) ...")
    nl_vocab = Vocabulary("NL")
    nl_vocab.build(nl_lists, min_freq=2)

    # ── SQL vocab ─────────────────────────────────────
    print("\n[4] Building SQL vocab (min_freq=1) ...")
    sql_vocab = Vocabulary("SQL")
    sql_vocab.build(sql_lists, min_freq=1)

    # ── Save ──────────────────────────────────────────
    print("\n[5] Saving ...")
    nl_vocab.save(VOCAB_DIR  / "nl_vocab.pkl")
    sql_vocab.save(VOCAB_DIR / "sql_vocab.pkl")

    pd.DataFrame({
        "token": list(sql_vocab.token2idx.keys()),
        "idx"  : list(sql_vocab.token2idx.values())
    }).to_csv(VOCAB_DIR / "sql_vocab.csv", index=False)

    # ── Module 2 analysis ─────────────────────────────
    print("\n[6] NLP Analysis")

    # Frequency distribution
    all_nl_tokens = [t for lst in nl_lists for t in lst]
    freq = Counter(all_nl_tokens)
    print("\n  Top 15 NL tokens (frequency distribution):")
    for tok, cnt in freq.most_common(15):
        print(f"    {tok:20s} {cnt}")

    # POS tagging sample
    sample_q = str(df.iloc[0]["question"])
    sample_tokens = word_tokenize(sample_q)
    pos_tags = pos_tag(sample_tokens)
    print(f"\n  POS tagging sample: '{sample_q}'")
    print(f"    {pos_tags}")

    # Bigrams
    bigram_counter = Counter()
    for lst in nl_lists[:5000]:
        bigram_counter.update(ngrams(lst, 2))
    print("\n  Top 10 bigrams:")
    for bg, cnt in bigram_counter.most_common(10):
        print(f"    {' '.join(bg):30s} {cnt}")

    # Trigrams
    trigram_counter = Counter()
    for lst in nl_lists[:5000]:
        trigram_counter.update(ngrams(lst, 3))
    print("\n  Top 5 trigrams:")
    for tg, cnt in trigram_counter.most_common(5):
        print(f"    {' '.join(tg):40s} {cnt}")

    print(f"\n  NL  vocab size : {len(nl_vocab)}")
    print(f"  SQL vocab size : {len(sql_vocab)}")
    print("\n✅ Vocabulary building complete!")
    print("Next: python seq2sql/train.py")

if __name__ == "__main__":
    main()
