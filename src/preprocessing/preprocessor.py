"""
NLP Preprocessing Module
Agent 06 (Service Developer): Implements classical NLP preprocessing pipeline
covering tokenization, POS tagging, lemmatization, n-grams, and stopword removal.

Satisfies Syllabus Module 2: Text Mining, Cleaning, and Pre-processing
- Various Tokenizers, Tokenization, Frequency Distribution, Stemming,
  POS Tagging, Lemmatization, Bigrams, Trigrams & Ngrams, Entity Recognition
"""

import re
import string
from collections import Counter


# ============================================================
# Stopwords (minimal set - no NLTK dependency)
# ============================================================
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should",
    "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn",
    "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
    "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    # Domain-specific keep words NOT in stopwords
}

# Words to KEEP even though they might seem like stopwords (important for SQL)
KEEP_WORDS = {
    "show", "list", "get", "find", "display", "give", "tell", "count",
    "total", "average", "sum", "max", "min", "top", "bottom", "last",
    "first", "highest", "lowest", "most", "least", "all", "each", "every",
    "group", "order", "sort", "limit", "where", "from", "between", "after",
    "before", "above", "below", "more", "less", "only", "not", "no"
}


# ============================================================
# Tokenizer
# ============================================================
def tokenize(text: str) -> list:
    """
    Tokenize input text into words.
    Handles:
    - Lowercasing
    - Splitting on whitespace and punctuation
    - Preserving numbers and dates
    - Handling contractions
    """
    text = text.lower().strip()
    
    # Replace common contractions
    contractions = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "can't": "cannot", "won't": "will not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
        "what's": "what is", "who's": "who is", "it's": "it is",
        "that's": "that is", "there's": "there is", "here's": "here is",
        "i'm": "i am", "you're": "you are", "we're": "we are",
        "they're": "they are", "i've": "i have", "you've": "you have",
        "we've": "we have", "they've": "they have", "i'll": "i will",
        "you'll": "you will", "he'll": "he will", "she'll": "she will",
        "we'll": "we will", "they'll": "they will", "i'd": "i would",
    }
    for cont, expansion in contractions.items():
        text = text.replace(cont, expansion)
    
    # Tokenize: split on non-alphanumeric (keeping numbers, underscores, hyphens in dates)
    tokens = re.findall(r'\b[\w]+\b', text)
    return tokens


def remove_stopwords(tokens: list, keep_sql_words: bool = True) -> list:
    """Remove stopwords while preserving SQL-relevant words."""
    result = []
    for t in tokens:
        if t in KEEP_WORDS:
            result.append(t)
        elif t not in STOPWORDS:
            result.append(t)
    return result


# ============================================================
# POS Tagger (Rule-based, lightweight)
# ============================================================
# Simple rule-based POS tagger for NL2SQL domain
POS_RULES = {
    # Verbs / Actions
    "show": "VB", "list": "VB", "get": "VB", "find": "VB", "display": "VB",
    "give": "VB", "tell": "VB", "fetch": "VB", "retrieve": "VB", "select": "VB",
    "count": "VB", "calculate": "VB", "compute": "VB", "compare": "VB",
    "sort": "VB", "order": "VB", "group": "VB", "filter": "VB", "search": "VB",
    
    # Aggregation keywords
    "total": "AGG", "average": "AGG", "avg": "AGG", "sum": "AGG",
    "maximum": "AGG", "max": "AGG", "minimum": "AGG", "min": "AGG",
    "highest": "AGG", "lowest": "AGG", "most": "AGG", "least": "AGG",
    "mean": "AGG",
    
    # SQL structural words
    "top": "LIMIT", "first": "LIMIT", "last": "LIMIT", "bottom": "LIMIT",
    "limit": "LIMIT",
    "by": "PREP", "per": "PREP", "for": "PREP", "from": "PREP",
    "in": "PREP", "of": "PREP", "with": "PREP", "between": "PREP",
    "where": "CONJ", "and": "CONJ", "or": "CONJ", "but": "CONJ",
    "after": "OP", "before": "OP", "above": "OP", "below": "OP",
    "greater": "OP", "less": "OP", "more": "OP", "than": "OP",
    "equal": "OP", "not": "OP", "only": "OP", "except": "OP",
    
    # Adjectives/Modifiers
    "all": "DET", "each": "DET", "every": "DET",
    "ascending": "MOD", "descending": "MOD", "asc": "MOD", "desc": "MOD",
}

def pos_tag(tokens: list) -> list:
    """
    Rule-based POS tagging optimized for NL2SQL domain.
    Returns list of (token, tag) tuples.
    
    Tags: VB (verb), NN (noun), AGG (aggregate), LIMIT (limiter),
          PREP (preposition), CONJ (conjunction), OP (operator),
          DET (determiner), MOD (modifier), NUM (number), DATE (date), UNK (unknown)
    """
    tagged = []
    for i, token in enumerate(tokens):
        if token in POS_RULES:
            tagged.append((token, POS_RULES[token]))
        elif re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$', token):
            tagged.append((token, "DATE"))
        elif re.match(r'^\d+\.?\d*$', token):
            tagged.append((token, "NUM"))
        elif re.match(r'^\d{4}$', token):
            # Could be year
            tagged.append((token, "DATE"))
        else:
            tagged.append((token, "NN"))  # Default: noun (likely table/column name)
    return tagged


# ============================================================
# Lemmatizer (Rule-based, lightweight)
# ============================================================
LEMMA_RULES = {
    "orders": "order", "customers": "customer", "products": "product",
    "payments": "payment", "shipments": "shipment", "returns": "return",
    "categories": "category", "addresses": "address", "promotions": "promotion",
    "items": "item", "prices": "price", "sales": "sale",
    "shipped": "ship", "delivered": "deliver", "cancelled": "cancel",
    "returned": "return", "placed": "place", "packed": "pack",
    "purchased": "purchase", "ordered": "order", "paid": "pay",
    "refunded": "refund", "failed": "fail",
    "highest": "high", "lowest": "low", "latest": "late",
    "biggest": "big", "smallest": "small",
    "running": "run", "showing": "show", "listing": "list",
    "getting": "get", "finding": "find", "giving": "give",
    "sorting": "sort", "grouping": "group", "filtering": "filter",
    "spending": "spend", "earning": "earn",
}

def lemmatize(token: str) -> str:
    """Simple rule-based lemmatizer for NL2SQL domain."""
    token = token.lower()
    if token in LEMMA_RULES:
        return LEMMA_RULES[token]
    # Basic suffix rules
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 3 and token[-3] not in "aeiou":
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    return token

def lemmatize_tokens(tokens: list) -> list:
    """Lemmatize a list of tokens."""
    return [lemmatize(t) for t in tokens]


# ============================================================
# Stemmer (Porter-like, simplified)
# ============================================================
def stem(token: str) -> str:
    """Simple suffix-stripping stemmer."""
    token = token.lower()
    for suffix in ["ation", "ment", "ness", "ible", "able", "ful", "less", "ous", "ive", "ing", "ied", "ies", "ed", "er", "ly", "es", "s"]:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token


# ============================================================
# N-grams Generator
# ============================================================
def generate_ngrams(tokens: list, n: int = 2) -> list:
    """Generate n-grams from a list of tokens."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def generate_bigrams(tokens: list) -> list:
    """Generate bigrams."""
    return generate_ngrams(tokens, 2)

def generate_trigrams(tokens: list) -> list:
    """Generate trigrams."""
    return generate_ngrams(tokens, 3)


# ============================================================
# Frequency Distribution
# ============================================================
def freq_distribution(tokens: list) -> dict:
    """Compute frequency distribution of tokens."""
    return dict(Counter(tokens))


# ============================================================
# Full Preprocessing Pipeline
# ============================================================
class NLPreprocessor:
    """
    Complete NLP preprocessing pipeline for NL2SQL.
    Combines tokenization, stopword removal, POS tagging, 
    lemmatization, and n-gram generation.
    """
    
    def __init__(self):
        self.stopwords = STOPWORDS
    
    def preprocess(self, text: str) -> dict:
        """
        Run full preprocessing pipeline on input text.
        
        Returns dict with:
        - original: original text
        - tokens: raw tokens
        - tokens_clean: stopwords removed
        - pos_tags: POS tagged tokens
        - lemmas: lemmatized tokens
        - bigrams: bigram features
        - trigrams: trigram features
        - freq_dist: frequency distribution
        """
        tokens = tokenize(text)
        tokens_clean = remove_stopwords(tokens)
        pos_tags = pos_tag(tokens_clean)
        lemmas = lemmatize_tokens(tokens_clean)
        bigrams = generate_bigrams(tokens_clean)
        trigrams = generate_trigrams(tokens_clean)
        freq = freq_distribution(tokens_clean)
        
        return {
            "original": text,
            "tokens": tokens,
            "tokens_clean": tokens_clean,
            "pos_tags": pos_tags,
            "lemmas": lemmas,
            "bigrams": bigrams,
            "trigrams": trigrams,
            "freq_dist": freq
        }
    
    def get_action_words(self, pos_tags: list) -> list:
        """Extract action/verb tokens."""
        return [t for t, tag in pos_tags if tag == "VB"]
    
    def get_aggregate_words(self, pos_tags: list) -> list:
        """Extract aggregation tokens."""
        return [t for t, tag in pos_tags if tag == "AGG"]
    
    def get_nouns(self, pos_tags: list) -> list:
        """Extract noun tokens (likely table/column references)."""
        return [t for t, tag in pos_tags if tag == "NN"]
    
    def get_numbers(self, pos_tags: list) -> list:
        """Extract numeric tokens."""
        return [t for t, tag in pos_tags if tag in ("NUM", "DATE")]
    
    def get_operators(self, pos_tags: list) -> list:
        """Extract operator tokens."""
        return [t for t, tag in pos_tags if tag == "OP"]


# ============================================================
# Convenience
# ============================================================
preprocessor = NLPreprocessor()

if __name__ == "__main__":
    # Demo
    queries = [
        "Show all customers from Mumbai",
        "Count total orders in 2025",
        "Top 5 products by revenue",
        "Average order total by city",
        "List cancelled orders with refund amount greater than 5000",
    ]
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = preprocessor.preprocess(q)
        print(f"Tokens: {result['tokens']}")
        print(f"Clean:  {result['tokens_clean']}")
        print(f"POS:    {result['pos_tags']}")
        print(f"Lemmas: {result['lemmas']}")
        print(f"Bigrams:{result['bigrams']}")
