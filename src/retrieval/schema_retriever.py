"""
Schema Retrieval Module (RAG Layer)
Agent 06 (Service Developer): Implements retrieval-augmented schema understanding.
- TF-IDF baseline retrieval
- Embedding-style retrieval (using TF-IDF as proxy without internet)
- Schema linking: maps NL tokens to database tables/columns

This is the RAG component - retrieves relevant schema context for SQL generation.
"""

import os
import sys
import json
import math
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.preprocessor import tokenize, remove_stopwords, lemmatize_tokens

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")


class SchemaRetriever:
    """
    Retrieves relevant tables and columns from schema metadata
    given a natural language query.
    
    Two retrieval strategies:
    1. TF-IDF cosine similarity (classical NLP baseline)
    2. Synonym-based exact matching (rule-based)
    
    Combined score = alpha * tfidf_score + (1-alpha) * synonym_score
    """
    
    def __init__(self, schema_path=None, alpha=0.6):
        if schema_path is None:
            schema_path = os.path.join(DATA_DIR, "schema.json")
        
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        
        self.alpha = alpha
        self.tables = self.schema["tables"]
        self.joins = self.schema.get("joins", [])
        
        # Build indices
        self._build_synonym_index()
        self._build_tfidf_index()
    
    def _build_synonym_index(self):
        """Build reverse synonym lookup: synonym → (table, column?)."""
        self.synonym_to_table = {}  # synonym → table_name
        self.synonym_to_column = {}  # synonym → (table_name, column_name)
        
        for table in self.tables:
            tname = table["name"]
            # Table-level synonyms
            for syn in table.get("synonyms", []):
                self.synonym_to_table[syn.lower()] = tname
            self.synonym_to_table[tname.lower()] = tname
            
            # Column-level synonyms
            for col in table.get("columns", []):
                cname = col["name"]
                for syn in col.get("synonyms", []):
                    self.synonym_to_column[syn.lower()] = (tname, cname)
                self.synonym_to_column[cname.lower()] = (tname, cname)
    
    def _build_tfidf_index(self):
        """Build TF-IDF index over schema descriptions."""
        # Create documents: one per table (combining description + column descriptions)
        self.schema_docs = {}
        self.schema_tokens = {}
        
        for table in self.tables:
            tname = table["name"]
            text_parts = [tname, table.get("description", "")]
            text_parts.extend(table.get("synonyms", []))
            
            for col in table.get("columns", []):
                text_parts.append(col["name"])
                text_parts.append(col.get("description", ""))
                text_parts.extend(col.get("synonyms", []))
            
            full_text = " ".join(text_parts)
            tokens = tokenize(full_text)
            tokens = lemmatize_tokens(tokens)
            self.schema_docs[tname] = full_text
            self.schema_tokens[tname] = tokens
        
        # Compute IDF
        N = len(self.schema_tokens)
        self.df = Counter()
        for tokens in self.schema_tokens.values():
            for t in set(tokens):
                self.df[t] += 1
        
        self.idf = {t: math.log(N / (1 + df)) for t, df in self.df.items()}
        
        # Pre-compute TF-IDF vectors for each table
        self.table_vectors = {}
        all_terms = sorted(self.idf.keys())
        self.term_to_idx = {t: i for i, t in enumerate(all_terms)}
        self.n_terms = len(all_terms)
        
        for tname, tokens in self.schema_tokens.items():
            vec = self._tfidf_vector(tokens)
            self.table_vectors[tname] = vec
    
    def _tfidf_vector(self, tokens: list) -> list:
        """Compute TF-IDF vector for a token list."""
        vec = [0.0] * self.n_terms
        token_counts = Counter(tokens)
        doc_len = len(tokens) if len(tokens) > 0 else 1
        
        for token, count in token_counts.items():
            if token in self.term_to_idx:
                idx = self.term_to_idx[token]
                tf = count / doc_len
                idf = self.idf.get(token, 0)
                vec[idx] = tf * idf
        
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
    
    def _cosine_similarity(self, v1: list, v2: list) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        return dot  # Already L2 normalized
    
    def retrieve_tables(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve top-k relevant tables for a query.
        
        Returns list of dicts:
        [{"table": name, "score": float, "matched_columns": [...], "reason": str}]
        """
        tokens = tokenize(query)
        tokens_clean = remove_stopwords(tokens)
        lemmas = lemmatize_tokens(tokens_clean)
        
        # Strategy 1: TF-IDF similarity
        query_vec = self._tfidf_vector(lemmas)
        tfidf_scores = {}
        for tname, tvec in self.table_vectors.items():
            tfidf_scores[tname] = self._cosine_similarity(query_vec, tvec)
        
        # Strategy 2: Synonym matching
        synonym_scores = defaultdict(float)
        matched_columns = defaultdict(list)
        
        for token in lemmas:
            # Check table synonyms
            if token in self.synonym_to_table:
                tname = self.synonym_to_table[token]
                synonym_scores[tname] += 1.0
            
            # Check column synonyms
            if token in self.synonym_to_column:
                tname, cname = self.synonym_to_column[token]
                synonym_scores[tname] += 0.5
                matched_columns[tname].append(cname)
            
            # Partial matching (token contained in synonym)
            for syn, tname in self.synonym_to_table.items():
                if token in syn or syn in token:
                    synonym_scores[tname] += 0.3
        
        # Normalize synonym scores
        max_syn = max(synonym_scores.values()) if synonym_scores else 1.0
        if max_syn > 0:
            synonym_scores = {k: v / max_syn for k, v in synonym_scores.items()}
        
        # Combine scores
        all_tables = set(list(tfidf_scores.keys()) + list(synonym_scores.keys()))
        combined = {}
        for tname in all_tables:
            ts = tfidf_scores.get(tname, 0)
            ss = synonym_scores.get(tname, 0)
            combined[tname] = self.alpha * ts + (1 - self.alpha) * ss
        
        # Sort and return top-k
        sorted_tables = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for tname, score in sorted_tables[:top_k]:
            # Find matching columns
            cols = list(set(matched_columns.get(tname, [])))
            
            # Build reason string
            reasons = []
            if tfidf_scores.get(tname, 0) > 0.1:
                reasons.append("TF-IDF content match")
            if synonym_scores.get(tname, 0) > 0:
                reasons.append("synonym match")
            if cols:
                reasons.append(f"columns: {', '.join(cols)}")
            
            results.append({
                "table": tname,
                "score": round(score, 4),
                "tfidf_score": round(tfidf_scores.get(tname, 0), 4),
                "synonym_score": round(synonym_scores.get(tname, 0), 4),
                "matched_columns": cols,
                "reason": "; ".join(reasons) if reasons else "low relevance",
            })
        
        return results
    
    def retrieve_columns(self, query: str, table_name: str) -> list:
        """
        Retrieve relevant columns for a specific table given a query.
        Returns sorted list of (column_name, score) tuples.
        """
        tokens = tokenize(query)
        tokens_clean = remove_stopwords(tokens)
        lemmas = lemmatize_tokens(tokens_clean)
        
        table_info = None
        for t in self.tables:
            if t["name"] == table_name:
                table_info = t
                break
        
        if not table_info:
            return []
        
        col_scores = []
        for col in table_info["columns"]:
            score = 0.0
            cname = col["name"]
            synonyms = [s.lower() for s in col.get("synonyms", [])] + [cname.lower()]
            
            for token in lemmas:
                for syn in synonyms:
                    if token == syn:
                        score += 1.0
                    elif token in syn or syn in token:
                        score += 0.5
            
            col_scores.append((cname, round(score, 2)))
        
        # Sort by score descending
        col_scores.sort(key=lambda x: x[1], reverse=True)
        return col_scores
    
    def find_join_path(self, tables: list) -> list:
        """
        Find join conditions between a set of tables.
        Returns list of join condition strings.
        """
        if len(tables) <= 1:
            return []
        
        join_conditions = []
        visited = {tables[0]}
        remaining = set(tables[1:])
        
        # BFS to find join paths
        max_iterations = 10
        iteration = 0
        while remaining and iteration < max_iterations:
            iteration += 1
            found = False
            for join in self.joins:
                from_t = join["from"]
                to_t = join["to"]
                
                if from_t in visited and to_t in remaining:
                    join_conditions.append(join["on"])
                    visited.add(to_t)
                    remaining.discard(to_t)
                    found = True
                elif to_t in visited and from_t in remaining:
                    join_conditions.append(join["on"])
                    visited.add(from_t)
                    remaining.discard(from_t)
                    found = True
            
            if not found:
                # Try indirect joins through intermediate tables
                for join in self.joins:
                    from_t = join["from"]
                    to_t = join["to"]
                    if from_t in visited and to_t not in visited:
                        join_conditions.append(join["on"])
                        visited.add(to_t)
                        if to_t in remaining:
                            remaining.discard(to_t)
                        found = True
                        break
                    elif to_t in visited and from_t not in visited:
                        join_conditions.append(join["on"])
                        visited.add(from_t)
                        if from_t in remaining:
                            remaining.discard(from_t)
                        found = True
                        break
                
                if not found:
                    break
        
        return join_conditions
    
    def get_table_schema(self, table_name: str) -> dict:
        """Get full schema info for a table."""
        for t in self.tables:
            if t["name"] == table_name:
                return t
        return None
    
    def get_retrieval_context(self, query: str, top_k: int = 3) -> dict:
        """
        Full RAG retrieval: get tables, columns, and joins for a query.
        This is the main entry point for the SQL generator.
        """
        tables = self.retrieve_tables(query, top_k=top_k)
        
        context = {
            "query": query,
            "retrieved_tables": tables,
            "table_schemas": {},
            "join_conditions": [],
            "top_columns": {},
        }
        
        # Get full schema for retrieved tables
        table_names = [t["table"] for t in tables if t["score"] > 0.05]
        for tname in table_names:
            schema = self.get_table_schema(tname)
            if schema:
                context["table_schemas"][tname] = schema
                # Get relevant columns
                cols = self.retrieve_columns(query, tname)
                context["top_columns"][tname] = cols
        
        # Find joins
        if len(table_names) > 1:
            context["join_conditions"] = self.find_join_path(table_names)
        
        return context


# ============================================================
# Evaluation
# ============================================================
def evaluate_retrieval(retriever, test_queries=None):
    """Evaluate retrieval precision@k."""
    if test_queries is None:
        from classification.intent_dataset import INTENT_DATASET
        test_queries = [(q, tables.split(",")) for q, _, tables, _ in INTENT_DATASET]
    
    precision_at_1 = []
    precision_at_3 = []
    
    for query, gold_tables in test_queries:
        results = retriever.retrieve_tables(query, top_k=3)
        retrieved = [r["table"] for r in results]
        
        # Precision@1
        if retrieved and retrieved[0] in gold_tables:
            precision_at_1.append(1.0)
        else:
            precision_at_1.append(0.0)
        
        # Precision@3
        hits = sum(1 for t in retrieved[:3] if t in gold_tables)
        p3 = hits / min(3, len(gold_tables)) if gold_tables else 0
        precision_at_3.append(p3)
    
    return {
        "precision_at_1": round(sum(precision_at_1) / len(precision_at_1), 4),
        "precision_at_3": round(sum(precision_at_3) / len(precision_at_3), 4),
        "n_queries": len(test_queries),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("SCHEMA RETRIEVER - RAG Layer")
    print("=" * 80)
    
    retriever = SchemaRetriever()
    
    # Demo queries
    test_queries = [
        "show all customers from Mumbai",
        "total revenue by category",
        "top 5 products by sales",
        "count cancelled orders with refund",
        "average order value by city",
        "show payment method for each order",
    ]
    
    for q in test_queries:
        print(f"\nQuery: \"{q}\"")
        results = retriever.retrieve_tables(q, top_k=3)
        for r in results:
            print(f"  → {r['table']} (score: {r['score']:.3f}) [{r['reason']}]")
    
    # Evaluate
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)
    eval_results = evaluate_retrieval(retriever)
    print(f"Precision@1: {eval_results['precision_at_1']:.4f}")
    print(f"Precision@3: {eval_results['precision_at_3']:.4f}")
    print(f"Evaluated on: {eval_results['n_queries']} queries")
