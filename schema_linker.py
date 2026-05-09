"""
schema_linker.py
================
TF-IDF based schema linking for the Chinook database.

Covers all three levels:
    Level 1 — Table linking  : "billing records" → Invoice
    Level 2 — Column linking : "song length"     → Track.Milliseconds
    Level 3 — Value linking  : "rock music"      → Genre.Name = 'Rock'

Uses:
    - TF-IDF cosine similarity (satisfies syllabus Module 3)
    - Synonym expansion for schema elements
    - Pattern matching for value extraction

Used by:
    - rules/sql_rules.py  (replaces hardcoded TABLE_KEYWORDS)
    - engine_v2.py        (schema context for neural model)
    - app_v2.py           (Schema Retrieval RAG panel display)

No external APIs. No pretrained models. Pure classical NLP.
"""

import re
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
import sys

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ═══════════════════════════════════════════════════════════════════════════
# Chinook Schema Definition
# ═══════════════════════════════════════════════════════════════════════════

CHINOOK_SCHEMA = {
    "Album": {
        "description": "music albums releases collections records",
        "synonyms"   : ["album", "albums", "release", "releases",
                        "record", "records", "collection"],
        "columns": {
            "AlbumId" : {"desc": "album identifier id", "syns": ["album id", "albumid"]},
            "Title"   : {"desc": "album title name", "syns": ["title", "album name", "album title"]},
            "ArtistId": {"desc": "artist identifier foreign key", "syns": ["artist id"]},
        }
    },
    "Artist": {
        "description": "musicians singers bands performers music creators",
        "synonyms"   : ["artist", "artists", "musician", "musicians",
                        "singer", "singers", "band", "bands", "performer"],
        "columns": {
            "ArtistId": {"desc": "artist identifier id", "syns": ["artist id"]},
            "Name"    : {"desc": "artist name band name", "syns": ["name", "artist name"]},
        }
    },
    "Customer": {
        "description": "customers buyers clients users consumers people who purchase shop",
        "synonyms"   : ["customer", "customers", "buyer", "buyers",
                        "client", "clients", "user", "users", "person",
                        "people", "subscriber", "consumer", "consumers",
                        "shopper", "shoppers"],
        "columns": {
            "CustomerId"  : {"desc": "customer id identifier", "syns": ["customer id"]},
            "FirstName"   : {"desc": "first name given name", "syns": ["first name", "firstname"]},
            "LastName"    : {"desc": "last name surname family name", "syns": ["last name", "lastname", "surname"]},
            "Company"     : {"desc": "company organization employer", "syns": ["company", "organization", "employer"]},
            "Address"     : {"desc": "street address location", "syns": ["address", "street"]},
            "City"        : {"desc": "city town location", "syns": ["city", "town", "location"]},
            "State"       : {"desc": "state province region", "syns": ["state", "province", "region"]},
            "Country"     : {"desc": "country nation", "syns": ["country", "nation", "nationality"]},
            "PostalCode"  : {"desc": "postal code zip code", "syns": ["postal code", "zip", "zipcode"]},
            "Phone"       : {"desc": "phone number telephone contact", "syns": ["phone", "telephone", "contact"]},
            "Email"       : {"desc": "email address", "syns": ["email", "mail"]},
            "SupportRepId": {"desc": "support representative employee", "syns": ["support rep", "representative"]},
        }
    },
    "Employee": {
        "description": "employees staff workers support representatives managers",
        "synonyms"   : ["employee", "employees", "staff", "worker",
                        "workers", "representative", "support rep"],
        "columns": {
            "EmployeeId": {"desc": "employee id identifier", "syns": ["employee id"]},
            "LastName"  : {"desc": "last name surname", "syns": ["last name", "surname"]},
            "FirstName" : {"desc": "first name given name", "syns": ["first name"]},
            "Title"     : {"desc": "job title position role", "syns": ["title", "position", "role", "job"]},
            "ReportsTo" : {"desc": "manager supervisor reports to", "syns": ["manager", "supervisor", "reports to"]},
            "BirthDate" : {"desc": "birth date birthday born", "syns": ["birth date", "birthday", "born"]},
            "HireDate"  : {"desc": "hire date started joined employment date", "syns": ["hire date", "start date", "joined"]},
            "City"      : {"desc": "city location", "syns": ["city"]},
            "Country"   : {"desc": "country", "syns": ["country"]},
            "Email"     : {"desc": "email", "syns": ["email"]},
        }
    },
    "Genre": {
        "description": "music genres categories types styles rock pop jazz classical",
        "synonyms"   : ["genre", "genres", "style", "styles",
                        "category", "categories", "type", "music type"],
        "columns": {
            "GenreId": {"desc": "genre id identifier", "syns": ["genre id"]},
            "Name"   : {"desc": "genre name style type rock pop jazz", "syns": ["name", "genre name", "style"]},
        }
    },
    "Invoice": {
        "description": "invoices bills purchases orders transactions revenue sales payments",
        "synonyms"   : ["invoice", "invoices", "bill", "bills",
                        "purchase", "purchases", "order", "orders",
                        "transaction", "transactions", "sale", "sales",
                        "revenue", "billing", "receipt"],
        "columns": {
            "InvoiceId"         : {"desc": "invoice id identifier", "syns": ["invoice id"]},
            "CustomerId"        : {"desc": "customer id foreign key", "syns": ["customer id"]},
            "InvoiceDate"       : {"desc": "invoice date purchase date transaction date", "syns": ["date", "invoice date", "purchase date", "transaction date"]},
            "BillingAddress"    : {"desc": "billing address street", "syns": ["billing address", "address"]},
            "BillingCity"       : {"desc": "billing city location", "syns": ["billing city", "city"]},
            "BillingState"      : {"desc": "billing state province", "syns": ["billing state", "state"]},
            "BillingCountry"    : {"desc": "billing country nation", "syns": ["billing country", "country", "nation"]},
            "BillingPostalCode" : {"desc": "billing postal code zip", "syns": ["postal code", "zip"]},
            "Total"             : {"desc": "total amount price cost revenue spending", "syns": ["total", "amount", "price", "cost", "revenue", "value", "spending"]},
        }
    },
    "InvoiceLine": {
        "description": "invoice line items tracks purchased quantities prices",
        "synonyms"   : ["invoice line", "invoice lines", "invoiceline",
                        "line item", "line items", "purchase item",
                        "order item", "order line"],
        "columns": {
            "InvoiceLineId": {"desc": "invoice line id", "syns": ["line id"]},
            "InvoiceId"    : {"desc": "invoice id foreign key", "syns": ["invoice id"]},
            "TrackId"      : {"desc": "track id foreign key", "syns": ["track id"]},
            "UnitPrice"    : {"desc": "unit price cost per item", "syns": ["unit price", "price", "cost"]},
            "Quantity"     : {"desc": "quantity number of items purchased", "syns": ["quantity", "count", "number", "amount purchased"]},
        }
    },
    "MediaType": {
        "description": "media types audio formats file types mp3 aac mpeg",
        "synonyms"   : ["media type", "media types", "mediatype",
                        "format", "formats", "audio format",
                        "file type", "file format", "type"],
        "columns": {
            "MediaTypeId": {"desc": "media type id", "syns": ["media type id"]},
            "Name"        : {"desc": "media type name format mp3 aac mpeg", "syns": ["name", "format name", "type name"]},
        }
    },
    "Playlist": {
        "description": "playlists collections of tracks music lists",
        "synonyms"   : ["playlist", "playlists", "list", "music list",
                        "collection", "track list"],
        "columns": {
            "PlaylistId": {"desc": "playlist id", "syns": ["playlist id"]},
            "Name"       : {"desc": "playlist name title", "syns": ["name", "playlist name"]},
        }
    },
    "PlaylistTrack": {
        "description": "playlist track associations which tracks are in which playlists",
        "synonyms"   : ["playlist track", "playlist tracks", "playlisttrack"],
        "columns": {
            "PlaylistId": {"desc": "playlist id foreign key", "syns": ["playlist id"]},
            "TrackId"   : {"desc": "track id foreign key", "syns": ["track id"]},
        }
    },
    "Track": {
        "description": "tracks songs music audio files with duration price composer",
        "synonyms"   : ["track", "tracks", "song", "songs",
                        "music", "audio", "recording"],
        "columns": {
            "TrackId"    : {"desc": "track id identifier", "syns": ["track id"]},
            "Name"       : {"desc": "track name song title", "syns": ["name", "title", "song name", "track name"]},
            "AlbumId"    : {"desc": "album id foreign key", "syns": ["album id"]},
            "MediaTypeId": {"desc": "media type id foreign key", "syns": ["media type id"]},
            "GenreId"    : {"desc": "genre id foreign key", "syns": ["genre id"]},
            "Composer"   : {"desc": "composer writer songwriter", "syns": ["composer", "writer", "songwriter", "author"]},
            "Milliseconds": {"desc": "duration length time milliseconds seconds minutes", "syns": ["duration", "length", "time", "long"]},
            "Bytes"      : {"desc": "file size bytes", "syns": ["size", "bytes", "file size"]},
            "UnitPrice"  : {"desc": "price cost per track", "syns": ["price", "cost", "unit price"]},
        }
    },
}

# Foreign key join paths
CHINOOK_JOINS = [
    {"from": "Album",        "to": "Artist",       "on": "Album.ArtistId = Artist.ArtistId"},
    {"from": "Track",        "to": "Album",        "on": "Track.AlbumId = Album.AlbumId"},
    {"from": "Track",        "to": "Genre",        "on": "Track.GenreId = Genre.GenreId"},
    {"from": "Track",        "to": "MediaType",    "on": "Track.MediaTypeId = MediaType.MediaTypeId"},
    {"from": "InvoiceLine",  "to": "Track",        "on": "InvoiceLine.TrackId = Track.TrackId"},
    {"from": "InvoiceLine",  "to": "Invoice",      "on": "InvoiceLine.InvoiceId = Invoice.InvoiceId"},
    {"from": "Invoice",      "to": "Customer",     "on": "Invoice.CustomerId = Customer.CustomerId"},
    {"from": "Customer",     "to": "Employee",     "on": "Customer.SupportRepId = Employee.EmployeeId"},
    {"from": "PlaylistTrack","to": "Playlist",     "on": "PlaylistTrack.PlaylistId = Playlist.PlaylistId"},
    {"from": "PlaylistTrack","to": "Track",        "on": "PlaylistTrack.TrackId = Track.TrackId"},
]

# Known values in the database for value linking
CHINOOK_VALUES = {
    "Genre.Name"      : ["Rock","Jazz","Metal","Alternative & Punk","Rock And Roll",
                         "Blues","Latin","Reggae","Pop","Soundtrack","Bossa Nova",
                         "Easy Listening","Heavy Metal","R&B/Soul","Electronica/Dance",
                         "World","Hip Hop/Rap","Science Fiction","TV Shows","Sci Fi & Fantasy",
                         "Drama","Comedy","Alternative","Classical","Opera"],
    "MediaType.Name"  : ["MPEG audio file","Protected AAC audio file",
                         "Protected MPEG-4 video file","Purchased AAC audio file",
                         "AAC audio file"],
    "Customer.Country": ["Argentina","Australia","Austria","Belgium","Brazil",
                         "Canada","Chile","Czech Republic","Denmark","Finland",
                         "France","Germany","Hungary","India","Ireland","Italy",
                         "Netherlands","Norway","Poland","Portugal","Spain",
                         "Sweden","United Kingdom","USA"],
    "Employee.Title"  : ["General Manager","Sales Manager","Sales Support Agent",
                         "IT Manager","IT Staff"],
}


# ═══════════════════════════════════════════════════════════════════════════
# Tokenizer (simple, no NLTK dependency)
# ═══════════════════════════════════════════════════════════════════════════

STOPWORDS = {"a","an","the","is","are","was","were","be","been","being",
             "have","has","had","do","does","did","will","would","could",
             "should","may","might","shall","can","need","dare","ought",
             "used","to","of","in","on","at","by","for","with","about",
             "against","between","into","through","during","before","after",
             "above","below","from","up","down","out","off","over","under",
             "again","further","then","once","here","there","when","where",
             "why","how","all","both","each","few","more","most","other",
             "some","such","no","nor","not","only","own","same","so","than",
             "too","very","just","but","and","or","as","i","me","my","we",
             "our","you","your","he","his","she","her","it","its","they",
             "their","what","which","who","whom","this","that","these","those"}

def _tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t and t not in STOPWORDS and len(t) > 1]


# ═══════════════════════════════════════════════════════════════════════════
# TF-IDF Schema Linker
# ═══════════════════════════════════════════════════════════════════════════

class ChinookSchemaLinker:
    """
    TF-IDF based schema linker for Chinook database.

    Replaces hardcoded TABLE_KEYWORDS with semantic similarity.
    Covers tables, columns, and values.

    Usage:
        linker = ChinookSchemaLinker()
        result = linker.link("show customers from usa spending more than 40")
        # result.tables  = [("Customer", 0.92), ("Invoice", 0.71)]
        # result.columns = [("Customer", "Country", 0.88), ...]
        # result.filters = [{"column": "Country", "op": "=", "value": "USA"}]
    """

    def __init__(self):
        self._build_tfidf_index()
        self._build_synonym_index()
        self._build_value_index()

    # ── Build indices ──────────────────────────────────────────────────────

    def _build_tfidf_index(self):
        """Build TF-IDF index over table descriptions + column descriptions."""

        # Create one document per table
        self._docs = {}
        self._tokens = {}

        for table, info in CHINOOK_SCHEMA.items():
            parts = [table, info["description"]]
            parts += info["synonyms"]
            for col, cinfo in info["columns"].items():
                parts += [col, cinfo["desc"]] + cinfo["syns"]
            doc = " ".join(parts)
            toks = _tokenize(doc)
            self._docs[table]   = doc
            self._tokens[table] = toks

        # IDF
        N = len(self._tokens)
        df = Counter()
        for toks in self._tokens.values():
            for t in set(toks):
                df[t] += 1
        self._idf = {t: math.log((N + 1) / (df_t + 1)) + 1
                     for t, df_t in df.items()}

        # TF-IDF vectors per table
        all_terms        = sorted(self._idf.keys())
        self._term2idx   = {t: i for i, t in enumerate(all_terms)}
        self._table_vecs = {t: self._vec(toks)
                            for t, toks in self._tokens.items()}

    def _build_synonym_index(self):
        """Exact synonym lookup for fast matching."""
        self._tbl_syn  = {}   # word → table
        self._col_syn  = {}   # word → (table, col)

        for table, info in CHINOOK_SCHEMA.items():
            self._tbl_syn[table.lower()] = table
            for syn in info["synonyms"]:
                self._tbl_syn[syn.lower()] = table

            for col, cinfo in info["columns"].items():
                self._col_syn[col.lower()] = (table, col)
                for syn in cinfo["syns"]:
                    self._col_syn[syn.lower()] = (table, col)

    def _build_value_index(self):
        """Build lowercase value → (table.col, original_value) map."""
        self._val_idx = {}
        for col_path, values in CHINOOK_VALUES.items():
            for v in values:
                self._val_idx[v.lower()] = (col_path, v)
                # also index individual words of multi-word values
                for word in v.lower().split():
                    if len(word) > 2 and word not in STOPWORDS:
                        if word not in self._val_idx:
                            self._val_idx[word] = (col_path, v)

    # ── Vector helpers ─────────────────────────────────────────────────────

    def _vec(self, tokens: list) -> list:
        v    = [0.0] * len(self._term2idx)
        cnt  = Counter(tokens)
        dlen = max(len(tokens), 1)
        for tok, c in cnt.items():
            if tok in self._term2idx:
                tf       = c / dlen
                v[self._term2idx[tok]] = tf * self._idf.get(tok, 1.0)
        norm = math.sqrt(sum(x*x for x in v))
        if norm > 0:
            v = [x/norm for x in v]
        return v

    def _cosine(self, v1, v2) -> float:
        return sum(a*b for a, b in zip(v1, v2))

    # ── Public API ─────────────────────────────────────────────────────────

    def link(self, query: str, top_k_tables: int = 3) -> "LinkResult":
        """
        Main entry point.
        Returns LinkResult with tables, columns, filters, join_path.
        """
        q_lower = query.lower()
        tokens  = _tokenize(query)
        q_vec   = self._vec(tokens)

        # ── Level 1: Table linking ─────────────────────────────────────
        tfidf_scores   = {t: self._cosine(q_vec, tv)
                          for t, tv in self._table_vecs.items()}
        synonym_scores = defaultdict(float)

        for tok in tokens:
            if tok in self._tbl_syn:
                synonym_scores[self._tbl_syn[tok]] += 1.0
            if tok in self._col_syn:
                synonym_scores[self._col_syn[tok][0]] += 0.5
            # partial match
            for syn, tbl in self._tbl_syn.items():
                if len(syn) > 3 and (tok in syn or syn in tok):
                    synonym_scores[tbl] += 0.3

        # normalise synonym scores
        max_s = max(synonym_scores.values(), default=1.0)
        if max_s > 0:
            synonym_scores = {k: v/max_s for k, v in synonym_scores.items()}

        # combined score (60% tfidf, 40% synonym)
        combined = {}
        for t in CHINOOK_SCHEMA:
            combined[t] = (0.6 * tfidf_scores.get(t, 0) +
                           0.4 * synonym_scores.get(t, 0))

        tables = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        tables = [(t, round(s, 4)) for t, s in tables[:top_k_tables] if s > 0.01]

        # ── Level 2: Column linking ────────────────────────────────────
        columns = []
        col_match_tables = defaultdict(list)

        for tok in tokens:
            if tok in self._col_syn:
                tbl, col = self._col_syn[tok]
                columns.append((tbl, col, 1.0))
                col_match_tables[tbl].append(col)

        # also score columns in top tables by TF-IDF similarity
        for tbl, _ in tables[:2]:
            for col, cinfo in CHINOOK_SCHEMA[tbl]["columns"].items():
                col_tokens = _tokenize(col + " " + cinfo["desc"])
                score = sum(1 for t in tokens if t in col_tokens)
                if score > 0 and (tbl, col, score) not in columns:
                    columns.append((tbl, col, round(score * 0.5, 2)))

        # deduplicate columns
        seen_cols = set()
        unique_cols = []
        for tbl, col, sc in sorted(columns, key=lambda x: -x[2]):
            key = (tbl, col)
            if key not in seen_cols:
                seen_cols.add(key)
                unique_cols.append((tbl, col, sc))
        columns = unique_cols

        # ── Level 3: Value linking ─────────────────────────────────────
        filters = []
        for tok in tokens:
            if tok in self._val_idx:
                col_path, original = self._val_idx[tok]
                tbl, col = col_path.split(".")
                filters.append({
                    "table" : tbl,
                    "column": col,
                    "op"    : "=",
                    "value" : original,
                    "score" : 0.9,
                })

        # also check multi-word values
        for col_path, values in CHINOOK_VALUES.items():
            for v in values:
                if v.lower() in q_lower:
                    tbl, col = col_path.split(".")
                    if not any(f["value"] == v for f in filters):
                        filters.append({
                            "table" : tbl,
                            "column": col,
                            "op"    : "=",
                            "value" : v,
                            "score" : 1.0,
                        })

        # ── Join path ──────────────────────────────────────────────────
        table_names = [t for t, _ in tables]
        join_path   = self._find_joins(table_names)

        return LinkResult(
            query      = query,
            tables     = tables,
            columns    = columns,
            filters    = filters,
            join_path  = join_path,
            tfidf_scores   = tfidf_scores,
            synonym_scores = dict(synonym_scores),
        )

    def _find_joins(self, tables: list) -> list:
        """BFS to find join conditions between requested tables."""
        if len(tables) <= 1:
            return []
        visited    = {tables[0]}
        remaining  = set(tables[1:])
        joins_used = []
        for _ in range(10):
            if not remaining:
                break
            for j in CHINOOK_JOINS:
                if j["from"] in visited and j["to"] in remaining:
                    joins_used.append(j["on"])
                    visited.add(j["to"])
                    remaining.discard(j["to"])
                elif j["to"] in visited and j["from"] in remaining:
                    joins_used.append(j["on"])
                    visited.add(j["from"])
                    remaining.discard(j["from"])
        return joins_used

    def detect_table(self, query: str) -> str:
        """
        Fast single-table detection.
        Drop-in replacement for hardcoded TABLE_KEYWORDS.
        Returns the most likely table name or None.
        """
        result = self.link(query, top_k_tables=1)
        if result.tables and result.tables[0][1] > 0.05:
            return result.tables[0][0]
        return None

    def get_table_schema(self, table_name: str) -> dict:
        """Compatibility method for app.py sidebar."""
        if table_name not in CHINOOK_SCHEMA:
            return None
        info = CHINOOK_SCHEMA[table_name]
        return {
            "table"  : table_name,
            "columns": [
                {
                    "name"       : col,
                    "type"       : "TEXT" if "id" not in col.lower() else "INTEGER",
                    "description": cinfo["desc"],
                }
                for col, cinfo in info["columns"].items()
            ]
        }


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════

class LinkResult:
    def __init__(self, query, tables, columns, filters,
                 join_path, tfidf_scores, synonym_scores):
        self.query         = query
        self.tables        = tables        # [(table, score), ...]
        self.columns       = columns       # [(table, col, score), ...]
        self.filters       = filters       # [{"table","column","op","value"}, ...]
        self.join_path     = join_path     # ["T1.col = T2.col", ...]
        self.tfidf_scores  = tfidf_scores
        self.synonym_scores= synonym_scores

    def top_table(self) -> str:
        return self.tables[0][0] if self.tables else None

    def top_tables(self, n=2) -> list:
        return [t for t, _ in self.tables[:n]]

    def to_schema_context(self) -> str:
        """Format for BERT input / display."""
        parts = []
        for tbl, score in self.tables:
            cols = CHINOOK_SCHEMA[tbl]["columns"]
            col_names = " | ".join(cols.keys())
            parts.append(f"{tbl} : {col_names}")
        return " ; ".join(parts)

    def to_debug_dict(self) -> dict:
        """For display in Streamlit Schema Retrieval panel."""
        return {
            "tables": [
                {
                    "table"         : t,
                    "score"         : s,
                    "tfidf_score"   : round(self.tfidf_scores.get(t, 0), 4),
                    "synonym_score" : round(self.synonym_scores.get(t, 0), 4),
                    "matched_columns": [c for tb, c, _ in self.columns if tb == t],
                }
                for t, s in self.tables
            ],
            "columns": [
                {"table": tb, "column": c, "score": s}
                for tb, c, s in self.columns[:8]
            ],
            "filters": self.filters,
            "join_path": self.join_path,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════

_linker_instance = None

def get_schema_linker() -> ChinookSchemaLinker:
    global _linker_instance
    if _linker_instance is None:
        _linker_instance = ChinookSchemaLinker()
    return _linker_instance


# ═══════════════════════════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    linker = ChinookSchemaLinker()

    tests = [
        "show all customers from usa",
        "what is the total revenue",
        "top 5 artists by number of albums",
        "show tracks with rock genre",
        "customers who spent more than 40",
        "show billing records from germany",
        "audio formats available",
        "employees who are sales managers",
        "show invoice date and total for each customer",
        "find mpeg audio tracks",
        "select all from MediaType table",
        "give me everything from invoice",
    ]

    print("\n" + "="*65)
    print("  TF-IDF Schema Linker — Chinook")
    print("="*65)

    for q in tests:
        result = linker.link(q)
        print(f"\nQ: {q}")
        print(f"  Tables  : {result.tables[:3]}")
        print(f"  Columns : {[(t,c) for t,c,_ in result.columns[:3]]}")
        print(f"  Filters : {result.filters}")
        print(f"  Joins   : {result.join_path}")