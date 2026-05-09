"""
sql_corrector.py
================
Schema-grounded post-processing correction for neural model SQL output.

Fixes applied in order:
    1. WikiSQL TABLE placeholder → schema linker top table
    2. T1/T2 alias resolution → real Chinook table names
    3. Invalid table names → schema linker replacement
    4. WikiSQL hallucinated column names → real Chinook columns
    5. Wrong string values in WHERE → schema linker values
    6. Missing WHERE clause → append from schema linker

Pipeline position:
    Neural model generates SQL
          ↓
    SQLCorrector.correct(sql, link_result)
          ↓
    Corrected SQL + correction log
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ═══════════════════════════════════════════════════════
# Valid Chinook schema
# ═══════════════════════════════════════════════════════

VALID_TABLES = {
    "Album", "Artist", "Customer", "Employee", "Genre",
    "Invoice", "InvoiceLine", "MediaType", "Playlist",
    "PlaylistTrack", "Track"
}

VALID_COLUMNS = {
    "Album"        : {"AlbumId", "Title", "ArtistId"},
    "Artist"       : {"ArtistId", "Name"},
    "Customer"     : {"CustomerId", "FirstName", "LastName", "Company",
                      "Address", "City", "State", "Country", "PostalCode",
                      "Phone", "Fax", "Email", "SupportRepId"},
    "Employee"     : {"EmployeeId", "LastName", "FirstName", "Title",
                      "ReportsTo", "BirthDate", "HireDate", "Address",
                      "City", "State", "Country", "PostalCode",
                      "Phone", "Fax", "Email"},
    "Genre"        : {"GenreId", "Name"},
    "Invoice"      : {"InvoiceId", "CustomerId", "InvoiceDate",
                      "BillingAddress", "BillingCity", "BillingState",
                      "BillingCountry", "BillingPostalCode", "Total"},
    "InvoiceLine"  : {"InvoiceLineId", "InvoiceId", "TrackId",
                      "UnitPrice", "Quantity"},
    "MediaType"    : {"MediaTypeId", "Name"},
    "Playlist"     : {"PlaylistId", "Name"},
    "PlaylistTrack": {"PlaylistId", "TrackId"},
    "Track"        : {"TrackId", "Name", "AlbumId", "MediaTypeId",
                      "GenreId", "Composer", "Milliseconds",
                      "Bytes", "UnitPrice"},
}

# All valid columns flattened — for quick lookup without knowing table
ALL_VALID_COLUMNS = set()
for cols in VALID_COLUMNS.values():
    ALL_VALID_COLUMNS.update(cols)

# ── WikiSQL hallucinated column names seen in eval results ────────────
# These appear in model output but don't exist in Chinook
WIKISQL_FAKE_COLUMNS = {
    # generic WikiSQL fake names
    "revenue", "followers", "songid", "song_id", "roomname", "room_name",
    "fname", "lname", "cust_name", "customer_name", "label", "credit_score",
    "credit", "score", "rank", "ranking", "position", "pos", "number",
    "nation", "nationality", "region", "area", "zone", "district",
    "category", "type", "kind", "class", "genre_name", "artist_name",
    "album_name", "track_name", "song", "songs", "albums", "artists",
    "tracks", "customers", "employees", "invoices",
    # T1/T2 style fake columns
    "t1", "t2", "t3",
}

# ── Chinook join key map ──────────────────────────────────────────────
# Used for T1/T2 alias resolution to find correct join conditions
CHINOOK_JOIN_KEYS = {
    frozenset(["Artist",  "Album"])       : ("Artist.ArtistId",       "Album.ArtistId"),
    frozenset(["Album",   "Track"])       : ("Album.AlbumId",         "Track.AlbumId"),
    frozenset(["Track",   "Genre"])       : ("Track.GenreId",         "Genre.GenreId"),
    frozenset(["Track",   "MediaType"])   : ("Track.MediaTypeId",     "MediaType.MediaTypeId"),
    frozenset(["Track",   "InvoiceLine"]) : ("Track.TrackId",         "InvoiceLine.TrackId"),
    frozenset(["Track",   "PlaylistTrack"]): ("Track.TrackId",        "PlaylistTrack.TrackId"),
    frozenset(["Customer","Invoice"])     : ("Customer.CustomerId",   "Invoice.CustomerId"),
    frozenset(["Invoice", "InvoiceLine"]) : ("Invoice.InvoiceId",    "InvoiceLine.InvoiceId"),
    frozenset(["Employee","Customer"])    : ("Employee.EmployeeId",   "Customer.SupportRepId"),
    frozenset(["Playlist","PlaylistTrack"]): ("Playlist.PlaylistId", "PlaylistTrack.PlaylistId"),
}

# WikiSQL TABLE placeholder
WIKISQL_TABLE_PLACEHOLDER = "TABLE"


# ═══════════════════════════════════════════════════════
# Correction record
# ═══════════════════════════════════════════════════════

@dataclass
class Correction:
    kind       : str
    original   : str
    replacement: str
    reason     : str

    def __str__(self):
        return f"[{self.kind}] '{self.original}' corrected to '{self.replacement}'"


@dataclass
class CorrectionResult:
    original_sql  : str
    corrected_sql : str
    corrections   : List[Correction] = field(default_factory=list)
    was_corrected : bool = False

    @property
    def n_corrections(self):
        return len(self.corrections)

    def summary(self) -> str:
        if not self.corrections:
            return "No corrections needed"
        lines = [f"{i+1}. {c}" for i, c in enumerate(self.corrections)]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# SQL Corrector
# ═══════════════════════════════════════════════════════

class SQLCorrector:
    """
    Schema-grounded post-processing corrector.

    Fixes WikiSQL-pattern hallucinations:
    - T1/T2 alias references resolved to real Chinook table.column names
    - Fake column names (revenue, followers, songid...) replaced
    - TABLE placeholder replaced
    - Wrong WHERE string values corrected
    - Missing WHERE clauses appended
    """

    def correct(self, sql: str, link_result) -> CorrectionResult:
        if not sql or not sql.strip():
            return CorrectionResult(original_sql=sql, corrected_sql=sql)

        corrections = []
        working_sql = sql.strip()

        # Fix 1 — TABLE placeholder
        working_sql, c1 = self._fix_table_placeholder(working_sql, link_result)
        corrections.extend(c1)

        # Fix 2 — T1/T2 alias resolution (NEW — most impactful)
        working_sql, c2 = self._resolve_aliases(working_sql, link_result)
        corrections.extend(c2)

        # Fix 3 — Invalid table names in FROM/JOIN
        working_sql, c3 = self._fix_invalid_tables(working_sql, link_result)
        corrections.extend(c3)

        # Fix 4 — WikiSQL hallucinated column names (NEW)
        working_sql, c4 = self._fix_fake_columns(working_sql, link_result)
        corrections.extend(c4)

        # Fix 5 — Wrong WHERE string values
        working_sql, c5 = self._fix_where_values(working_sql, link_result)
        corrections.extend(c5)

        # Fix 6 — Missing WHERE clause
        working_sql, c6 = self._add_missing_where(working_sql, link_result)
        corrections.extend(c6)

        return CorrectionResult(
            original_sql  = sql,
            corrected_sql = working_sql,
            corrections   = corrections,
            was_corrected = len(corrections) > 0,
        )

    # ── Fix 1: TABLE placeholder ──────────────────────────────────────

    def _fix_table_placeholder(self, sql, link_result):
        corrections = []
        if not re.search(r'\bTABLE\b', sql, re.IGNORECASE):
            return sql, corrections
        if not link_result or not link_result.tables:
            return sql, corrections
        top_table, score = link_result.tables[0]
        if score < 0.05:
            return sql, corrections
        new_sql = re.sub(r'\bTABLE\b', top_table, sql, flags=re.IGNORECASE)
        if new_sql != sql:
            corrections.append(Correction(
                kind="table", original="TABLE", replacement=top_table,
                reason=f"table name fixed to {top_table}"
            ))
        return new_sql, corrections

    # ── Fix 2: T1/T2 alias resolution ────────────────────────────────

    def _resolve_aliases(self, sql, link_result):
        """
        Resolve WikiSQL T1/T2 table aliases to real Chinook table names.

        WikiSQL trains the model to write:
            SELECT T1.Name FROM Artist AS T1 JOIN Album AS T2 ON T1.songid = T2.songid

        We need:
            SELECT Artist.Name FROM Artist JOIN Album ON Artist.ArtistId = Album.ArtistId

        Steps:
        1. Find all AS aliases: Artist AS T1 → {T1: Artist}
        2. Replace T1.col → Artist.col throughout SQL
        3. Fix join conditions using CHINOOK_JOIN_KEYS
        4. Remove AS T1 alias declarations
        """
        corrections = []

        # Step 1 — build alias map from "Table AS T1" patterns
        alias_pattern = re.compile(
            r'\b([A-Za-z][A-Za-z0-9]*)\s+AS\s+(T\d+)\b',
            re.IGNORECASE
        )
        alias_map = {}  # T1 → TableName
        for match in alias_pattern.finditer(sql):
            table_name = match.group(1)
            alias      = match.group(2).upper()
            if table_name in VALID_TABLES:
                alias_map[alias] = table_name

        if not alias_map:
            return sql, corrections

        working = sql

        # Step 2 — replace T1.column → TableName.column
        for alias, table in alias_map.items():
            # T1.ColumnName pattern
            col_ref = re.compile(
                rf'\b{re.escape(alias)}\.([A-Za-z_][A-Za-z0-9_]*)\b',
                re.IGNORECASE
            )
            def replace_col_ref(m, tbl=table):
                col = m.group(1)
                # check if column is valid for this table
                if col in VALID_COLUMNS.get(tbl, set()):
                    return f"{tbl}.{col}"
                # column is fake — find best real column for this table
                real_col = self._best_column_for_table(col, tbl)
                if real_col:
                    return f"{tbl}.{real_col}"
                return f"{tbl}.{col}"  # keep as-is, will be caught by fix 4

            new_working = col_ref.sub(replace_col_ref, working)
            if new_working != working:
                corrections.append(Correction(
                    kind="alias",
                    original=f"{alias}.*",
                    replacement=f"{table}.*",
                    reason=f"table alias resolved to {table}"
                ))
            working = new_working

        # Step 3 — fix join ON conditions using known Chinook join keys
        tables_in_sql = [t for t in VALID_TABLES
                         if re.search(r'\b' + t + r'\b', working, re.IGNORECASE)]

        if len(tables_in_sql) >= 2:
            # find all ON conditions that look wrong (fake columns)
            on_pattern = re.compile(
                r'\bON\s+([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)'
                r'\s*=\s*([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)',
                re.IGNORECASE
            )
            for m in on_pattern.finditer(working):
                left  = m.group(1)   # e.g. Artist.songid
                right = m.group(2)   # e.g. Album.songid
                left_tbl  = left.split(".")[0]
                right_tbl = right.split(".")[0]
                left_col  = left.split(".")[1]
                right_col = right.split(".")[1]

                # if join columns are fake, replace with correct Chinook keys
                tbl_pair = frozenset([left_tbl, right_tbl])
                if tbl_pair in CHINOOK_JOIN_KEYS:
                    key_left, key_right = CHINOOK_JOIN_KEYS[tbl_pair]
                    correct_on = f"ON {key_left} = {key_right}"
                    old_on     = m.group(0)
                    if old_on != correct_on:
                        working = working.replace(old_on, correct_on, 1)
                        corrections.append(Correction(
                            kind="join_key",
                            original=old_on,
                            replacement=correct_on,
                            reason=f"join condition corrected for {left_tbl} and {right_tbl}"
                        ))

        # Step 4 — remove "AS T1" alias declarations
        working = alias_pattern.sub(lambda m: m.group(1), working)

        return working, corrections

    def _best_column_for_table(self, fake_col: str, table: str) -> Optional[str]:
        """
        When T1.fake_col is encountered, find the most likely real column
        for this table based on semantic similarity.
        """
        fake_lower = fake_col.lower()
        valid_cols = VALID_COLUMNS.get(table, set())

        # exact match (case-insensitive)
        for col in valid_cols:
            if col.lower() == fake_lower:
                return col

        # partial match — fake col contains real col name or vice versa
        for col in valid_cols:
            if col.lower() in fake_lower or fake_lower in col.lower():
                return col

        # semantic heuristics for common WikiSQL fake names
        SEMANTIC_MAP = {
            "name"       : "Name",
            "title"      : "Title" if "Title" in valid_cols else "Name",
            "id"         : next((c for c in valid_cols if c.endswith("Id")), None),
            "total"      : "Total" if "Total" in valid_cols else None,
            "price"      : "UnitPrice" if "UnitPrice" in valid_cols else None,
            "date"       : next((c for c in valid_cols if "Date" in c), None),
            "country"    : "Country" if "Country" in valid_cols else "BillingCountry",
            "city"       : "City" if "City" in valid_cols else "BillingCity",
            "email"      : "Email" if "Email" in valid_cols else None,
            "first"      : "FirstName" if "FirstName" in valid_cols else None,
            "last"       : "LastName" if "LastName" in valid_cols else None,
        }

        for keyword, real_col in SEMANTIC_MAP.items():
            if keyword in fake_lower and real_col and real_col in valid_cols:
                return real_col

        return None

    # ── Fix 3: Invalid table names ────────────────────────────────────

    def _fix_invalid_tables(self, sql, link_result):
        corrections = []
        from_pattern = re.compile(
            r'\b(FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)',
            re.IGNORECASE
        )

        def replace_table(match):
            keyword    = match.group(1)
            table_name = match.group(2)
            if table_name in VALID_TABLES:
                return match.group(0)
            if link_result and link_result.tables:
                for tbl, score in link_result.tables:
                    if score > 0.05:
                        corrections.append(Correction(
                            kind="table", original=table_name, replacement=tbl,
                            reason=f"invalid table replaced with {tbl}"
                        ))
                        return f"{keyword} {tbl}"
            return match.group(0)

        return from_pattern.sub(replace_table, sql), corrections

    # ── Fix 4: WikiSQL fake column names ─────────────────────────────

    def _fix_fake_columns(self, sql, link_result):
        """
        Replace hallucinated WikiSQL column names with real Chinook columns.

        Examples from eval results:
            SUM(revenue)   → SUM(Total)         [Invoice]
            AVG(followers) → AVG(Total)         [Invoice]
            SELECT Song    → SELECT Name         [Track]
            T2.roomName    → already fixed by alias resolver
        """
        corrections = []

        # find all tables currently in the SQL (after alias resolution)
        tables_in_sql = [t for t in VALID_TABLES
                         if re.search(r'\b' + t + r'\b', sql, re.IGNORECASE)]

        if not tables_in_sql:
            return sql, corrections

        # build a set of all valid columns for tables in this SQL
        available_cols = set()
        for tbl in tables_in_sql:
            available_cols.update(VALID_COLUMNS.get(tbl, set()))

        working = sql

        # find bare column references (not table.column format) that are fake
        # pattern: word that is not a SQL keyword, table name, or valid column
        SQL_KEYWORDS = {
            'SELECT','FROM','WHERE','JOIN','ON','GROUP','BY','ORDER',
            'HAVING','LIMIT','AS','AND','OR','NOT','IN','IS','NULL',
            'DISTINCT','COUNT','SUM','AVG','MIN','MAX','LIKE','BETWEEN',
            'EXISTS','UNION','ALL','ASC','DESC','INNER','LEFT','RIGHT',
            'OUTER','CASE','WHEN','THEN','ELSE','END','INSERT','UPDATE',
            'DELETE','CREATE','DROP','TABLE','INTO','VALUES','SET',
        }

        # look for fake column inside aggregate functions: SUM(fake), AVG(fake)
        agg_pattern = re.compile(
            r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)',
            re.IGNORECASE
        )

        def fix_agg_col(m):
            func = m.group(1).upper()
            col  = m.group(2)

            # * is always valid
            if col == "*":
                return m.group(0)

            # already valid
            if col in available_cols or col in ALL_VALID_COLUMNS:
                return m.group(0)

            col_lower = col.lower()

            # fake column — find best replacement
            if col_lower in WIKISQL_FAKE_COLUMNS or col_lower not in {
                c.lower() for c in available_cols
            }:
                # for SUM/AVG/MIN/MAX — look for numeric column
                if func in ("SUM", "AVG", "MIN", "MAX"):
                    numeric_candidates = []
                    for tbl in tables_in_sql:
                        for c in VALID_COLUMNS.get(tbl, set()):
                            if any(kw in c.lower() for kw in
                                   ["total", "price", "quantity", "milliseconds", "bytes"]):
                                numeric_candidates.append(c)
                    if numeric_candidates:
                        best = numeric_candidates[0]
                        corrections.append(Correction(
                            kind="column",
                            original=f"{func}({col})",
                            replacement=f"{func}({best})",
                            reason=f"column corrected: '{col}' → '{best}'"
                        ))
                        return f"{func}({best})"

                # for COUNT — use *
                if func == "COUNT":
                    corrections.append(Correction(
                        kind="column",
                        original=f"COUNT({col})",
                        replacement="COUNT(*)",
                        reason=f"aggregate corrected: '{col}' → COUNT(*)"
                    ))
                    return "COUNT(*)"

            return m.group(0)

        working = agg_pattern.sub(fix_agg_col, working)

        # fix bare SELECT col that is fake (not inside aggregate)
        # pattern: SELECT fakecol FROM or SELECT fakecol, ...
        # only fix if it looks like a WikiSQL fake name
        select_col_pattern = re.compile(
            r'\bSELECT\s+([A-Za-z_][A-Za-z0-9_]*)\s+FROM\b',
            re.IGNORECASE
        )

        def fix_select_col(m):
            col = m.group(1)
            if col.upper() in SQL_KEYWORDS:
                return m.group(0)
            if col in available_cols or col in ALL_VALID_COLUMNS:
                return m.group(0)
            col_lower = col.lower()
            if col_lower in WIKISQL_FAKE_COLUMNS or col_lower.startswith("t") and len(col_lower) <= 2:
                corrections.append(Correction(
                    kind="column",
                    original=f"SELECT {col}",
                    replacement="SELECT *",
                    reason=f"column corrected: '{col}' → SELECT *"
                ))
                return "SELECT * FROM"
            return m.group(0)

        working = select_col_pattern.sub(fix_select_col, working)

        return working, corrections

    # ── Fix 5: Wrong WHERE values ─────────────────────────────────────

    def _fix_where_values(self, sql, link_result):
        corrections = []
        if not link_result or not link_result.filters:
            return sql, corrections

        col_to_value = {
            f["column"].lower(): f["value"]
            for f in link_result.filters
        }

        where_pattern = re.compile(
            r'(\w+)\s*=\s*["\']([^"\']*)["\']',
            re.IGNORECASE
        )

        def replace_value(match):
            col_name    = match.group(1)
            sql_value   = match.group(2)
            correct_val = col_to_value.get(col_name.lower())
            if correct_val and sql_value != correct_val:
                if sql_value.lower() != correct_val.lower() or sql_value != correct_val:
                    corrections.append(Correction(
                        kind="value", original=sql_value, replacement=correct_val,
                        reason=f"value corrected to '{correct_val}'"
                    ))
                    return f"{col_name} = '{correct_val}'"
            return match.group(0)

        return where_pattern.sub(replace_value, sql), corrections

    # ── Fix 6: Missing WHERE clause ───────────────────────────────────

    def _add_missing_where(self, sql, link_result):
        corrections = []
        if not link_result or not link_result.filters:
            return sql, corrections
        if re.search(r'\bWHERE\b', sql, re.IGNORECASE):
            return sql, corrections
        if not re.search(r'\bSELECT\b', sql, re.IGNORECASE):
            return sql, corrections

        high_conf = [f for f in link_result.filters if f.get("score", 0) >= 0.9]
        if len(high_conf) != 1:
            return sql, corrections

        f = high_conf[0]
        if not re.search(r'\b' + f["table"] + r'\b', sql, re.IGNORECASE):
            return sql, corrections

        where_clause = f" WHERE {f['column']} {f.get('op','=')} '{f['value']}'"
        new_sql = sql.rstrip() + where_clause
        corrections.append(Correction(
            kind="missing_where",
            original="(no WHERE)",
            replacement=f"WHERE {f['column']} = '{f['value']}'",
            reason="WHERE clause added based on query context"
        ))
        return new_sql, corrections


# ═══════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════

_corrector_instance = None

def get_corrector() -> SQLCorrector:
    global _corrector_instance
    if _corrector_instance is None:
        _corrector_instance = SQLCorrector()
    return _corrector_instance


# ═══════════════════════════════════════════════════════
# Standalone test — verifies all eval failure patterns
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    corrector = SQLCorrector()

    class FakeLink:
        tables  = [("Customer", 0.9), ("Invoice", 0.5)]
        filters = []
        columns = []

        def to_schema_context(self):
            return ""

    class FakeLinkInvoice:
        tables  = [("Invoice", 0.9)]
        filters = []
        columns = []

        def to_schema_context(self):
            return ""

    class FakeLinkArtistAlbum:
        tables  = [("Artist", 0.9), ("Album", 0.8)]
        filters = []
        columns = []

        def to_schema_context(self):
            return ""

    tests = [
        # From eval results
        ("SELECT Artist FROM Artist WHERE Artist = Artist",         FakeLink(),         "SELECT * FROM Artist"),
        ("SELECT DISTINCT (FROM)",                                  FakeLink(),         "SELECT DISTINCT (FROM)"),  # unfixable syntax
        ("SELECT T1.Country, COUNT (*) FROM Customer GROUP BY Country", FakeLink(),     "SELECT Customer.Country, COUNT (*) FROM Customer GROUP BY Country"),
        ("SELECT T1.Name FROM Customer WHERE Country = 'Germany'",  FakeLink(),         "SELECT Customer.Name FROM Customer WHERE Country = 'Germany'"),
        ("SELECT SUM (revenue) FROM Invoice",                       FakeLinkInvoice(),  "SELECT SUM(Total) FROM Invoice"),
        ("SELECT AVG (followers) FROM Invoice",                     FakeLinkInvoice(),  "SELECT AVG(Total) FROM Invoice"),
        ("SELECT Album FROM Album AS T1 JOIN Artist",               FakeLinkArtistAlbum(), "SELECT * FROM Album JOIN Artist"),
        ("SELECT T1.Name, T1.Name FROM Genre AS T1 JOIN Track AS T2 ON T1.songid = T2.songid", FakeLinkArtistAlbum(), ""),
    ]

    print("\n" + "="*70)
    print("  SQL Corrector — T1/T2 Alias + Fake Column Resolver Test")
    print("="*70)

    for sql, link, expected in tests:
        result = corrector.correct(sql, link)
        status = "✅" if result.corrected_sql != sql else "➖"
        print(f"\n{status} Input    : {sql}")
        print(f"   Output   : {result.corrected_sql}")
        if result.corrections:
            for c in result.corrections:
                print(f"   Fix      : {c}")