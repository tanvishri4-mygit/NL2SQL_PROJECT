"""
sql_rules.py
============
Rule-based SQL generator for the Chinook database.
This is Layer 3 of the hybrid system.

Handles queries the neural model struggles with:
    - Complex multi-table JOINs
    - Window functions (TOP N with RANK)
    - Subqueries
    - Specific aggregation patterns

Also acts as fallback when neural model confidence < threshold.

Satisfies syllabus Module 4: Context Free Grammars, Chunking.

Usage:
    from rules.sql_rules import RuleBasedSQL
    rb = RuleBasedSQL()
    result = rb.try_generate("show top 5 customers by spending")
    if result["matched"]:
        print(result["sql"])
"""

import re
import sys
from pathlib import Path

# ensure project root in path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from config import CHINOOK_SCHEMA

# ═══════════════════════════════════════════════════════
# Keyword lists for intent + entity detection
# ═══════════════════════════════════════════════════════

# Aggregation keywords
AGG_MAP = {
    "total"   : "SUM",
    "sum"     : "SUM",
    "revenue" : "SUM",
    "average" : "AVG",
    "avg"     : "AVG",
    "mean"    : "AVG",
    "maximum" : "MAX",
    "max"     : "MAX",
    "highest" : "MAX",
    "minimum" : "MIN",
    "min"     : "MIN",
    "lowest"  : "MIN",
    "count"   : "COUNT",
    "number"  : "COUNT",
    "how many": "COUNT",
    "total number": "COUNT",
}

# Table detection keywords
TABLE_KEYWORDS = {
    "artist"       : "Artist",
    "artists"      : "Artist",
    "album"        : "Album",
    "albums"       : "Album",
    "customer"     : "Customer",
    "customers"    : "Customer",
    "employee"     : "Employee",
    "employees"    : "Employee",
    "genre"        : "Genre",
    "genres"       : "Genre",
    "invoice"      : "Invoice",
    "invoices"     : "Invoice",
    "invoice line" : "InvoiceLine",
    "invoice lines": "InvoiceLine",
    "invoiceline"  : "InvoiceLine",
    "invoicelines" : "InvoiceLine",
    "track"        : "Track",
    "tracks"       : "Track",
    "song"         : "Track",
    "songs"        : "Track",
    "playlist"     : "Playlist",
    "playlists"    : "Playlist",
    "media type"   : "MediaType",
    "media types"  : "MediaType",
    "mediatype"    : "MediaType",
    "mediatypes"   : "MediaType",
    "media"        : "MediaType",
    "sale"         : "Invoice",
    "sales"        : "Invoice",
    "purchase"     : "Invoice",
    "purchases"    : "Invoice",
    "spending"     : "Invoice",
}

# Column detection keywords
COLUMN_KEYWORDS = {
    "name"       : "Name",
    "title"      : "Title",
    "total"      : "Total",
    "price"      : "UnitPrice",
    "unit price" : "UnitPrice",
    "quantity"   : "Quantity",
    "country"    : "Country",
    "city"       : "City",
    "email"      : "Email",
    "date"       : "InvoiceDate",
    "composer"   : "Composer",
    "duration"   : "Milliseconds",
    "length"     : "Milliseconds",
    "billing country": "BillingCountry",
    "billing city"   : "BillingCity",
}

# Comparison operators
OPERATOR_MAP = {
    "greater than"   : ">",
    "more than"      : ">",
    "above"          : ">",
    "over"           : ">",
    "less than"      : "<",
    "below"          : "<",
    "under"          : "<",
    "equal to"       : "=",
    "equals"         : "=",
    "is"             : "=",
}

# Country names for WHERE detection
COUNTRIES = [
    "usa", "united states", "brazil", "canada", "germany",
    "france", "uk", "united kingdom", "india", "australia",
    "argentina", "portugal", "spain", "italy", "netherlands",
    "norway", "sweden", "finland", "denmark", "poland",
    "austria", "belgium", "czech republic", "hungary",
]


# ═══════════════════════════════════════════════════════
# Rule-based SQL generator
# ═══════════════════════════════════════════════════════

class RuleBasedSQL:
    """
    CFG-inspired rule-based SQL generator for Chinook.
    Tries to match the question against known patterns.
    Returns matched=True with SQL if pattern found,
    matched=False otherwise (neural model takes over).
    """

    def __init__(self):
        # Load schema linker — replaces hardcoded TABLE_KEYWORDS
        try:
            from schema_linker import get_schema_linker
            self._linker = get_schema_linker()
            self._use_linker = True
        except Exception:
            self._linker = None
            self._use_linker = False

        # ordered list of (pattern_name, method)
        self.rules = [
            # most specific first — prevents false matches
            ("top_n_join",           self._top_n_join),
            ("top_n_single",         self._top_n_single),
            ("where_country",        self._where_country),
            ("where_numeric",        self._where_numeric),
            ("avg_groupby",          self._avg_groupby),
            ("total_revenue",        self._total_revenue),
            ("sum_groupby",          self._sum_groupby),
            ("count_groupby",        self._count_groupby),
            ("distinct_query",       self._distinct_query),  # before count_all
            ("count_all",            self._count_all),
            ("join_two_tables",      self._join_two_tables),
            ("agg_single",           self._agg_single),
            ("select_all",           self._select_all),
        ]

    # ── Public interface ──────────────────────────────

    def try_generate(self, question: str) -> dict:
        """
        Try to generate SQL using rules.
        Returns:
            matched : bool
            sql     : str (if matched)
            rule    : str (which rule matched)
        """
        q = question.lower().strip()

        for rule_name, rule_fn in self.rules:
            result = rule_fn(q)
            if result:
                return {
                    "matched": True,
                    "sql"    : result,
                    "rule"   : rule_name,
                }

        return {"matched": False, "sql": "", "rule": "none"}

    # ── Helper methods ────────────────────────────────

    def _detect_table(self, q: str) -> str:
        """
        Detect primary table using TF-IDF schema linker.
        Falls back to keyword matching if linker unavailable.
        """
        if self._use_linker:
            table = self._linker.detect_table(q)
            if table:
                return table
        # fallback keyword matching
        for keyword, table in sorted(
            TABLE_KEYWORDS.items(), key=lambda x: -len(x[0])
        ):
            if keyword in q:
                return table
        return None

    def _detect_n(self, q: str, default: int = 5) -> int:
        """Extract number from 'top N' or 'first N' pattern."""
        m = re.search(r'top\s+(\d+)|first\s+(\d+)|(\d+)\s+most', q)
        if m:
            return int(next(x for x in m.groups() if x))
        return default

    def _detect_country(self, q: str) -> str:
        """
        Detect country using TF-IDF value linking.
        Falls back to COUNTRIES list.
        """
        if self._use_linker:
            result = self._linker.link(q)
            for f in result.filters:
                if f["column"] in ("Country", "BillingCountry"):
                    return f["value"]
        # fallback
        for country in COUNTRIES:
            if country in q:
                return country.title().replace("Usa","USA").replace("Uk","UK")
        return None

    def _detect_number(self, q: str) -> str:
        """Extract numeric value from question."""
        m = re.search(r'\b(\d+(?:\.\d+)?)\b', q)
        return m.group(1) if m else None

    def _detect_operator(self, q: str) -> str:
        """Detect comparison operator."""
        for phrase, op in OPERATOR_MAP.items():
            if phrase in q:
                return op
        return ">"

    # ── Rules ─────────────────────────────────────────

    def _top_n_join(self, q: str) -> str:
        """
        Pattern: top N [table] by [metric that requires join]
        e.g. "top 5 customers by total spending"
             "top 10 artists by number of albums"
             "top 5 genres by track count"
        """
        if "top" not in q and "most" not in q:
            return None

        n = self._detect_n(q)

        # top customers by spending/revenue/total
        if ("customer" in q and
                any(w in q for w in ["spending","revenue","total","invoice"])):
            return (
                f"SELECT Customer.FirstName, Customer.LastName, "
                f"SUM(Invoice.Total) AS TotalSpending "
                f"FROM Customer "
                f"JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId "
                f"GROUP BY Customer.CustomerId "
                f"ORDER BY TotalSpending DESC "
                f"LIMIT {n}"
            )

        # top artists by number of albums
        if "artist" in q and "album" in q:
            return (
                f"SELECT Artist.Name, COUNT(*) AS AlbumCount "
                f"FROM Artist "
                f"JOIN Album ON Artist.ArtistId = Album.ArtistId "
                f"GROUP BY Artist.ArtistId "
                f"ORDER BY AlbumCount DESC "
                f"LIMIT {n}"
            )

        # top genres by track count
        if "genre" in q and any(w in q for w in ["track","song","count"]):
            return (
                f"SELECT Genre.Name, COUNT(*) AS TrackCount "
                f"FROM Genre "
                f"JOIN Track ON Genre.GenreId = Track.GenreId "
                f"GROUP BY Genre.GenreId "
                f"ORDER BY TrackCount DESC "
                f"LIMIT {n}"
            )

        # top albums by track count
        if "album" in q and any(w in q for w in ["track","song","count"]):
            return (
                f"SELECT Album.Title, COUNT(*) AS TrackCount "
                f"FROM Album "
                f"JOIN Track ON Album.AlbumId = Track.AlbumId "
                f"GROUP BY Album.AlbumId "
                f"ORDER BY TrackCount DESC "
                f"LIMIT {n}"
            )

        # top countries by revenue/sales
        if "countr" in q and any(w in q for w in ["revenue","sales","total"]):
            return (
                f"SELECT BillingCountry, SUM(Total) AS TotalRevenue "
                f"FROM Invoice "
                f"GROUP BY BillingCountry "
                f"ORDER BY TotalRevenue DESC "
                f"LIMIT {n}"
            )

        return None

    def _top_n_single(self, q: str) -> str:
        """
        Pattern: top N [table] by [column in same table]
        e.g. "top 5 invoices by total"
             "top 10 tracks by price"
        """
        if "top" not in q and "most" not in q and "highest" not in q:
            return None

        n     = self._detect_n(q)
        table = self._detect_table(q)
        if not table:
            return None

        order_col = "Total"   # default
        if table == "Track":
            order_col = "UnitPrice"
        elif table == "Invoice":
            order_col = "Total"
        elif table == "Artist":
            order_col = "Name"
        elif table == "Album":
            order_col = "Title"

        return (
            f"SELECT * FROM {table} "
            f"ORDER BY {order_col} DESC "
            f"LIMIT {n}"
        )

    def _count_groupby(self, q: str) -> str:
        """
        Pattern: count [table] by/per [group column]
        e.g. "count customers by country"
             "number of invoices per country"
        """
        if not any(w in q for w in
                   ["count","number","how many","per","by country",
                    "by city","by genre","by artist","by album"]):
            return None

        table = self._detect_table(q)
        if not table:
            return None

        # determine group column
        if "country" in q:
            if table == "Invoice":
                group_col = "BillingCountry"
                label     = "BillingCountry"
            else:
                group_col = "Country"
                label     = "Country"
        elif "city" in q:
            if table == "Invoice":
                group_col = "BillingCity"
                label     = "BillingCity"
            else:
                group_col = "City"
                label     = "City"
        elif "genre" in q and table == "Track":
            return (
                "SELECT Genre.Name, COUNT(*) AS TrackCount "
                "FROM Track "
                "JOIN Genre ON Track.GenreId = Genre.GenreId "
                "GROUP BY Genre.GenreId "
                "ORDER BY TrackCount DESC"
            )
        elif "artist" in q and table == "Album":
            return (
                "SELECT Artist.Name, COUNT(*) AS AlbumCount "
                "FROM Album "
                "JOIN Artist ON Album.ArtistId = Artist.ArtistId "
                "GROUP BY Artist.ArtistId "
                "ORDER BY AlbumCount DESC"
            )
        else:
            return None

        return (
            f"SELECT {label}, COUNT(*) AS Count "
            f"FROM {table} "
            f"GROUP BY {group_col} "
            f"ORDER BY Count DESC"
        )

    def _sum_groupby(self, q: str) -> str:
        """
        Pattern: total/revenue [metric] by [group]
        e.g. "total revenue by country"
             "total sales by genre"
        """
        # must have both an aggregation word AND a grouping word
        has_agg   = any(w in q for w in ["total","revenue","sales","sum"])
        has_group = any(w in q for w in ["by country","by genre","by customer",
                                          "by artist","by city","per country",
                                          "per genre","per customer"])
        if not has_agg or not has_group:
            return None

        if "country" in q:
            return (
                "SELECT BillingCountry, SUM(Total) AS TotalRevenue "
                "FROM Invoice "
                "GROUP BY BillingCountry "
                "ORDER BY TotalRevenue DESC"
            )

        if "genre" in q:
            return (
                "SELECT Genre.Name, "
                "SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity) AS Revenue "
                "FROM InvoiceLine "
                "JOIN Track ON InvoiceLine.TrackId = Track.TrackId "
                "JOIN Genre ON Track.GenreId = Genre.GenreId "
                "GROUP BY Genre.GenreId "
                "ORDER BY Revenue DESC"
            )

        if "customer" in q:
            return (
                "SELECT Customer.FirstName, Customer.LastName, "
                "SUM(Invoice.Total) AS TotalSpent "
                "FROM Customer "
                "JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId "
                "GROUP BY Customer.CustomerId "
                "ORDER BY TotalSpent DESC"
            )

        if "artist" in q:
            return (
                "SELECT Artist.Name, "
                "SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity) AS Revenue "
                "FROM InvoiceLine "
                "JOIN Track ON InvoiceLine.TrackId = Track.TrackId "
                "JOIN Album ON Track.AlbumId = Album.AlbumId "
                "JOIN Artist ON Album.ArtistId = Artist.ArtistId "
                "GROUP BY Artist.ArtistId "
                "ORDER BY Revenue DESC"
            )

        return None

    def _total_revenue(self, q: str) -> str:
        """
        Pattern: what is the total revenue / total sales
        Single aggregate with no GROUP BY
        """
        if not any(w in q for w in
                   ["total revenue", "total sales", "total income",
                    "overall revenue", "overall sales",
                    "how much revenue", "how much sales",
                    "sum of total", "sum of invoice"]):
            return None
        return "SELECT SUM(Total) AS TotalRevenue FROM Invoice"

    def _avg_groupby(self, q: str) -> str:
        """
        Pattern: average [metric] by [group]
        e.g. "average invoice total by country"
        """
        if not any(w in q for w in ["average","avg","mean"]):
            return None

        if "invoice" in q and "country" in q:
            return (
                "SELECT BillingCountry, AVG(Total) AS AvgInvoice "
                "FROM Invoice "
                "GROUP BY BillingCountry "
                "ORDER BY AvgInvoice DESC"
            )

        return None

    def _count_all(self, q: str) -> str:
        """
        Pattern: how many [table] are there
        e.g. "how many customers are there"
        """
        if not any(w in q for w in
                   ["how many","count","total number","number of"]):
            return None

        table = self._detect_table(q)
        if not table:
            return None

        return f"SELECT COUNT(*) FROM {table}"

    def _distinct_query(self, q: str) -> str:
        """
        Pattern: distinct/unique list of [column] from [table]
        e.g. "show distinct list of all countries of customers"
             "list unique countries"
             "what are the different genres"
        """
        # MUST have an explicit distinct/unique/different keyword
        # "all countries" alone is not enough — too ambiguous
        has_distinct = any(w in q for w in
                           ["distinct", "unique", "different"])
        if not has_distinct:
            return None

        # country queries
        if "countr" in q:
            if "billing" in q:
                return "SELECT DISTINCT BillingCountry FROM Invoice ORDER BY BillingCountry"
            return "SELECT DISTINCT Country FROM Customer ORDER BY Country"

        # city queries
        if "cit" in q:
            return "SELECT DISTINCT City FROM Customer ORDER BY City"

        # genre queries
        if "genre" in q:
            return "SELECT DISTINCT Name FROM Genre ORDER BY Name"

        # artist queries
        if "artist" in q:
            return "SELECT DISTINCT Name FROM Artist ORDER BY Name"

        # media type queries
        if "media" in q:
            return "SELECT DISTINCT Name FROM MediaType ORDER BY Name"

        return None

    def _agg_single(self, q: str) -> str:
        """
        Pattern: what is the [agg] [column]
        e.g. "what is the total revenue"
             "what is the average invoice total"
        """
        # must have a clear aggregation intent word
        # exclude "minutes/minimum" false positives
        agg_trigger_words = [
            "what is the total", "what is the average", "what is the avg",
            "what is the max", "what is the min", "what is the maximum",
            "what is the minimum", "find the total", "find the average",
            "find the max", "find the min", "find the maximum",
            "find the minimum", "total revenue", "total sales",
            "average invoice", "average track", "maximum invoice",
            "minimum invoice", "maximum track", "minimum track",
        ]
        if not any(trigger in q for trigger in agg_trigger_words):
            return None

        # also skip if "by" present — that means GROUP BY, handled elsewhere
        if " by " in q:
            return None

        table = self._detect_table(q)
        if not table:
            return None

        for phrase, agg_fn in sorted(
            AGG_MAP.items(), key=lambda x: -len(x[0])
        ):
            if phrase in q:
                if table == "Invoice" and agg_fn in ("SUM","AVG","MAX","MIN"):
                    col = "Total"
                elif table == "Track" and agg_fn in ("AVG","MAX","MIN"):
                    col = "UnitPrice"
                else:
                    col = "*" if agg_fn == "COUNT" else "Total"

                return f"SELECT {agg_fn}({col}) FROM {table}"

        return None

    def _where_country(self, q: str) -> str:
        """
        Pattern: show [table] from [country]
        e.g. "show customers from usa"
        """
        table   = self._detect_table(q)
        country = self._detect_country(q)

        if not table or not country:
            return None

        if table == "Customer":
            return (
                f"SELECT * FROM Customer "
                f"WHERE Country = '{country}'"
            )
        elif table == "Invoice":
            return (
                f"SELECT * FROM Invoice "
                f"WHERE BillingCountry = '{country}'"
            )

        return None

    def _where_numeric(self, q: str) -> str:
        """
        Pattern: show [table] where [col] [op] [number]
        e.g. "show invoices with total greater than 10"
        """
        # skip duration queries — too complex for simple rule
        if any(w in q for w in
               ["minute", "second", "hour", "duration",
                "longer", "shorter", "length"]):
            return None

        table  = self._detect_table(q)
        number = self._detect_number(q)
        op     = self._detect_operator(q)

        if not table or not number:
            return None

        if table == "Invoice":
            return (
                f"SELECT * FROM Invoice "
                f"WHERE Total {op} {number}"
            )
        elif table == "Track":
            return (
                f"SELECT * FROM Track "
                f"WHERE UnitPrice {op} {number}"
            )

        return None

    def _select_all(self, q: str) -> str:
        """
        Pattern: show/list/select all [table]
        e.g. "show all customers"
             "list all tracks"
             "select all from Invoice table"
             "get all artists"
        """
        # must have an "all" or "every" or "from X table" pattern
        has_trigger = any(w in q for w in [
            "show all", "list all", "display all",
            "show every", "get all", "fetch all",
            "select all", "give all", "print all",
            "select * from", "select all from",
            "from invoice table", "from customer table",
            "from artist table", "from album table",
            "from track table", "from genre table",
            "from employee table", "from playlist table",
            "from mediatype table", "from invoiceline table",
        ])
        if not has_trigger:
            return None

        table = self._detect_table(q)
        if not table:
            return None

        return f"SELECT * FROM {table}"

    def _join_two_tables(self, q: str) -> str:
        """
        Pattern: show [table1] with [table2 info]
        e.g. "show albums with artist names"
             "show tracks with genre names"
             "show customers with their invoices"

        Does NOT fire if query has filter conditions —
        those should go to neural model.
        """
        # if query has filter words — pass to neural model
        filter_words = [
            "where", "when", "date", "year", "month", "after",
            "before", "since", "between", "greater", "less",
            "above", "below", "more than", "less than",
            "equal", "like", "in 2009", "in 2010", "in 2011",
            "in 2012", "in 2013", "from 2", "at least", "at most"
        ]
        if any(w in q for w in filter_words):
            return None

        # album + artist
        if "album" in q and "artist" in q:
            return (
                "SELECT Album.Title, Artist.Name "
                "FROM Album "
                "JOIN Artist ON Album.ArtistId = Artist.ArtistId"
            )

        # track + genre
        if "track" in q and "genre" in q:
            return (
                "SELECT Track.Name, Genre.Name AS GenreName "
                "FROM Track "
                "JOIN Genre ON Track.GenreId = Genre.GenreId"
            )

        # track + album
        if "track" in q and "album" in q:
            return (
                "SELECT Track.Name, Album.Title "
                "FROM Track "
                "JOIN Album ON Track.AlbumId = Album.AlbumId"
            )

        # customer + invoice
        if "customer" in q and any(
            w in q for w in ["invoice","purchase","order","spending"]
        ):
            return (
                "SELECT Customer.FirstName, Customer.LastName, "
                "Invoice.InvoiceDate, Invoice.Total "
                "FROM Customer "
                "JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId"
            )

        # invoice + track (via invoice lines)
        if "invoice" in q and "track" in q:
            return (
                "SELECT Invoice.InvoiceId, Track.Name, "
                "InvoiceLine.UnitPrice, InvoiceLine.Quantity "
                "FROM Invoice "
                "JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId "
                "JOIN Track ON InvoiceLine.TrackId = Track.TrackId"
            )

        return None


# ═══════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    rb = RuleBasedSQL()

    tests = [
        # should match rules
        "show all customers",
        "how many artists are there",
        "show customers from usa",
        "show invoices with total greater than 10",
        "count customers by country",
        "total revenue by country",
        "top 5 customers by total spending",
        "top 5 artists by number of albums",
        "top 5 genres by track count",
        "show albums with artist names",
        "show tracks with genre names",
        "average invoice total by country",
        # should NOT match (neural model handles)
        "show me tracks that are longer than 5 minutes",
        "find customers who have never bought anything",
    ]

    print("Rule-Based SQL Generator — Chinook\n")
    print("="*55)
    for q in tests:
        result = rb.try_generate(q)
        status = "✅ MATCHED" if result["matched"] else "❌ no match"
        print(f"\nQ    : {q}")
        print(f"Rule : {result['rule']:20s}  {status}")
        if result["matched"]:
            print(f"SQL  : {result['sql']}")
