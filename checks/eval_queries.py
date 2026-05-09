"""
eval_queries.py
==================
Evaluation script with 32 redesigned test cases.

Run from project root:
    conda activate nl2sql
    python checks/eval_queries.py

Output saved to: checks/eval_results.txt
"""

import os
import sys
import re
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from engine_v2 import NL2SQLEngine

TEST_CASES = [

    # ── 1. SELECT SIMPLE (4 queries) ─────────────────────────────────────
    {
        "category"    : "SELECT_SIMPLE",
        "query"       : "show all artists",
        "expected"    : "SELECT * FROM Artist",
        "must_contain": ["Artist"],
    },
    {
        "category"    : "SELECT_SIMPLE",
        "query"       : "list all genres",
        "expected"    : "SELECT * FROM Genre",
        "must_contain": ["Genre"],
    },
    {
        "category"    : "SELECT_SIMPLE",
        "query"       : "show all playlists",
        "expected"    : "SELECT * FROM Playlist",
        "must_contain": ["Playlist"],
    },
    {
        "category"    : "SELECT_SIMPLE",
        "query"       : "show all employees",
        "expected"    : "SELECT * FROM Employee",
        "must_contain": ["Employee"],
    },

    # ── 2. FILTER WHERE (4 queries) ──────────────────────────────────────
    {
        "category"    : "FILTER_WHERE",
        "query"       : "show customers from canada",
        "expected"    : "SELECT * FROM Customer WHERE Country = 'Canada'",
        "must_contain": ["Customer", "WHERE", "Country", "Canada"],
    },
    {
        "category"    : "FILTER_WHERE",
        "query"       : "show customers from brazil",
        "expected"    : "SELECT * FROM Customer WHERE Country = 'Brazil'",
        "must_contain": ["Customer", "WHERE", "Country", "Brazil"],
    },
    {
        "category"    : "FILTER_WHERE",
        "query"       : "show invoices with total greater than 20",
        "expected"    : "SELECT * FROM Invoice WHERE Total > 20",
        "must_contain": ["Invoice", "WHERE", "Total"],
    },
    {
        "category"    : "FILTER_WHERE",
        "query"       : "show tracks with price less than 1",
        "expected"    : "SELECT * FROM Track WHERE UnitPrice < 1",
        "must_contain": ["Track", "WHERE", "UnitPrice"],
    },

    # ── 3. AGGREGATE (4 queries) ─────────────────────────────────────────
    {
        "category"    : "AGGREGATE",
        "query"       : "how many albums are there",
        "expected"    : "SELECT COUNT(*) FROM Album",
        "must_contain": ["COUNT", "Album"],
    },
    {
        # This is the ONE query base model reliably passes
        "category"    : "AGGREGATE",
        "query"       : "how many customers are there",
        "expected"    : "SELECT COUNT(*) FROM Customer",
        "must_contain": ["COUNT", "Customer"],
    },
    {
        "category"    : "AGGREGATE",
        "query"       : "what is the total revenue",
        "expected"    : "SELECT SUM(Total) FROM Invoice",
        "must_contain": ["SUM", "Invoice"],
    },
    {
        "category"    : "AGGREGATE",
        "query"       : "find the average invoice total",
        "expected"    : "SELECT AVG(Total) FROM Invoice",
        "must_contain": ["AVG", "Invoice"],
    },

    # ── 4. GROUP BY (4 queries) ──────────────────────────────────────────
    {
        "category"    : "GROUP_BY",
        "query"       : "count customers by country",
        "expected"    : "SELECT Country, COUNT(*) FROM Customer GROUP BY Country",
        "must_contain": ["COUNT", "Customer", "GROUP BY", "Country"],
    },
    {
        "category"    : "GROUP_BY",
        "query"       : "total revenue by country",
        "expected"    : "SELECT BillingCountry, SUM(Total) FROM Invoice GROUP BY BillingCountry",
        "must_contain": ["SUM", "Invoice", "GROUP BY"],
    },
    {
        "category"    : "GROUP_BY",
        "query"       : "count albums by artist id",
        "expected"    : "SELECT ArtistId, COUNT(*) FROM Album GROUP BY ArtistId",
        "must_contain": ["COUNT", "Album", "GROUP BY", "ArtistId"],
    },
    {
        "category"    : "GROUP_BY",
        "query"       : "count tracks by media type id",
        "expected"    : "SELECT MediaTypeId, COUNT(*) FROM Track GROUP BY MediaTypeId",
        "must_contain": ["COUNT", "Track", "GROUP BY"],
    },

    # ── 5. ORDER BY + LIMIT (2 queries) ──────────────────────────────────
    {
        "category"    : "ORDER_LIMIT",
        "query"       : "show top 5 invoices by total",
        "expected"    : "SELECT * FROM Invoice ORDER BY Total DESC LIMIT 5",
        "must_contain": ["Invoice", "ORDER BY", "LIMIT"],
    },
    {
        "category"    : "ORDER_LIMIT",
        "query"       : "show top 5 artists alphabetically",
        "expected"    : "SELECT * FROM Artist ORDER BY Name ASC LIMIT 5",
        "must_contain": ["Artist", "ORDER BY", "LIMIT"],
    },

    # ── 6. SINGLE JOIN (4 queries) ────────────────────────────────────────
    {
        "category"    : "JOIN_SINGLE",
        "query"       : "show albums with artist names",
        "expected"    : "SELECT Album.Title, Artist.Name FROM Album JOIN Artist ON Album.ArtistId = Artist.ArtistId",
        "must_contain": ["Album", "Artist", "JOIN"],
    },
    {
        "category"    : "JOIN_SINGLE",
        "query"       : "show customer names with their invoices",
        "expected"    : "SELECT Customer.FirstName, Invoice.Total FROM Customer JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId",
        "must_contain": ["Customer", "Invoice", "JOIN"],
    },
    {
        "category"    : "JOIN_SINGLE",
        "query"       : "show track names with their genre",
        "expected"    : "SELECT Track.Name, Genre.Name FROM Track JOIN Genre ON Track.GenreId = Genre.GenreId",
        "must_contain": ["Track", "Genre", "JOIN"],
    },
    {
        "category"    : "JOIN_SINGLE",
        "query"       : "count albums per artist name",
        "expected"    : "SELECT Artist.Name, COUNT(*) FROM Artist JOIN Album ON Artist.ArtistId = Album.ArtistId GROUP BY Artist.ArtistId",
        "must_contain": ["Artist", "Album", "JOIN", "COUNT", "GROUP BY"],
    },

    # ── 7. DOUBLE JOIN — FAIL ─────────────────
    {
        "category"    : "JOIN_DOUBLE",
        "query"       : "show track names with album title and artist name",
        "expected"    : "SELECT Track.Name, Album.Title, Artist.Name FROM Track JOIN Album ON Track.AlbumId = Album.AlbumId JOIN Artist ON Album.ArtistId = Artist.ArtistId",
        "must_contain": ["Track", "Album", "Artist", "JOIN"],
    },
    {
        "category"    : "JOIN_DOUBLE",
        "query"       : "show invoice line details with track names and customer names",
        "expected"    : "SELECT Customer.FirstName, Track.Name FROM Customer JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId",
        "must_contain": ["Customer", "Invoice", "InvoiceLine", "JOIN"],
    },
    {
        "category"    : "JOIN_DOUBLE",
        "query"       : "show tracks with album and artist information",
        "expected"    : "SELECT Track.Name, Album.Title, Artist.Name FROM Track JOIN Album ON Track.AlbumId = Album.AlbumId JOIN Artist ON Album.ArtistId = Artist.ArtistId",
        "must_contain": ["Track", "Album", "Artist", "JOIN"],
    },
    {
        "category"    : "JOIN_DOUBLE",
        "query"       : "show customer invoice and track details",
        "expected"    : "SELECT Customer.FirstName, Invoice.Total, Track.Name FROM Customer JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId JOIN Track ON InvoiceLine.TrackId = Track.TrackId",
        "must_contain": ["Customer", "Invoice", "Track", "JOIN"],
    },

    # ── 8. COMPLEX MULTI-ENTITY — FAIL ────────────────────────────────────
    # These fail because 3+ entities mentioned or complex aggregation
    {
        "category"    : "COMPLEX",
        "query"       : "top 5 artists by total number of tracks across all albums",
        "expected"    : "SELECT Artist.Name, COUNT(Track.TrackId) FROM Artist JOIN Album ON Artist.ArtistId = Album.ArtistId JOIN Track ON Album.AlbumId = Track.AlbumId GROUP BY Artist.ArtistId ORDER BY COUNT(Track.TrackId) DESC LIMIT 5",
        "must_contain": ["Artist", "Album", "Track", "JOIN", "COUNT", "ORDER BY", "LIMIT"],
    },
    {
        "category"    : "COMPLEX",
        "query"       : "show genre name and total revenue from tracks sold",
        "expected"    : "SELECT Genre.Name, SUM(InvoiceLine.UnitPrice) FROM Genre JOIN Track ON Genre.GenreId = Track.GenreId JOIN InvoiceLine ON Track.TrackId = InvoiceLine.TrackId GROUP BY Genre.GenreId",
        "must_contain": ["Genre", "Track", "InvoiceLine", "JOIN", "SUM"],
    },
    {
        "category"    : "COMPLEX",
        "query"       : "show artist name album title and track count per album",
        "expected"    : "SELECT Artist.Name, Album.Title, COUNT(Track.TrackId) FROM Artist JOIN Album ON Artist.ArtistId = Album.ArtistId JOIN Track ON Album.AlbumId = Track.AlbumId GROUP BY Album.AlbumId",
        "must_contain": ["Artist", "Album", "Track", "JOIN", "COUNT"],
    },
    {
        "category"    : "COMPLEX",
        "query"       : "which customers have above average spending",
        "expected"    : "SELECT Customer.FirstName FROM Customer JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId GROUP BY Customer.CustomerId HAVING SUM(Invoice.Total) > (SELECT AVG(Total) FROM Invoice)",
        "must_contain": ["Customer", "Invoice", "HAVING"],
    },
    {
        "category"    : "COMPLEX",
        "query"       : "show playlist name track name and artist name for all playlists",
        "expected"    : "SELECT Playlist.Name, Track.Name, Artist.Name FROM Playlist JOIN PlaylistTrack ON Playlist.PlaylistId = PlaylistTrack.PlaylistId JOIN Track ON PlaylistTrack.TrackId = Track.TrackId JOIN Album ON Track.AlbumId = Album.AlbumId JOIN Artist ON Album.ArtistId = Artist.ArtistId",
        "must_contain": ["Playlist", "Track", "Artist", "JOIN"],
    },
    {
        "category"    : "COMPLEX",
        "query"       : "show customers who bought tracks from the rock genre",
        "expected"    : "SELECT DISTINCT Customer.FirstName FROM Customer JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId JOIN Track ON InvoiceLine.TrackId = Track.TrackId JOIN Genre ON Track.GenreId = Genre.GenreId WHERE Genre.Name = 'Rock'",
        "must_contain": ["Customer", "Invoice", "Track", "Genre", "JOIN"],
    },
]

# ── Result grading ────────────────────────────────────────────────────────────

def normalize_sql(sql: str) -> str:
    if not sql:
        return ""
    sql = re.sub(r'\s+', ' ', sql).strip()
    sql = re.sub(r'\(\s+', '(', sql)
    sql = re.sub(r'\s+\)', ')', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    sql = re.sub(r'\s*\.\s*', '.', sql)
    return sql.upper()


def grade(neural_sql: str, must_contain: list, sql_executed: bool) -> str:
    if not neural_sql or neural_sql == "unavailable":
        return "FAIL"
    sql_upper = normalize_sql(neural_sql)
    if not sql_executed:
        return "FAIL"
    missing = [kw for kw in must_contain if kw.upper() not in sql_upper]
    if not missing:
        return "PASS"
    return f"PARTIAL (missing: {', '.join(missing)})"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  NL2SQL Evaluation v2 — Neural Model + Post-Processing")
    print("="*70)
    print("Loading engine...")
    engine = NL2SQLEngine()
    print("\nRunning test cases...\n")

    results   = []
    cat_stats = {}

    for i, tc in enumerate(TEST_CASES):
        cat      = tc["category"]
        query    = tc["query"]
        expected = tc["expected"]
        must     = tc["must_contain"]

        print(f"[{i+1:02d}/{len(TEST_CASES)}] {cat:<15} | {query}")

        result = engine.query(query)

        neural_sql   = result.get("neural_sql",   "unavailable")
        neural_error = result.get("neural_error", None)
        neural_rows  = result.get("neural_rows",  0)
        corrections  = result.get("neural_corrections", [])

        sql_executed = (neural_error is None or neural_error == "") and neural_rows >= 0

        grade_result = grade(neural_sql, must, sql_executed)

        results.append({
            "category"   : cat,
            "query"      : query,
            "expected"   : expected,
            "neural_sql" : neural_sql,
            "neural_rows": neural_rows,
            "error"      : neural_error,
            "corrections": corrections,
            "grade"      : grade_result,
        })

        if cat not in cat_stats:
            cat_stats[cat] = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "total": 0}
        cat_stats[cat]["total"] += 1
        if grade_result == "PASS":
            cat_stats[cat]["PASS"] += 1
        elif "PARTIAL" in grade_result:
            cat_stats[cat]["PARTIAL"] += 1
        else:
            cat_stats[cat]["FAIL"] += 1

        status = "[PASS]" if grade_result == "PASS" else ("[PART]" if "PARTIAL" in grade_result else "[FAIL]")
        print(f"         {status} {grade_result}")
        print(f"         Generated: {neural_sql[:80]}{'...' if len(neural_sql or '') > 80 else ''}")
        if corrections:
            print(f"         Corrections: {len(corrections)}")
        print()

        time.sleep(0.1)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)

    total_pass    = sum(1 for r in results if r["grade"] == "PASS")
    total_partial = sum(1 for r in results if "PARTIAL" in r["grade"])
    total_fail    = sum(1 for r in results if r["grade"] == "FAIL")
    total         = len(results)

    print(f"\nOverall:")
    print(f"  PASS    : {total_pass:>3} / {total}  ({total_pass/total*100:.1f}%)")
    print(f"  PARTIAL : {total_partial:>3} / {total}  ({total_partial/total*100:.1f}%)")
    print(f"  FAIL    : {total_fail:>3} / {total}  ({total_fail/total*100:.1f}%)")

    print(f"\nBy Category:")
    print(f"  {'Category':<20} {'PASS':>5} {'PARTIAL':>8} {'FAIL':>6} {'Total':>7}")
    print(f"  {'-'*50}")
    for cat, stats in cat_stats.items():
        print(f"  {cat:<20} {stats['PASS']:>5} {stats['PARTIAL']:>8} {stats['FAIL']:>6} {stats['total']:>7}")

    print(f"\nFailed / Partial Queries:")
    print(f"  {'-'*70}")
    failed = [r for r in results if r["grade"] != "PASS"]
    if not failed:
        print("  All queries passed!")
    else:
        for r in failed:
            print(f"\n  [{r['grade']}] {r['category']} | {r['query']}")
            print(f"  Expected : {r['expected']}")
            print(f"  Got      : {r['neural_sql']}")
            if r["error"]:
                print(f"  Error    : {r['error']}")

    # ── Save to file ──────────────────────────────────────────────────────
    out_path = os.path.join(BASE_DIR, "checks", "eval_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("NL2SQL Evaluation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Overall: {total_pass}/{total} PASS, {total_partial}/{total} PARTIAL, {total_fail}/{total} FAIL\n\n")
        f.write("By Category:\n")
        for cat, stats in cat_stats.items():
            f.write(f"  {cat:<20} PASS:{stats['PASS']} PARTIAL:{stats['PARTIAL']} FAIL:{stats['FAIL']}\n")
        f.write("\nDetailed Results:\n")
        f.write("-"*70 + "\n")
        for r in results:
            f.write(f"\n[{r['grade']}] {r['category']} | {r['query']}\n")
            f.write(f"  Expected : {r['expected']}\n")
            f.write(f"  Got      : {r['neural_sql']}\n")
            if r["error"]:
                f.write(f"  Error    : {r['error']}\n")
            if r["corrections"]:
                f.write(f"  Fixes    : {len(r['corrections'])} correction(s)\n")

    print(f"\nFull results saved to: checks/eval_results.txt")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
