"""
config.py
=========
Master configuration for the entire NL2SQL system.

TO SWITCH ENCODER — change ONE line:
    ENCODER_TYPE = "bert"    → BERT encoder  (primary, better accuracy)
    ENCODER_TYPE = "bilstm"  → Bi-LSTM encoder (fallback, if prof says no)

Nothing else needs to change anywhere in the codebase.
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════
# 🔧 MAIN SWITCH — change this one line to swap encoder
# ═══════════════════════════════════════════════════════
ENCODER_TYPE = "bert"       # "bert" or "bilstm"

# ═══════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
WIKISQL_DIR     = DATA_DIR / "wikisql"
SPIDER_DIR      = DATA_DIR / "spider"
PROCESSED_DIR   = DATA_DIR / "processed"
VOCAB_DIR       = DATA_DIR / "vocab"
MODEL_DIR       = BASE_DIR / "models"
CHINOOK_DB_PATH = SPIDER_DIR / "database" / "chinook_1" / "chinook_1.sqlite"

for d in [PROCESSED_DIR, VOCAB_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════
# Chinook Schema  (used for schema linking + BERT input)
# ═══════════════════════════════════════════════════════
CHINOOK_SCHEMA = {
    "Album"        : ["AlbumId", "Title", "ArtistId"],
    "Artist"       : ["ArtistId", "Name"],
    "Customer"     : ["CustomerId", "FirstName", "LastName", "Company",
                      "Address", "City", "State", "Country", "PostalCode",
                      "Phone", "Fax", "Email", "SupportRepId"],
    "Employee"     : ["EmployeeId", "LastName", "FirstName", "Title",
                      "ReportsTo", "BirthDate", "HireDate", "Address",
                      "City", "State", "Country", "PostalCode",
                      "Phone", "Fax", "Email"],
    "Genre"        : ["GenreId", "Name"],
    "Invoice"      : ["InvoiceId", "CustomerId", "InvoiceDate",
                      "BillingAddress", "BillingCity", "BillingState",
                      "BillingCountry", "BillingPostalCode", "Total"],
    "InvoiceLine"  : ["InvoiceLineId", "InvoiceId", "TrackId",
                      "UnitPrice", "Quantity"],
    "MediaType"    : ["MediaTypeId", "Name"],
    "Playlist"     : ["PlaylistId", "Name"],
    "PlaylistTrack": ["PlaylistId", "TrackId"],
    "Track"        : ["TrackId", "Name", "AlbumId", "MediaTypeId",
                      "GenreId", "Composer", "Milliseconds",
                      "Bytes", "UnitPrice"],
}

STORE1_SCHEMA = {
    "artists"        : ["id", "name"],
    "albums"         : ["id", "title", "artist_id"],
    "customers"      : ["id", "first_name", "last_name", "company",
                        "address", "city", "state", "country",
                        "postal_code", "phone", "fax", "email",
                        "support_rep_id"],
    "employees"      : ["id", "last_name", "first_name", "title",
                        "reports_to", "birth_date", "hire_date",
                        "address", "city", "state", "country",
                        "postal_code", "phone", "fax", "email"],
    "genres"         : ["id", "name"],
    "invoices"       : ["id", "customer_id", "invoice_date",
                        "billing_address", "billing_city",
                        "billing_state", "billing_country",
                        "billing_postal_code", "total"],
    "invoice_lines"  : ["id", "invoice_id", "track_id",
                        "unit_price", "quantity"],
    "media_types"    : ["id", "name"],
    "playlists"      : ["id", "name"],
    "playlist_tracks": ["playlist_id", "track_id"],
    "tracks"         : ["id", "name", "album_id", "media_type_id",
                        "genre_id", "composer", "milliseconds",
                        "bytes", "unit_price"],
}

def _schema_to_context(schema: dict) -> str:
    return " ; ".join([
        f"{table} : {' | '.join(cols)}"
        for table, cols in schema.items()
    ])

CHINOOK_SCHEMA_CONTEXT = _schema_to_context(CHINOOK_SCHEMA)
STORE1_SCHEMA_CONTEXT  = _schema_to_context(STORE1_SCHEMA)

# ═══════════════════════════════════════════════════════
# BERT Encoder settings
# ═══════════════════════════════════════════════════════
BERT_MODEL_NAME    = "bert-base-uncased"
BERT_MAX_SEQ_LEN   = 256
BERT_HIDDEN_DIM    = 768
BERT_FREEZE_LAYERS = 6

# ═══════════════════════════════════════════════════════
# Bi-LSTM Encoder settings (fallback)
# ═══════════════════════════════════════════════════════
BILSTM_EMBED_DIM  = 256
BILSTM_HIDDEN_DIM = 512
BILSTM_N_LAYERS   = 2
BILSTM_DROPOUT    = 0.3

# ═══════════════════════════════════════════════════════
# Decoder settings  (same for both encoders)
# ═══════════════════════════════════════════════════════
DECODER_EMBED_DIM  = 256
DECODER_HIDDEN_DIM = 512
DECODER_N_LAYERS   = 1
DECODER_DROPOUT    = 0.3

# ═══════════════════════════════════════════════════════
# Training settings
# ═══════════════════════════════════════════════════════
BATCH_SIZE      = 32
N_EPOCHS        = 30
LR_BERT         = 2e-5
LR_BILSTM       = 1e-3
CLIP_GRAD       = 1.0
TEACHER_FORCING = 0.5
TF_DECAY        = 0.02
TF_MIN          = 0.0
PATIENCE        = 5
MAX_SRC_LEN     = 256
MAX_TGT_LEN     = 100

# ═══════════════════════════════════════════════════════
# Special tokens
# ═══════════════════════════════════════════════════════
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_IDX   = 0
UNK_IDX   = 1
SOS_IDX   = 2
EOS_IDX   = 3

# ═══════════════════════════════════════════════════════
# Hybrid system settings
# ═══════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.40
