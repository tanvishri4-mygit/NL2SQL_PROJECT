"""
Sequence Slot Tagger for NL2SQL
Agent 06: Tags each token in a NL query with its SQL role.
Satisfies Syllabus Module 4: Sequence Tagging, Predicting Sequence of Tags
"""

import os, sys, re
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.preprocessor import tokenize, lemmatize
from retrieval.schema_retriever import SchemaRetriever

# Slot tag constants
TAG_TABLE = "TABLE"
TAG_COLUMN = "COLUMN"
TAG_AGG = "AGG_FUNC"
TAG_VALUE = "VALUE"
TAG_OP = "OP"
TAG_GROUPBY = "GROUPBY"
TAG_ORDERBY = "ORDERBY"
TAG_LIMIT = "LIMIT"
TAG_ACTION = "ACTION"
TAG_SORT_DIR = "SORT_DIR"
TAG_OTHER = "O"


class SlotTagger:
    """Hybrid rule-based + retrieval-enhanced sequence tagger."""
    
    def __init__(self, schema_retriever=None):
        self.retriever = schema_retriever or SchemaRetriever()
        
        self.action_words = {"show","list","get","find","display","give","tell","fetch","retrieve","select","search","what","how"}
        
        self.agg_map = {
            "count":"COUNT","total":"SUM","sum":"SUM","average":"AVG","avg":"AVG",
            "mean":"AVG","maximum":"MAX","max":"MAX","highest":"MAX",
            "minimum":"MIN","min":"MIN","lowest":"MIN","most":"COUNT","least":"COUNT",
        }
        
        self.op_map = {
            "greater":">","more":">","above":">","over":">","exceeding":">",
            "less":"<","below":"<","under":"<","fewer":"<",
            "equal":"=","equals":"=","is":"=","exactly":"=",
            "between":"BETWEEN","not":"!=","except":"!=",
            "like":"LIKE","containing":"LIKE","contains":"LIKE",
            "after":">","before":"<","since":">=","than":"_SKIP",
        }
        
        self.groupby_triggers = {"by","per","each","every","wise","grouped"}
        self.orderby_triggers = {"sort","sorted","order","ordered","rank","ranked","arrange"}
        self.limit_triggers = {"top","first","last","bottom","limit","latest","newest","oldest"}
        
        self.sort_dir_map = {
            "ascending":"ASC","asc":"ASC","increasing":"ASC",
            "descending":"DESC","desc":"DESC","decreasing":"DESC",
            "highest":"DESC","lowest":"ASC","most":"DESC","least":"ASC",
            "top":"DESC","bottom":"ASC","latest":"DESC","newest":"DESC","oldest":"ASC",
        }
        
        self.status_values = {
            "delivered","cancelled","placed","packed","shipped","returned",
            "success","failed","pending","refunded",
            "requested","approved","pickedup","rejected",
            "intransit","lost","returnedtoseller",
            "damaged","wrongitem","notneeded","delayed",
            "home","office","bronze","silver","gold","platinum",
            "upi","card","cod","netbanking","wallet","percent","fixed",
            "male","female","other",
        }
        
        self.city_names = {
            "mumbai","pune","delhi","bangalore","bengaluru","hyderabad",
            "chennai","kolkata","ahmedabad","jaipur","lucknow","chandigarh",
            "bhopal","indore","nagpur","coimbatore","kochi","visakhapatnam",
            "surat","noida","gurgaon","gurugram",
        }
        
        # Month names for date parsing
        self.months = {
            "january":"01","february":"02","march":"03","april":"04","may":"05","june":"06",
            "july":"07","august":"08","september":"09","october":"10","november":"11","december":"12",
            "jan":"01","feb":"02","mar":"03","apr":"04","jun":"06","jul":"07",
            "aug":"08","sep":"09","oct":"10","nov":"11","dec":"12",
        }
        
        # Brand names (will be detected as values)
        self.brand_names = {
            "samsung","apple","oneplus","xiaomi","sony","lg","boat","jbl","dell","hp","lenovo",
            "nike","adidas","puma","levi","zara","tata","amul","nestle","britannia",
            "prestige","philips","havells","delhivery","bluedart","dtdc","ekart","fedex",
        }
    
    def tag(self, query: str, retrieved_context: dict = None) -> list:
        """
        Tag tokens with SQL semantic roles.
        Returns list of (token, tag, metadata) tuples.
        """
        tokens = tokenize(query)
        
        if retrieved_context is None:
            retrieved_context = self.retriever.get_retrieval_context(query, top_k=5)
        
        # Build lookup from retrieved context
        retrieved_tables = set()
        col_lookup = {}  # synonym/name -> (table, col_name)
        table_synonyms = {}
        
        for tinfo in retrieved_context.get("retrieved_tables", []):
            tname = tinfo["table"]
            retrieved_tables.add(tname)
            schema = retrieved_context.get("table_schemas", {}).get(tname, {})
            for syn in schema.get("synonyms", []):
                table_synonyms[syn.lower()] = tname
            table_synonyms[tname] = tname
            for col in schema.get("columns", []):
                cname = col["name"]
                col_lookup[cname.lower()] = (tname, cname)
                for syn in col.get("synonyms", []):
                    col_lookup[syn.lower()] = (tname, cname)
        
        tagged = []
        agg_detected = False
        group_by_context = False
        order_by_context = False
        
        for i, token in enumerate(tokens):
            lemma = lemmatize(token)
            tag = TAG_OTHER
            meta = {}
            
            # Action verbs
            if token in self.action_words or lemma in self.action_words:
                tag = TAG_ACTION
            
            # Aggregation
            elif token in self.agg_map or lemma in self.agg_map:
                tag = TAG_AGG
                meta["agg"] = self.agg_map.get(token, self.agg_map.get(lemma, ""))
                agg_detected = True
                if token in self.sort_dir_map:
                    meta["sort_hint"] = self.sort_dir_map[token]
            
            # Numbers
            elif re.match(r'^\d+\.?\d*$', token):
                tag = TAG_VALUE
                meta["value_type"] = "number"
                meta["value"] = float(token) if '.' in token else int(token)
            
            # Year
            elif re.match(r'^\d{4}$', token) and 1900 <= int(token) <= 2100:
                tag = TAG_VALUE
                meta["value_type"] = "year"
                meta["value"] = int(token)
            
            # Month name
            elif token in self.months:
                tag = TAG_VALUE
                meta["value_type"] = "month"
                meta["value"] = self.months[token]
            
            # Status/enum values
            elif token in self.status_values:
                tag = TAG_VALUE
                meta["value_type"] = "enum"
                # Map to proper casing
                status_map = {
                    "intransit": "InTransit", "wrongitem": "WrongItem",
                    "notneeded": "NotNeeded", "returnedtoseller": "ReturnedToSeller",
                    "pickedup": "PickedUp",
                }
                meta["value"] = status_map.get(token, token.capitalize())
            
            # City names
            elif token in self.city_names:
                tag = TAG_VALUE
                meta["value_type"] = "city"
                meta["value"] = token.title()
            
            # Brand names
            elif token in self.brand_names:
                tag = TAG_VALUE
                meta["value_type"] = "brand"
                meta["value"] = token.title()
            
            # Operators
            elif token in self.op_map:
                op = self.op_map[token]
                if op != "_SKIP":
                    tag = TAG_OP
                    meta["operator"] = op
            
            # Limit triggers
            elif token in self.limit_triggers:
                tag = TAG_LIMIT
                if token in self.sort_dir_map:
                    meta["sort_hint"] = self.sort_dir_map[token]
            
            # Order triggers
            elif token in self.orderby_triggers:
                tag = TAG_ORDERBY
                order_by_context = True
            
            # Sort direction
            elif token in self.sort_dir_map and not agg_detected:
                tag = TAG_SORT_DIR
                meta["direction"] = self.sort_dir_map[token]
            
            # Group-by triggers (contextual - "by" after aggregate = GROUPBY)
            elif token in self.groupby_triggers:
                if agg_detected or any(t[1] == TAG_AGG for t in tagged):
                    tag = TAG_GROUPBY
                    group_by_context = True
                elif any(t[1] in (TAG_ORDERBY, TAG_LIMIT) for t in tagged[-3:]):
                    tag = TAG_ORDERBY
                    order_by_context = True
                else:
                    tag = TAG_GROUPBY
                    group_by_context = True
            
            # Table name match
            elif token in table_synonyms or lemma in table_synonyms:
                tag = TAG_TABLE
                meta["table"] = table_synonyms.get(token, table_synonyms.get(lemma, token))
            
            # Column name match
            elif token in col_lookup or lemma in col_lookup:
                tag = TAG_COLUMN
                tbl, col = col_lookup.get(token, col_lookup.get(lemma, (None, None)))
                meta["table"] = tbl
                meta["column"] = col
            
            # Partial column match (e.g., "name" → full_name, "date" → order_date)
            else:
                for syn, (tbl, col) in col_lookup.items():
                    if token == syn or (len(token) > 3 and token in syn):
                        tag = TAG_COLUMN
                        meta["table"] = tbl
                        meta["column"] = col
                        break
            
            tagged.append((token, tag, meta))
        
        return tagged
    
    def get_slots(self, tagged_sequence: list) -> dict:
        """
        Extract structured slot information from tagged sequence.
        Returns dict with extracted SQL components.
        """
        slots = {
            "action": None,
            "tables": [],
            "columns": [],
            "agg_functions": [],
            "values": [],
            "operators": [],
            "group_by": [],
            "order_by": [],
            "limit": None,
            "sort_direction": "DESC",  # default
        }
        
        group_by_mode = False
        order_by_mode = False
        
        for i, (token, tag, meta) in enumerate(tagged_sequence):
            if tag == TAG_ACTION:
                slots["action"] = token
            
            elif tag == TAG_TABLE:
                tname = meta.get("table", token)
                if tname not in slots["tables"]:
                    slots["tables"].append(tname)
            
            elif tag == TAG_COLUMN:
                col_info = {"table": meta.get("table"), "column": meta.get("column", token)}
                if group_by_mode:
                    slots["group_by"].append(col_info)
                elif order_by_mode:
                    slots["order_by"].append(col_info)
                else:
                    slots["columns"].append(col_info)
            
            elif tag == TAG_AGG:
                slots["agg_functions"].append(meta.get("agg", "COUNT"))
                if "sort_hint" in meta:
                    slots["sort_direction"] = meta["sort_hint"]
            
            elif tag == TAG_VALUE:
                slots["values"].append(meta)
            
            elif tag == TAG_OP:
                slots["operators"].append(meta.get("operator", "="))
            
            elif tag == TAG_GROUPBY:
                group_by_mode = True
                order_by_mode = False
            
            elif tag == TAG_ORDERBY:
                order_by_mode = True
                group_by_mode = False
            
            elif tag == TAG_LIMIT:
                # Look ahead for a number
                if "sort_hint" in meta:
                    slots["sort_direction"] = meta["sort_hint"]
                for j in range(i+1, min(i+3, len(tagged_sequence))):
                    if tagged_sequence[j][1] == TAG_VALUE and tagged_sequence[j][2].get("value_type") == "number":
                        slots["limit"] = int(tagged_sequence[j][2]["value"])
                        break
                if slots["limit"] is None:
                    slots["limit"] = 10  # default top-k
            
            elif tag == TAG_SORT_DIR:
                slots["sort_direction"] = meta.get("direction", "DESC")
        
        return slots


if __name__ == "__main__":
    tagger = SlotTagger()
    
    test_queries = [
        "show all customers from Mumbai",
        "count total orders in 2025",
        "top 5 products by revenue",
        "average order total by city",
        "show orders where total greater than 5000",
        "list cancelled orders with refund above 3000",
        "total revenue by category",
        "show customer name and order total for delivered orders",
    ]
    
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        tagged = tagger.tag(q)
        print("Tags:")
        for token, tag, meta in tagged:
            meta_str = f" {meta}" if meta else ""
            print(f"  {token:<20} → {tag:<12}{meta_str}")
        
        slots = tagger.get_slots(tagged)
        print(f"Slots: {slots}")
