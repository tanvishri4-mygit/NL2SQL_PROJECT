"""
Intent Dataset & TF-IDF Feature Extraction
Agent 06 (Service Developer): Creates labeled NL query dataset for training
and implements TF-IDF vectorization for intent classification.

Satisfies Syllabus Module 3: Text classification
- Term Frequency, CountVectorizer, Inverse Document Frequency, Text conversion
"""

import csv
import os
import math
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

# ============================================================
# Intent Labels
# ============================================================
INTENT_LABELS = [
    "SELECT_SIMPLE",   # Basic SELECT with optional WHERE
    "AGGREGATE",       # COUNT, SUM, AVG, MIN, MAX
    "GROUP_BY",        # GROUP BY queries
    "ORDER_LIMIT",     # ORDER BY + LIMIT / Top-K
    "JOIN_QUERY",      # Queries requiring JOINs
    "FILTER_WHERE",    # Complex WHERE conditions
]

# ============================================================
# 200 Labeled NL Queries
# ============================================================
INTENT_DATASET = [
    # === SELECT_SIMPLE (35) ===
    ("show all customers", "SELECT_SIMPLE", "customers", "Easy"),
    ("list all products", "SELECT_SIMPLE", "products", "Easy"),
    ("show all orders", "SELECT_SIMPLE", "orders", "Easy"),
    ("show all delivered orders", "SELECT_SIMPLE", "orders", "Easy"),
    ("list cancelled orders", "SELECT_SIMPLE", "orders", "Easy"),
    ("show orders placed in 2025", "SELECT_SIMPLE", "orders", "Easy"),
    ("show products in electronics category", "SELECT_SIMPLE", "products,categories", "Easy"),
    ("list products by brand samsung", "SELECT_SIMPLE", "products", "Easy"),
    ("show customers from mumbai", "SELECT_SIMPLE", "customers,addresses", "Easy"),
    ("list customers who joined after january 2025", "SELECT_SIMPLE", "customers", "Easy"),
    ("show all payments", "SELECT_SIMPLE", "payments", "Easy"),
    ("show failed payments", "SELECT_SIMPLE", "payments", "Easy"),
    ("list shipments in transit", "SELECT_SIMPLE", "shipments", "Easy"),
    ("show all returns", "SELECT_SIMPLE", "returns", "Easy"),
    ("list active promotions", "SELECT_SIMPLE", "promotions", "Easy"),
    ("show promotions ending this month", "SELECT_SIMPLE", "promotions", "Easy"),
    ("display all categories", "SELECT_SIMPLE", "categories", "Easy"),
    ("get all gold tier customers", "SELECT_SIMPLE", "customers", "Easy"),
    ("show female customers", "SELECT_SIMPLE", "customers", "Easy"),
    ("list products with rating above 4", "SELECT_SIMPLE", "products", "Easy"),
    ("find orders with cod payment", "SELECT_SIMPLE", "payments", "Easy"),
    ("show refunded payments", "SELECT_SIMPLE", "payments", "Easy"),
    ("list products in fashion", "SELECT_SIMPLE", "products,categories", "Easy"),
    ("display all brands", "SELECT_SIMPLE", "products", "Easy"),
    ("show orders from last month", "SELECT_SIMPLE", "orders", "Easy"),
    ("get platinum customers", "SELECT_SIMPLE", "customers", "Easy"),
    ("list all couriers", "SELECT_SIMPLE", "shipments", "Easy"),
    ("show products with zero stock", "SELECT_SIMPLE", "products", "Easy"),
    ("find customers with gmail", "SELECT_SIMPLE", "customers", "Easy"),
    ("show pending payments", "SELECT_SIMPLE", "payments", "Easy"),
    ("list wallet payments", "SELECT_SIMPLE", "payments", "Easy"),
    ("show grocery products", "SELECT_SIMPLE", "products,categories", "Easy"),
    ("display all addresses", "SELECT_SIMPLE", "addresses", "Easy"),
    ("get orders with status shipped", "SELECT_SIMPLE", "orders", "Easy"),
    ("show upi payments", "SELECT_SIMPLE", "payments", "Easy"),
    
    # === FILTER_WHERE (35) ===
    ("show orders where total is greater than 5000", "FILTER_WHERE", "orders", "Med"),
    ("show orders where status is delivered", "FILTER_WHERE", "orders", "Easy"),
    ("list products where price is less than 1000", "FILTER_WHERE", "products", "Easy"),
    ("show customers where loyalty tier is gold", "FILTER_WHERE", "customers", "Easy"),
    ("show payments where method is upi", "FILTER_WHERE", "payments", "Easy"),
    ("show shipments where courier is delhivery", "FILTER_WHERE", "shipments", "Easy"),
    ("show returns where reason is damaged", "FILTER_WHERE", "returns", "Easy"),
    ("show orders delivered after 10 february 2026", "FILTER_WHERE", "orders", "Med"),
    ("find products priced between 500 and 2000", "FILTER_WHERE", "products", "Med"),
    ("show orders with total above 10000", "FILTER_WHERE", "orders", "Med"),
    ("list customers joined in 2024", "FILTER_WHERE", "customers", "Easy"),
    ("show products with stock less than 10", "FILTER_WHERE", "products", "Easy"),
    ("find orders placed before january 2025", "FILTER_WHERE", "orders", "Med"),
    ("show shipments with fee more than 50", "FILTER_WHERE", "shipments", "Easy"),
    ("list returns with refund above 3000", "FILTER_WHERE", "returns", "Med"),
    ("show orders not delivered", "FILTER_WHERE", "orders", "Easy"),
    ("find products with rating below 2", "FILTER_WHERE", "products", "Easy"),
    ("show cancelled orders in 2025", "FILTER_WHERE", "orders", "Med"),
    ("list payments with amount more than 8000", "FILTER_WHERE", "payments", "Med"),
    ("show promotions with discount value above 20", "FILTER_WHERE", "promotions", "Easy"),
    ("find orders placed in last 30 days", "FILTER_WHERE", "orders", "Med"),
    ("show customers registered after march 2025", "FILTER_WHERE", "customers", "Med"),
    ("list products with price above 5000 and rating above 4", "FILTER_WHERE", "products", "Hard"),
    ("show shipments delivered in january 2026", "FILTER_WHERE", "shipments", "Med"),
    ("find netbanking payments that failed", "FILTER_WHERE", "payments", "Med"),
    ("show returns requested in last 30 days", "FILTER_WHERE", "returns", "Med"),
    ("list orders where total between 1000 and 5000", "FILTER_WHERE", "orders", "Med"),
    ("show silver tier customers from delhi", "FILTER_WHERE", "customers,addresses", "Med"),
    ("find products in electronics with price under 2000", "FILTER_WHERE", "products,categories", "Med"),
    ("show orders with status placed or packed", "FILTER_WHERE", "orders", "Med"),
    ("list shipments that are lost", "FILTER_WHERE", "shipments", "Easy"),
    ("show promotions with minimum order above 1000", "FILTER_WHERE", "promotions", "Easy"),
    ("find customers joined between 2024 and 2025", "FILTER_WHERE", "customers", "Med"),
    ("show card payments above 5000", "FILTER_WHERE", "payments", "Med"),
    ("list rejected returns", "FILTER_WHERE", "returns", "Easy"),
    
    # === AGGREGATE (35) ===
    ("count total orders", "AGGREGATE", "orders", "Easy"),
    ("count delivered orders", "AGGREGATE", "orders", "Easy"),
    ("count cancelled orders", "AGGREGATE", "orders", "Easy"),
    ("average order total", "AGGREGATE", "orders", "Easy"),
    ("total revenue", "AGGREGATE", "orders", "Easy"),
    ("total payments received", "AGGREGATE", "payments", "Easy"),
    ("average product price", "AGGREGATE", "products", "Easy"),
    ("maximum order total in 2025", "AGGREGATE", "orders", "Med"),
    ("minimum product price in fashion", "AGGREGATE", "products,categories", "Med"),
    ("count returns in 2025", "AGGREGATE", "returns", "Easy"),
    ("total refund amount", "AGGREGATE", "returns", "Easy"),
    ("count customers", "AGGREGATE", "customers", "Easy"),
    ("count products", "AGGREGATE", "products", "Easy"),
    ("average rating of all products", "AGGREGATE", "products", "Easy"),
    ("total shipping fees collected", "AGGREGATE", "shipments", "Easy"),
    ("count failed payments", "AGGREGATE", "payments", "Easy"),
    ("sum of all refunds", "AGGREGATE", "returns", "Easy"),
    ("average refund amount", "AGGREGATE", "returns", "Easy"),
    ("count gold customers", "AGGREGATE", "customers", "Easy"),
    ("total revenue from upi payments", "AGGREGATE", "payments", "Med"),
    ("average shipping fee", "AGGREGATE", "shipments", "Easy"),
    ("max product price", "AGGREGATE", "products", "Easy"),
    ("min order total", "AGGREGATE", "orders", "Easy"),
    ("count shipments by delhivery", "AGGREGATE", "shipments", "Easy"),
    ("total discount amount across all orders", "AGGREGATE", "order_items", "Med"),
    ("count orders in january 2026", "AGGREGATE", "orders", "Med"),
    ("average order value in 2025", "AGGREGATE", "orders", "Med"),
    ("how many products have rating above 4", "AGGREGATE", "products", "Easy"),
    ("what is the total number of returns", "AGGREGATE", "returns", "Easy"),
    ("how many customers joined this year", "AGGREGATE", "customers", "Med"),
    ("count pending shipments", "AGGREGATE", "shipments", "Easy"),
    ("total amount paid via card", "AGGREGATE", "payments", "Med"),
    ("average product stock", "AGGREGATE", "products", "Easy"),
    ("count products in electronics", "AGGREGATE", "products,categories", "Med"),
    ("total order value for delivered orders", "AGGREGATE", "orders", "Med"),
    
    # === GROUP_BY (35) ===
    ("total revenue by month", "GROUP_BY", "orders", "Med"),
    ("total revenue by city", "GROUP_BY", "orders,addresses", "Hard"),
    ("orders count by status", "GROUP_BY", "orders", "Easy"),
    ("orders count by payment method", "GROUP_BY", "payments", "Med"),
    ("revenue by category", "GROUP_BY", "orders,order_items,products,categories", "Hard"),
    ("returns count by reason", "GROUP_BY", "returns", "Easy"),
    ("refund amount by reason", "GROUP_BY", "returns", "Easy"),
    ("orders count by loyalty tier", "GROUP_BY", "orders,customers", "Med"),
    ("average order value by city", "GROUP_BY", "orders,addresses", "Hard"),
    ("customers by gender", "GROUP_BY", "customers", "Easy"),
    ("products count by category", "GROUP_BY", "products,categories", "Med"),
    ("orders by month in 2025", "GROUP_BY", "orders", "Med"),
    ("payments by method", "GROUP_BY", "payments", "Easy"),
    ("shipments by courier", "GROUP_BY", "shipments", "Easy"),
    ("returns by status", "GROUP_BY", "returns", "Easy"),
    ("revenue by brand", "GROUP_BY", "order_items,products", "Hard"),
    ("average price by category", "GROUP_BY", "products,categories", "Med"),
    ("customers count by tier", "GROUP_BY", "customers", "Easy"),
    ("orders count per customer", "GROUP_BY", "orders", "Med"),
    ("total spending by customer", "GROUP_BY", "orders", "Med"),
    ("average rating by category", "GROUP_BY", "products,categories", "Med"),
    ("count orders by year", "GROUP_BY", "orders", "Easy"),
    ("shipments count by status", "GROUP_BY", "shipments", "Easy"),
    ("returns by month", "GROUP_BY", "returns", "Med"),
    ("payment amount by method", "GROUP_BY", "payments", "Easy"),
    ("products by brand", "GROUP_BY", "products", "Easy"),
    ("orders per city", "GROUP_BY", "orders,addresses", "Hard"),
    ("revenue per month in 2025", "GROUP_BY", "orders", "Med"),
    ("average discount by category", "GROUP_BY", "order_items,products,categories", "Hard"),
    ("customers by state", "GROUP_BY", "addresses", "Easy"),
    ("orders grouped by status and month", "GROUP_BY", "orders", "Med"),
    ("revenue by payment method", "GROUP_BY", "orders,payments", "Med"),
    ("shipping fee by courier", "GROUP_BY", "shipments", "Easy"),
    ("order count by day of week", "GROUP_BY", "orders", "Med"),
    ("products count by brand", "GROUP_BY", "products", "Easy"),
    
    # === ORDER_LIMIT (30) ===
    ("top 5 customers by total spending", "ORDER_LIMIT", "orders", "Med"),
    ("top 10 products by sales quantity", "ORDER_LIMIT", "order_items", "Med"),
    ("top 5 cities by revenue", "ORDER_LIMIT", "orders,addresses", "Hard"),
    ("latest 10 orders", "ORDER_LIMIT", "orders", "Easy"),
    ("highest value orders in 2025", "ORDER_LIMIT", "orders", "Med"),
    ("top 5 brands by revenue", "ORDER_LIMIT", "order_items,products", "Hard"),
    ("most returned products", "ORDER_LIMIT", "returns,order_items,products", "Hard"),
    ("most used payment method", "ORDER_LIMIT", "payments", "Med"),
    ("promotions with highest discount", "ORDER_LIMIT", "promotions", "Easy"),
    ("customers with most orders", "ORDER_LIMIT", "orders", "Med"),
    ("cheapest products", "ORDER_LIMIT", "products", "Easy"),
    ("most expensive products", "ORDER_LIMIT", "products", "Easy"),
    ("top rated products", "ORDER_LIMIT", "products", "Easy"),
    ("lowest rated products", "ORDER_LIMIT", "products", "Easy"),
    ("newest customers", "ORDER_LIMIT", "customers", "Easy"),
    ("oldest customers", "ORDER_LIMIT", "customers", "Easy"),
    ("top 3 categories by product count", "ORDER_LIMIT", "products,categories", "Med"),
    ("last 5 returns", "ORDER_LIMIT", "returns", "Easy"),
    ("top 10 highest order totals", "ORDER_LIMIT", "orders", "Easy"),
    ("bottom 5 products by rating", "ORDER_LIMIT", "products", "Easy"),
    ("first 20 orders placed", "ORDER_LIMIT", "orders", "Easy"),
    ("top spending customers", "ORDER_LIMIT", "orders", "Med"),
    ("most popular products by quantity sold", "ORDER_LIMIT", "order_items", "Med"),
    ("largest refunds", "ORDER_LIMIT", "returns", "Easy"),
    ("highest shipping fees", "ORDER_LIMIT", "shipments", "Easy"),
    ("top couriers by delivery count", "ORDER_LIMIT", "shipments", "Med"),
    ("brands with most products", "ORDER_LIMIT", "products", "Easy"),
    ("categories with highest average price", "ORDER_LIMIT", "products,categories", "Med"),
    ("top 5 products by revenue", "ORDER_LIMIT", "order_items,products", "Hard"),
    ("most active customers", "ORDER_LIMIT", "orders", "Med"),
    
    # === JOIN_QUERY (30) ===
    ("show customer name and total for each order", "JOIN_QUERY", "orders,customers", "Med"),
    ("show order with payment status and payment method", "JOIN_QUERY", "orders,payments", "Med"),
    ("show product name and quantity for order 100", "JOIN_QUERY", "order_items,products", "Med"),
    ("show city wise delivered orders count", "JOIN_QUERY", "orders,addresses", "Hard"),
    ("show category name and revenue", "JOIN_QUERY", "order_items,products,categories", "Hard"),
    ("show courier wise delivered orders", "JOIN_QUERY", "shipments,orders", "Med"),
    ("show customers and their loyalty tier for top spenders", "JOIN_QUERY", "orders,customers", "Hard"),
    ("show returns with customer name and refund amount", "JOIN_QUERY", "returns,orders,customers", "Hard"),
    ("show orders with shipment status", "JOIN_QUERY", "orders,shipments", "Med"),
    ("display customer name with order date and status", "JOIN_QUERY", "orders,customers", "Med"),
    ("show product name with category for all items", "JOIN_QUERY", "products,categories", "Med"),
    ("list orders with customer email", "JOIN_QUERY", "orders,customers", "Med"),
    ("show payment details for delivered orders", "JOIN_QUERY", "orders,payments", "Med"),
    ("display shipment details with customer name", "JOIN_QUERY", "shipments,orders,customers", "Hard"),
    ("show product brand with order count", "JOIN_QUERY", "order_items,products", "Med"),
    ("list returns with product name", "JOIN_QUERY", "returns,order_items,products", "Hard"),
    ("show customer address with order total", "JOIN_QUERY", "orders,customers,addresses", "Hard"),
    ("display order items with product price", "JOIN_QUERY", "order_items,products", "Med"),
    ("show all orders with city and state", "JOIN_QUERY", "orders,addresses", "Med"),
    ("list products with their category name", "JOIN_QUERY", "products,categories", "Easy"),
    ("show customers and number of orders each placed", "JOIN_QUERY", "orders,customers", "Med"),
    ("display orders with coupon applied", "JOIN_QUERY", "orders,promotions", "Med"),
    ("show average order value per customer with name", "JOIN_QUERY", "orders,customers", "Hard"),
    ("list products bought from mumbai", "JOIN_QUERY", "order_items,orders,addresses", "Hard"),
    ("show customer name email and their return count", "JOIN_QUERY", "returns,orders,customers", "Hard"),
    ("display category revenue with product count", "JOIN_QUERY", "order_items,products,categories", "Hard"),
    ("show orders delivered by bluedart", "JOIN_QUERY", "orders,shipments", "Med"),
    ("list customers with their city", "JOIN_QUERY", "customers,addresses", "Easy"),
    ("show payment method used per customer", "JOIN_QUERY", "payments,orders,customers", "Hard"),
    ("display shipped orders with courier and customer name", "JOIN_QUERY", "orders,shipments,customers", "Hard"),
]


def save_dataset(filepath=None):
    """Save the intent dataset as CSV."""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "intent_dataset.csv")
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["query_text", "intent_label", "gold_tables", "difficulty"])
        for row in INTENT_DATASET:
            writer.writerow(row)
    
    print(f"Saved {len(INTENT_DATASET)} queries to {filepath}")
    
    # Print distribution
    dist = Counter(row[1] for row in INTENT_DATASET)
    print("\nIntent Distribution:")
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count}")
    
    return filepath


# ============================================================
# TF-IDF Vectorizer (from scratch, no sklearn dependency needed)
# ============================================================
class TFIDFVectorizer:
    """
    Custom TF-IDF Vectorizer implementation.
    Satisfies syllabus requirement for understanding TF-IDF from ground up.
    
    TF(t,d) = count(t in d) / |d|
    IDF(t) = log(N / (1 + df(t)))
    TF-IDF(t,d) = TF(t,d) * IDF(t)
    """
    
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocabulary_ = {}
        self.idf_ = {}
        self.feature_names_ = []
    
    def fit(self, documents: list):
        """Build vocabulary and compute IDF from documents (list of token lists)."""
        N = len(documents)
        df = Counter()  # Document frequency
        tf_all = Counter()  # Total term frequency (for feature selection)
        
        for doc in documents:
            unique_tokens = set(doc)
            for token in unique_tokens:
                df[token] += 1
            for token in doc:
                tf_all[token] += 1
        
        # Select top features if max_features is set
        if self.max_features:
            top_tokens = [t for t, _ in tf_all.most_common(self.max_features)]
        else:
            top_tokens = sorted(tf_all.keys())
        
        self.vocabulary_ = {token: idx for idx, token in enumerate(top_tokens)}
        self.feature_names_ = top_tokens
        
        # Compute IDF
        for token in top_tokens:
            self.idf_[token] = math.log(N / (1 + df.get(token, 0)))
        
        return self
    
    def transform(self, documents: list) -> list:
        """Transform documents to TF-IDF vectors."""
        vectors = []
        for doc in documents:
            vec = [0.0] * len(self.vocabulary_)
            doc_len = len(doc) if len(doc) > 0 else 1
            token_counts = Counter(doc)
            
            for token, count in token_counts.items():
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    tf = count / doc_len
                    idf = self.idf_.get(token, 0)
                    vec[idx] = tf * idf
            
            # L2 normalize
            norm = math.sqrt(sum(v*v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            
            vectors.append(vec)
        
        return vectors
    
    def fit_transform(self, documents: list) -> list:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)


# ============================================================
# CountVectorizer (for syllabus completeness)
# ============================================================
class CountVectorizer:
    """
    Simple Bag-of-Words Count Vectorizer.
    Satisfies syllabus: CountVectorizer mention.
    """
    
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocabulary_ = {}
        self.feature_names_ = []
    
    def fit(self, documents: list):
        tf_all = Counter()
        for doc in documents:
            for token in doc:
                tf_all[token] += 1
        
        if self.max_features:
            top_tokens = [t for t, _ in tf_all.most_common(self.max_features)]
        else:
            top_tokens = sorted(tf_all.keys())
        
        self.vocabulary_ = {token: idx for idx, token in enumerate(top_tokens)}
        self.feature_names_ = top_tokens
        return self
    
    def transform(self, documents: list) -> list:
        vectors = []
        for doc in documents:
            vec = [0] * len(self.vocabulary_)
            for token in doc:
                if token in self.vocabulary_:
                    vec[self.vocabulary_[token]] += 1
            vectors.append(vec)
        return vectors
    
    def fit_transform(self, documents: list) -> list:
        self.fit(documents)
        return self.transform(documents)


if __name__ == "__main__":
    save_dataset()
