"""
Database Creator - E-commerce NL2SQL System
Agent 07 (Database Developer): Creates and seeds the SQLite e-commerce database
with 10 tables and realistic data for NL-to-SQL query testing.
"""

import sqlite3
import random
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "ecommerce.db")

SCHEMA_SQL = """
-- 1) customers
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    gender TEXT CHECK(gender IN ('Male','Female','Other')),
    date_joined DATE NOT NULL,
    loyalty_tier TEXT CHECK(loyalty_tier IN ('Bronze','Silver','Gold','Platinum'))
);

-- 2) addresses
CREATE TABLE IF NOT EXISTS addresses (
    address_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    address_type TEXT CHECK(address_type IN ('Home','Office')),
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    pincode TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- 3) categories
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

-- 4) products
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT NOT NULL,
    category_id INTEGER NOT NULL,
    brand TEXT NOT NULL,
    unit_price REAL NOT NULL,
    stock_qty INTEGER NOT NULL DEFAULT 0,
    rating_avg REAL DEFAULT 0.0 CHECK(rating_avg >= 0 AND rating_avg <= 5),
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

-- 5) orders
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    order_date DATE NOT NULL,
    order_status TEXT CHECK(order_status IN ('Placed','Packed','Shipped','Delivered','Cancelled','Returned')),
    shipping_address_id INTEGER,
    order_total REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (shipping_address_id) REFERENCES addresses(address_id)
);

-- 6) order_items
CREATE TABLE IF NOT EXISTS order_items (
    order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price REAL NOT NULL,
    discount_amount REAL DEFAULT 0.0,
    line_total REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- 7) payments
CREATE TABLE IF NOT EXISTS payments (
    payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    payment_date DATE NOT NULL,
    payment_method TEXT CHECK(payment_method IN ('UPI','Card','COD','NetBanking','Wallet')),
    payment_status TEXT CHECK(payment_status IN ('Success','Failed','Pending','Refunded')),
    amount_paid REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- 8) shipments
CREATE TABLE IF NOT EXISTS shipments (
    shipment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    courier_name TEXT NOT NULL,
    shipped_date DATE,
    delivered_date DATE,
    shipping_status TEXT CHECK(shipping_status IN ('Shipped','InTransit','Delivered','Lost','ReturnedToSeller')),
    shipping_fee REAL DEFAULT 0.0,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- 9) returns
CREATE TABLE IF NOT EXISTS returns (
    return_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    return_date DATE NOT NULL,
    return_reason TEXT CHECK(return_reason IN ('Damaged','WrongItem','NotNeeded','Delayed','Other')),
    refund_amount REAL NOT NULL DEFAULT 0.0,
    return_status TEXT CHECK(return_status IN ('Requested','Approved','PickedUp','Refunded','Rejected')),
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- 10) promotions
CREATE TABLE IF NOT EXISTS promotions (
    promo_id INTEGER PRIMARY KEY AUTOINCREMENT,
    promo_code TEXT UNIQUE NOT NULL,
    discount_type TEXT CHECK(discount_type IN ('PERCENT','FIXED')),
    discount_value REAL NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    min_order_value REAL DEFAULT 0.0
);
"""

# Seed data constants
FIRST_NAMES_M = ["Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Reyansh","Ayaan","Krishna","Ishaan",
                  "Shaurya","Atharv","Advik","Pranav","Advaith","Aarush","Kabir","Ritvik","Darsh","Neel",
                  "Rohan","Amit","Raj","Vikram","Sahil","Kunal","Manish","Gaurav","Nikhil","Suresh"]
FIRST_NAMES_F = ["Ananya","Diya","Myra","Sara","Aanya","Aadhya","Isha","Riya","Priya","Kavya",
                  "Meera","Neha","Pooja","Shreya","Tanya","Anjali","Divya","Nisha","Sneha","Swati",
                  "Pallavi","Simran","Komal","Aisha","Zara","Tanvi","Aditi","Bhavna","Charvi","Sanya"]
LAST_NAMES = ["Sharma","Verma","Patel","Gupta","Singh","Kumar","Reddy","Joshi","Mehta","Shah",
              "Iyer","Nair","Das","Bose","Roy","Kapoor","Malhotra","Chopra","Banerjee","Mishra",
              "Sinha","Tiwari","Pandey","Rao","Pillai","Desai","Kulkarni","Thakur","Saxena","Arora"]

CITIES_STATES = [
    ("Mumbai","Maharashtra"),("Pune","Maharashtra"),("Delhi","Delhi"),("Bangalore","Karnataka"),
    ("Hyderabad","Telangana"),("Chennai","Tamil Nadu"),("Kolkata","West Bengal"),("Ahmedabad","Gujarat"),
    ("Jaipur","Rajasthan"),("Lucknow","Uttar Pradesh"),("Chandigarh","Punjab"),("Bhopal","Madhya Pradesh"),
    ("Indore","Madhya Pradesh"),("Nagpur","Maharashtra"),("Coimbatore","Tamil Nadu"),
    ("Kochi","Kerala"),("Visakhapatnam","Andhra Pradesh"),("Surat","Gujarat"),("Noida","Uttar Pradesh"),("Gurgaon","Haryana")
]

CATEGORIES = ["Electronics","Fashion","Grocery","Home & Kitchen","Books","Sports","Beauty","Toys","Automotive","Health"]

BRANDS = {
    "Electronics": ["Samsung","Apple","OnePlus","Xiaomi","Sony","LG","Boat","JBL","Dell","HP","Lenovo","Realme"],
    "Fashion": ["Nike","Adidas","Puma","Levi's","H&M","Zara","Allen Solly","Peter England","Biba","W"],
    "Grocery": ["Tata","Amul","Nestle","Britannia","ITC","MDH","Patanjali","Fortune","Aashirvaad","Saffola"],
    "Home & Kitchen": ["Prestige","Pigeon","Bajaj","Philips","Havells","Crompton","Borosil","Milton","Cello","Wonderchef"],
    "Books": ["Penguin","HarperCollins","Scholastic","Oxford","Cambridge","Pearson","Wiley","McGraw-Hill","Rupa","Bloomsbury"],
    "Sports": ["Nike","Adidas","Puma","Yonex","Cosco","Nivia","SG","SS","MRF","Spartan"],
    "Beauty": ["Lakme","Maybelline","L'Oreal","Nykaa","Himalaya","Biotique","Dove","Nivea","Garnier","Neutrogena"],
    "Toys": ["Lego","Hot Wheels","Barbie","Funskool","Fisher-Price","Hasbro","Nerf","Play-Doh","Mattel","Toyzone"],
    "Automotive": ["Bosch","Castrol","3M","Philips","Amaron","Exide","Ceat","MRF","Michelin","Goodyear"],
    "Health": ["Himalaya","Dabur","Patanjali","Baidyanath","Zandu","Revital","Centrum","Ensure","Protinex","HealthVit"]
}

PRODUCT_TEMPLATES = {
    "Electronics": ["Smartphone","Laptop","Earbuds","Smartwatch","Tablet","Speaker","Mouse","Keyboard","Monitor","Charger","Power Bank","Camera"],
    "Fashion": ["T-Shirt","Jeans","Sneakers","Jacket","Kurta","Saree","Watch","Sunglasses","Backpack","Wallet"],
    "Grocery": ["Rice 5kg","Wheat Flour 10kg","Cooking Oil 5L","Tea 500g","Coffee 200g","Sugar 5kg","Salt 1kg","Spice Box","Ghee 1L","Oats 1kg"],
    "Home & Kitchen": ["Mixer Grinder","Pressure Cooker","Non-stick Pan","Water Bottle","Dinner Set","Bedsheet","Curtains","Vacuum Cleaner","Iron","Toaster"],
    "Books": ["Fiction Novel","Self-Help Book","Textbook","Biography","Cookbook","Children Book","Science Book","History Book","Art Book","Poetry Collection"],
    "Sports": ["Cricket Bat","Football","Badminton Racket","Yoga Mat","Running Shoes","Gym Gloves","Resistance Band","Skipping Rope","Dumbbell Set","Helmet"],
    "Beauty": ["Face Wash","Moisturizer","Sunscreen","Lipstick","Foundation","Shampoo","Conditioner","Hair Oil","Perfume","Face Mask"],
    "Toys": ["Building Blocks","Remote Control Car","Board Game","Puzzle Set","Doll House","Action Figure","Art Kit","Science Kit","Toy Train","Plush Toy"],
    "Automotive": ["Car Charger","Dash Camera","Tyre Inflator","Car Cover","Seat Cover","Air Freshener","Tool Kit","Jump Starter","Wiper Blade","LED Bulb"],
    "Health": ["Multivitamin","Protein Powder","Omega-3","Calcium Tablets","Immunity Booster","Probiotic","Glucometer","BP Monitor","Thermometer","First Aid Kit"]
}

COURIERS = ["Delhivery","BlueDart","DTDC","Ekart","Shadowfax","XpressBees","FedEx","Ecom Express"]
PAYMENT_METHODS = ["UPI","Card","COD","NetBanking","Wallet"]
ORDER_STATUSES = ["Placed","Packed","Shipped","Delivered","Cancelled","Returned"]
PROMO_CODES = ["WELCOME10","SAVE20","FLAT50","MEGA100","FESTIVE15","SUMMER25","WINTER30","NEW500","FLASH10","LOYALTY20"]

random.seed(42)

def random_date(start_year=2024, end_year=2026):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 2, 15)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

def random_phone():
    return f"+91{random.randint(7000000000,9999999999)}"

def random_pincode():
    return str(random.randint(100000, 999999))

def create_and_seed():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    
    # 1) Seed customers (500)
    customers = []
    emails_used = set()
    for i in range(500):
        if random.random() < 0.5:
            fn = random.choice(FIRST_NAMES_M)
            gender = "Male"
        else:
            fn = random.choice(FIRST_NAMES_F)
            gender = "Female"
        ln = random.choice(LAST_NAMES)
        full_name = f"{fn} {ln}"
        base_email = f"{fn.lower()}.{ln.lower()}{random.randint(1,999)}@{'gmail.com' if random.random()<0.7 else 'yahoo.com'}"
        while base_email in emails_used:
            base_email = f"{fn.lower()}.{ln.lower()}{random.randint(1,9999)}@gmail.com"
        emails_used.add(base_email)
        dj = random_date(2022, 2026).strftime("%Y-%m-%d")
        tier = random.choice(["Bronze","Silver","Gold","Platinum"])
        customers.append((full_name, base_email, random_phone(), gender, dj, tier))
    
    cur.executemany("INSERT INTO customers (full_name,email,phone,gender,date_joined,loyalty_tier) VALUES (?,?,?,?,?,?)", customers)
    
    # 2) Seed addresses (1-2 per customer)
    addresses = []
    for cid in range(1, 501):
        city, state = random.choice(CITIES_STATES)
        addresses.append((cid, "Home", city, state, random_pincode()))
        if random.random() < 0.3:
            city2, state2 = random.choice(CITIES_STATES)
            addresses.append((cid, "Office", city2, state2, random_pincode()))
    cur.executemany("INSERT INTO addresses (customer_id,address_type,city,state,pincode) VALUES (?,?,?,?,?)", addresses)
    
    # 3) Seed categories
    for cat in CATEGORIES:
        cur.execute("INSERT INTO categories (category_name) VALUES (?)", (cat,))
    
    # 4) Seed products (2000)
    products = []
    for i in range(2000):
        cat_idx = random.randint(0, len(CATEGORIES)-1)
        cat_name = CATEGORIES[cat_idx]
        cat_id = cat_idx + 1
        brand = random.choice(BRANDS[cat_name])
        template = random.choice(PRODUCT_TEMPLATES[cat_name])
        pname = f"{brand} {template} {random.choice(['Pro','Lite','Max','Plus','Elite','Basic','Standard','Premium','Ultra','Mini'])}"
        price = round(random.uniform(99, 49999), 2)
        stock = random.randint(0, 500)
        rating = round(random.uniform(1.0, 5.0), 1)
        products.append((pname, cat_id, brand, price, stock, rating))
    cur.executemany("INSERT INTO products (product_name,category_id,brand,unit_price,stock_qty,rating_avg) VALUES (?,?,?,?,?,?)", products)
    
    # 5) Seed orders (5000)
    addr_count = len(addresses)
    orders_data = []
    for i in range(5000):
        cid = random.randint(1, 500)
        odate = random_date(2024, 2026).strftime("%Y-%m-%d")
        status = random.choices(ORDER_STATUSES, weights=[10,5,10,50,15,10])[0]
        aid = random.randint(1, addr_count)
        orders_data.append((cid, odate, status, aid, 0.0))
    cur.executemany("INSERT INTO orders (customer_id,order_date,order_status,shipping_address_id,order_total) VALUES (?,?,?,?,?)", orders_data)
    
    # 6) Seed order_items (1-5 items per order -> ~12500 items)
    order_items = []
    order_totals = {}
    for oid in range(1, 5001):
        n_items = random.randint(1, 5)
        total = 0.0
        for _ in range(n_items):
            pid = random.randint(1, 2000)
            qty = random.randint(1, 3)
            up = round(random.uniform(99, 9999), 2)
            disc = round(random.uniform(0, up*0.2*qty), 2)
            lt = round(qty * up - disc, 2)
            total += lt
            order_items.append((oid, pid, qty, up, disc, lt))
        order_totals[oid] = round(total, 2)
    cur.executemany("INSERT INTO order_items (order_id,product_id,quantity,unit_price,discount_amount,line_total) VALUES (?,?,?,?,?,?)", order_items)
    
    # Update order totals
    for oid, total in order_totals.items():
        cur.execute("UPDATE orders SET order_total=? WHERE order_id=?", (total, oid))
    
    # 7) Seed payments (one per order)
    payments = []
    for oid in range(1, 5001):
        odate = orders_data[oid-1][1]
        pm = random.choice(PAYMENT_METHODS)
        ps = random.choices(["Success","Failed","Pending","Refunded"], weights=[80,5,10,5])[0]
        payments.append((oid, odate, pm, ps, order_totals[oid]))
    cur.executemany("INSERT INTO payments (order_id,payment_date,payment_method,payment_status,amount_paid) VALUES (?,?,?,?,?)", payments)
    
    # 8) Seed shipments (for non-cancelled orders)
    shipments = []
    for oid in range(1, 5001):
        status = orders_data[oid-1][2]
        if status == "Cancelled":
            continue
        odate_str = orders_data[oid-1][1]
        odate = datetime.strptime(odate_str, "%Y-%m-%d")
        shipped = (odate + timedelta(days=random.randint(1,3))).strftime("%Y-%m-%d")
        delivered = None
        ss = "Shipped"
        if status == "Delivered":
            delivered = (odate + timedelta(days=random.randint(3,10))).strftime("%Y-%m-%d")
            ss = "Delivered"
        elif status == "Shipped":
            ss = random.choice(["Shipped","InTransit"])
        elif status == "Returned":
            delivered = (odate + timedelta(days=random.randint(3,7))).strftime("%Y-%m-%d")
            ss = "Delivered"
        courier = random.choice(COURIERS)
        fee = round(random.choice([0, 40, 50, 60, 99]), 2)
        shipments.append((oid, courier, shipped, delivered, ss, fee))
    cur.executemany("INSERT INTO shipments (order_id,courier_name,shipped_date,delivered_date,shipping_status,shipping_fee) VALUES (?,?,?,?,?,?)", shipments)
    
    # 9) Seed returns (~5-8% of delivered orders)
    returns_data = []
    for oid in range(1, 5001):
        status = orders_data[oid-1][2]
        if status == "Returned" or (status == "Delivered" and random.random() < 0.05):
            odate = datetime.strptime(orders_data[oid-1][1], "%Y-%m-%d")
            rdate = (odate + timedelta(days=random.randint(5,20))).strftime("%Y-%m-%d")
            reason = random.choice(["Damaged","WrongItem","NotNeeded","Delayed","Other"])
            refund = round(order_totals[oid] * random.uniform(0.5, 1.0), 2)
            rs = random.choice(["Requested","Approved","PickedUp","Refunded","Rejected"])
            returns_data.append((oid, rdate, reason, refund, rs))
    cur.executemany("INSERT INTO returns (order_id,return_date,return_reason,refund_amount,return_status) VALUES (?,?,?,?,?)", returns_data)
    
    # 10) Seed promotions
    promos = []
    for i, code in enumerate(PROMO_CODES):
        dt = random.choice(["PERCENT","FIXED"])
        dv = random.choice([5,10,15,20,25,50,100,200,500]) if dt == "FIXED" else random.choice([5,10,15,20,25,30])
        sd = random_date(2025, 2026).strftime("%Y-%m-%d")
        ed_dt = datetime.strptime(sd, "%Y-%m-%d") + timedelta(days=random.randint(15,90))
        ed = ed_dt.strftime("%Y-%m-%d")
        mov = random.choice([0, 500, 1000, 2000, 5000])
        promos.append((code, dt, dv, sd, ed, mov))
    cur.executemany("INSERT INTO promotions (promo_code,discount_type,discount_value,start_date,end_date,min_order_value) VALUES (?,?,?,?,?,?)", promos)
    
    conn.commit()
    
    # Print stats
    tables = ["customers","addresses","categories","products","orders","order_items","payments","shipments","returns","promotions"]
    print("=== E-Commerce Database Seeded ===")
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"  {t}: {cur.fetchone()[0]} rows")
    
    conn.close()
    print(f"\nDatabase saved to: {DB_PATH}")

if __name__ == "__main__":
    create_and_seed()
