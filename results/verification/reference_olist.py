"""Independent reference values for the Olist tables (dev split). pandas only."""

import sys
from pathlib import Path

import pandas as pd

orders_root = Path(sys.argv[1])
orders = pd.read_csv(orders_root / "olist_orders_dataset.csv")
items = pd.read_csv(orders_root / "olist_order_items_dataset.csv")
pay = pd.read_csv(orders_root / "olist_order_payments_dataset.csv")
rev = pd.read_csv(orders_root / "olist_order_reviews_dataset.csv")
cust = pd.read_csv(orders_root / "olist_customers_dataset.csv")
prod = pd.read_csv(orders_root / "olist_products_dataset.csv")
sell = pd.read_csv(orders_root / "olist_sellers_dataset.csv")
geo = pd.read_csv(orders_root / "olist_geolocation_dataset.csv")
tr = pd.read_csv(orders_root / "product_category_name_translation.csv")
print(
    "orders",
    orders.shape,
    orders.order_id.is_unique,
    "items",
    items.shape,
    "pay",
    pay.shape,
    "rev",
    rev.shape,
    "cust",
    cust.shape,
    cust.customer_id.is_unique,
    cust.customer_unique_id.nunique(),
)
print(
    "payment total",
    pay.payment_value.sum(),
    "mean per payment row",
    pay.payment_value.mean(),
    "mean per order",
    pay.groupby("order_id").payment_value.sum().mean(),
    "orders with payments",
    pay.order_id.nunique(),
)
print("orders per status", orders.order_status.value_counts().head(3).to_dict())
print(
    "items order_id not in orders",
    (~items.order_id.isin(orders.order_id)).sum(),
    "orders w/o items",
    (~orders.order_id.isin(items.order_id)).sum(),
)
print(
    "reviews review_id unique",
    rev.review_id.is_unique,
    "orders with >1 review",
    (rev.order_id.value_counts() > 1).sum(),
)
print(
    "geo zip prefix unique",
    geo.geolocation_zip_code_prefix.is_unique,
    "rows per prefix max",
    geo.geolocation_zip_code_prefix.value_counts().max(),
)
print("customers zip in geo", cust.customer_zip_code_prefix.isin(geo.geolocation_zip_code_prefix).mean())
print(
    "product categories untranslated",
    sorted(set(prod.product_category_name.dropna()) - set(tr.product_category_name)),
)
