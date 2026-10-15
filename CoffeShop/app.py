from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
PREPARED_FILE = BASE_DIR / "data_prepared" / "coffee_sales_prepared.csv"
SOURCE_FILE = BASE_DIR / "Coffee Shop Sales.xlsx"

st.set_page_config(page_title="Coffee Shop BI", layout="wide")


@st.cache_data
def load_data():
    """Load audited data when available; otherwise derive dashboard fields."""
    if PREPARED_FILE.exists():
        df = pd.read_csv(PREPARED_FILE)
    elif SOURCE_FILE.exists():
        df = pd.read_excel(SOURCE_FILE, sheet_name="Transactions")
    else:
        raise FileNotFoundError("File data tidak ditemukan di folder CoffeShop.")

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["sales"] = df["transaction_qty"] * df["unit_price"]
    if "month" not in df:
        df["month"] = df["transaction_date"].dt.strftime("%Y-%m")
    if "hour" not in df:
        df["hour"] = pd.to_datetime(df["transaction_time"], format="%H:%M:%S").dt.hour
    if "day_name_id" not in df:
        names = {0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis",
                 4: "Jumat", 5: "Sabtu", 6: "Minggu"}
        df["day_name_id"] = df["transaction_date"].dt.dayofweek.map(names)
    return df


df = load_data()
st.title("Dashboard Penjualan Coffee Shop")
st.caption("Periode data: Januari–Juni 2023. Nilai penjualan mengikuti satuan pada dataset; mata uang tidak dicantumkan.")

# Filter data
st.sidebar.header("Filter")
stores = sorted(df["store_location"].dropna().unique())
selected_stores = st.sidebar.multiselect("Gerai", stores, default=stores)
categories = sorted(df["product_category"].dropna().unique())
selected_categories = st.sidebar.multiselect("Kategori", categories, default=categories)
date_min = df["transaction_date"].min().date()
date_max = df["transaction_date"].max().date()
date_range = st.sidebar.date_input("Rentang tanggal", (date_min, date_max), min_value=date_min, max_value=date_max)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

d = df[
    df["store_location"].isin(selected_stores)
    & df["product_category"].isin(selected_categories)
    & df["transaction_date"].dt.date.between(start_date, end_date)
].copy()

if d.empty:
    st.warning("Tidak ada transaksi untuk kombinasi filter ini.")
    st.stop()

# Ringkasan KPI. Satu baris = satu transaction_id dalam file ini; label menghindari klaim nilai struk.
total_sales = d["sales"].sum()
transactions = d["transaction_id"].nunique()
units = d["transaction_qty"].sum()
recorded_days = d["transaction_date"].dt.date.nunique()
month_stats = d.groupby("month", as_index=False).agg(
    sales=("sales", "sum"), transactions=("transaction_id", "nunique"),
    units=("transaction_qty", "sum"), days=("transaction_date", lambda s: s.dt.date.nunique())
).sort_values("month")
month_stats["sales_per_day"] = month_stats["sales"] / month_stats["days"]
month_stats["transaction_value_per_id"] = month_stats["sales"] / month_stats["transactions"]
month_stats["growth_pct"] = month_stats["sales_per_day"].pct_change() * 100
last_growth = month_stats["growth_pct"].iloc[-1] if len(month_stats) > 1 else float("nan")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total penjualan", f"{total_sales:,.2f}")
k2.metric("ID transaksi tercatat", f"{transactions:,}")
k3.metric("Unit terjual", f"{units:,}")
k4.metric("Penjualan per hari", f"{total_sales / max(recorded_days, 1):,.2f}")
k5.metric("Perubahan penjualan harian bulan terakhir", f"{last_growth:+.1f}%" if pd.notna(last_growth) else "—")
st.caption("Rata-rata nilai per ID transaksi tidak ditampilkan sebagai nilai struk karena definisi ID belum dipastikan mewakili seluruh isi struk.")

st.subheader("Tren bulanan")
left, right = st.columns(2)
with left:
    fig = px.line(month_stats, x="month", y="sales", markers=True, title="Total penjualan")
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.line(month_stats, x="month", y="sales_per_day", markers=True,
                  title="Penjualan rata-rata per hari (memperhitungkan panjang bulan)")
    st.plotly_chart(fig, use_container_width=True)
st.dataframe(month_stats[["month", "sales", "transactions", "units", "growth_pct"]].rename(columns={
    "month": "Bulan", "sales": "Penjualan", "transactions": "ID transaksi",
    "units": "Unit", "growth_pct": "Perubahan penjualan harian (%)"
}).round(2), hide_index=True, use_container_width=True)

st.subheader("Perbandingan gerai")
store_stats = d.groupby("store_location", as_index=False).agg(
    sales=("sales", "sum"), transactions=("transaction_id", "nunique"), units=("transaction_qty", "sum")
).sort_values("sales", ascending=False)
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.bar(store_stats, x="store_location", y="sales", color="store_location",
                           title="Penjualan per gerai"), use_container_width=True)
with col2:
    store_month = d.groupby(["month", "store_location"], as_index=False)["sales"].sum()
    st.plotly_chart(px.line(store_month, x="month", y="sales", color="store_location", markers=True,
                            title="Tren penjualan bulanan per gerai"), use_container_width=True)

st.subheader("Kategori dan produk")
cat_stats = d.groupby("product_category", as_index=False).agg(
    sales=("sales", "sum"), units=("transaction_qty", "sum")
).sort_values("sales", ascending=False)
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.bar(cat_stats, x="sales", y="product_category", orientation="h",
                           title="Penjualan per kategori", color="sales"), use_container_width=True)
with col2:
    top_products = d.groupby("product_detail", as_index=False).agg(
        sales=("sales", "sum"), units=("transaction_qty", "sum")
    ).nlargest(10, "sales").sort_values("sales")
    st.plotly_chart(px.bar(top_products, x="sales", y="product_detail", orientation="h",
                           title="10 produk dengan penjualan tertinggi"), use_container_width=True)

st.subheader("Pola waktu transaksi")
col1, col2 = st.columns(2)
hour_stats = d.groupby("hour", as_index=False).agg(sales=("sales", "sum"), transactions=("transaction_id", "nunique"))
day_order = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
day_stats = d.groupby("day_name_id", as_index=False).agg(sales=("sales", "sum"), transactions=("transaction_id", "nunique"))
day_stats["day_name_id"] = pd.Categorical(day_stats["day_name_id"], categories=day_order, ordered=True)
day_stats = day_stats.sort_values("day_name_id")
with col1:
    st.plotly_chart(px.bar(hour_stats, x="hour", y="transactions", title="ID transaksi per jam",
                           labels={"hour": "Jam", "transactions": "ID transaksi"}), use_container_width=True)
with col2:
    st.plotly_chart(px.bar(day_stats, x="day_name_id", y="sales", title="Penjualan per hari dalam minggu",
                           labels={"day_name_id": "Hari", "sales": "Penjualan"}), use_container_width=True)

with st.expander("Rincian kategori dan jenis produk"):
    chosen_category = st.selectbox("Pilih kategori", cat_stats["product_category"].tolist())
    detail = d[d["product_category"] == chosen_category].groupby("product_type", as_index=False).agg(
        sales=("sales", "sum"), units=("transaction_qty", "sum")
    ).sort_values("sales", ascending=False)
    st.dataframe(detail.rename(columns={"product_type": "Jenis produk", "sales": "Penjualan", "units": "Unit"}),
                 hide_index=True, use_container_width=True)
