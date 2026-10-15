from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
PREPARED_FILE = BASE_DIR / "data_prepared" / "coffee_sales_prepared.csv"
SOURCE_FILE = BASE_DIR / "Coffee Shop Sales.xlsx"

st.set_page_config(page_title="Coffee Shop | Analisis Penjualan", page_icon="☕", layout="wide")

COFFEE = "#76513B"
PALETTE = ["#76513B", "#B67B52", "#D7A879", "#5B7C70", "#D9B44A", "#9B6A71", "#76889B"]
DAY_ORDER = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]


@st.cache_data
def load_data():
    """Load prepared transactions or derive dashboard fields from the source workbook."""
    if PREPARED_FILE.exists():
        data = pd.read_csv(PREPARED_FILE)
    elif SOURCE_FILE.exists():
        data = pd.read_excel(SOURCE_FILE, sheet_name="Transactions")
    else:
        raise FileNotFoundError("File data tidak ditemukan di folder CoffeShop.")

    data["transaction_date"] = pd.to_datetime(data["transaction_date"])
    data["sales"] = data["transaction_qty"] * data["unit_price"]
    if "month" not in data:
        data["month"] = data["transaction_date"].dt.strftime("%Y-%m")
    if "hour" not in data:
        data["hour"] = pd.to_datetime(data["transaction_time"], format="%H:%M:%S").dt.hour
    names = dict(enumerate(DAY_ORDER))
    if "day_name_id" not in data:
        data["day_name_id"] = data["transaction_date"].dt.dayofweek.map(names)
    return data

"""ini adalah file utama untuk menjalankan dashboard analisis penjualan Coffee Shop. Pastikan file data tersedia di folder yang sesuai."""
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root { --ink:#302821; --muted:#74685e; --cream:#faf7f2; --line:#eee6dc; --coffee:#76513b; }
.stApp { background:var(--cream); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:#f1e9df; border-right:1px solid var(--line); }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:var(--ink); }
[data-testid="stSidebar"] .block-container { padding:1.5rem 1.15rem 2rem; }
[data-testid="stSidebar"] [data-testid="stMarkdown"] h2 { font-family:'Playfair Display',serif; font-size:1.4rem; margin-bottom:.15rem; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#786b60; line-height:1.5; }
[data-testid="stSidebar"] label { color:#44372e; font-weight:600; font-size:.9rem; }
[data-testid="stSidebar"] [data-testid="stMultiSelect"] > div > div,
[data-testid="stSidebar"] [data-testid="stDateInput"] > div > div { background:#fffdfb; border-color:#dfd2c4; border-radius:10px; }
[data-testid="stSidebar"] [data-testid="stDateInput"] input,
[data-testid="stSidebar"] [data-testid="stDateInput"] input:focus,
[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"] *,
[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="popover"] * {
    color:#111 !important; -webkit-text-fill-color:#111 !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] { background:#76513b; border-radius:7px; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { margin-bottom:.35rem; }
[data-testid="stSidebar"] hr { border-color:#dfd2c4; margin:1.25rem 0; }
[data-testid="stSidebar"] [data-testid="stMarkdown"] p { font-size:.86rem; }
.block-container { padding-top:2.1rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3 { color:var(--ink); }
h1 { font-family:'Playfair Display',serif !important; letter-spacing:-.4px; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:17px 19px; border-radius:14px; box-shadow:0 3px 12px #39231108; }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { color:var(--coffee); font-weight:700; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.hero { padding:24px 28px; border-radius:18px; margin:0 0 22px; background:linear-gradient(115deg,#3b2d25,#76513b 75%,#9e7354); color:#fff; }
.hero h1 { color:#fff !important; margin:0 0 5px; font-size:2.15rem; }
.hero p { color:#f1e4d7; margin:0; font-size:1rem; }
.insight { background:#fff; border:1px solid var(--line); border-left:4px solid #b67b52; border-radius:12px; padding:13px 16px; min-height:74px; color:var(--ink); }
.section-note { color:var(--muted); margin-top:-12px; margin-bottom:16px; }
</style>
""", unsafe_allow_html=True)

df = load_data()
min_date, max_date = df["transaction_date"].min().date(), df["transaction_date"].max().date()

st.markdown(f"""
<div class="hero">
  <h1>☕ Coffee Shop — Analisis Penjualan</h1>
  <p>Pantau performa gerai, produk, dan pola permintaan untuk menemukan peluang operasional.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ☕ Ruang analisis")
    st.caption("Pilih cakupan data. Semua angka dan visual akan menyesuaikan filter ini.")
    st.markdown("#### FILTER DATA")
    stores = sorted(df["store_location"].dropna().unique())
    selected_stores = st.multiselect("Gerai", stores, default=stores)
    categories = sorted(df["product_category"].dropna().unique())
    selected_categories = st.multiselect("Kategori produk", categories, default=categories)
    date_range = st.date_input("Rentang tanggal", (min_date, max_date), min_value=min_date, max_value=max_date)
    st.divider()
    st.markdown("#### TENTANG ANGKA")
    st.caption("Penjualan = jumlah unit × harga satuan. Dataset tidak mencantumkan mata uang.")

def date_values(value):
    """Extract date values from Streamlit's scalar or range return value."""
    if isinstance(value, (tuple, list)):
        return [item for part in value for item in date_values(part)]
    return [value]


selected_dates = date_values(date_range)
if not selected_dates:
    start_date, end_date = min_date, max_date
elif len(selected_dates) == 1:
    start_date = end_date = pd.Timestamp(selected_dates[0]).date()
else:
    start_date = pd.Timestamp(selected_dates[0]).date()
    end_date = pd.Timestamp(selected_dates[-1]).date()
    if start_date > end_date:
        start_date, end_date = end_date, start_date

d = df[
    df["store_location"].isin(selected_stores)
    & df["product_category"].isin(selected_categories)
    & df["transaction_date"].dt.normalize().between(pd.Timestamp(start_date), pd.Timestamp(end_date))
].copy()

if d.empty:
    st.warning("Tidak ada transaksi untuk kombinasi filter ini. Coba perluas tanggal, gerai, atau kategori.")
    st.stop()

# Distinct transaction IDs are presented as recorded IDs, not assumed to be complete receipts.
total_sales = d["sales"].sum()
transactions = d["transaction_id"].nunique()
units = d["transaction_qty"].sum()
days = d["transaction_date"].dt.date.nunique()
month_stats = d.groupby("month", as_index=False).agg(
    sales=("sales", "sum"), transactions=("transaction_id", "nunique"),
    units=("transaction_qty", "sum"), days=("transaction_date", lambda s: s.dt.date.nunique())
).sort_values("month")
month_stats["sales_per_day"] = month_stats["sales"] / month_stats["days"].clip(lower=1)
month_stats["daily_growth"] = month_stats["sales_per_day"].pct_change() * 100
last_growth = month_stats["daily_growth"].iloc[-1] if len(month_stats) > 1 else float("nan")

peak_hour = int(d.groupby("hour")["transaction_id"].nunique().idxmax())
top_category = d.groupby("product_category")["sales"].sum().idxmax()
top_store = d.groupby("store_location")["sales"].sum().idxmax()
category_share = d.groupby("product_category")["sales"].sum().max() / total_sales * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total penjualan", f"{total_sales:,.2f}")
k2.metric("ID transaksi tercatat", f"{transactions:,}")
k3.metric("Unit terjual", f"{units:,}")
k4.metric("Penjualan rata-rata / hari", f"{total_sales / max(days, 1):,.2f}")
k5.metric("Perubahan harian bulan terakhir", f"{last_growth:+.1f}%" if pd.notna(last_growth) else "—")
st.caption("Pertumbuhan membandingkan penjualan rata-rata per hari antarbulan, sehingga panjang bulan tidak mendistorsi perbandingan. ID transaksi tidak diasumsikan mewakili satu struk lengkap.")

st.subheader("Ringkasan temuan")
i1, i2, i3 = st.columns(3)
i1.markdown(f'<div class="insight"><b>Kategori teratas</b><br>{top_category} menyumbang {category_share:.1f}% dari penjualan pada filter aktif.</div>', unsafe_allow_html=True)
i2.markdown(f'<div class="insight"><b>Gerai dengan penjualan tertinggi</b><br>{top_store} memimpin pada rentang dan kategori yang dipilih.</div>', unsafe_allow_html=True)
i3.markdown(f'<div class="insight"><b>Jam tersibuk</b><br>Pukul {peak_hour:02d}. Pertimbangkan kesiapan stok dan jadwal staf di sekitar jam ini.</div>', unsafe_allow_html=True)

def polish(fig, height=350):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=12, r=12, t=55, b=12),
        title=dict(font=dict(size=16, color="#302821")), font=dict(family="DM Sans, sans-serif", color="#574b42"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="", colorway=PALETTE,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#eee6dc", zeroline=False)
    return fig

st.subheader("Tren penjualan")
st.markdown('<div class="section-note">Lihat total bulanan bersama rata-rata harian untuk membedakan skala bulan dari perubahan laju penjualan.</div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    fig = px.line(month_stats, x="month", y="sales", markers=True, title="Total penjualan per bulan", labels={"month": "Bulan", "sales": "Penjualan"})
    fig.update_traces(line=dict(color=COFFEE, width=3), marker=dict(size=8))
    st.plotly_chart(polish(fig), use_container_width=True)
with right:
    fig = px.bar(month_stats, x="month", y="sales_per_day", title="Rata-rata penjualan per hari", labels={"month": "Bulan", "sales_per_day": "Penjualan / hari"})
    fig.update_traces(marker_color="#B67B52", marker_line_width=0)
    st.plotly_chart(polish(fig), use_container_width=True)
st.dataframe(month_stats[["month", "sales", "sales_per_day", "transactions", "units", "daily_growth"]].rename(columns={
    "month": "Bulan", "sales": "Penjualan", "sales_per_day": "Penjualan / hari", "transactions": "ID transaksi",
    "units": "Unit", "daily_growth": "Perubahan harian (%)"
}).round(2), hide_index=True, use_container_width=True)

st.subheader("Performa gerai")
store_stats = d.groupby("store_location", as_index=False).agg(
    sales=("sales", "sum"), transactions=("transaction_id", "nunique"), units=("transaction_qty", "sum"),
    days=("transaction_date", lambda s: s.dt.date.nunique())
).sort_values("sales", ascending=False)
store_stats["sales_per_day"] = store_stats["sales"] / store_stats["days"].clip(lower=1)
col1, col2 = st.columns(2)
with col1:
    fig = px.bar(store_stats.sort_values("sales"), x="sales", y="store_location", orientation="h", title="Total penjualan per gerai", labels={"sales": "Penjualan", "store_location": "Gerai"}, color="sales", color_continuous_scale=["#e8d8c8", COFFEE])
    st.plotly_chart(polish(fig), use_container_width=True)
with col2:
    store_month = d.groupby(["month", "store_location"], as_index=False)["sales"].sum()
    fig = px.line(store_month, x="month", y="sales", color="store_location", markers=True, title="Tren bulanan per gerai", labels={"month": "Bulan", "sales": "Penjualan", "store_location": "Gerai"})
    st.plotly_chart(polish(fig), use_container_width=True)
st.dataframe(store_stats[["store_location", "sales", "sales_per_day", "transactions", "units"]].rename(columns={"store_location": "Gerai", "sales": "Penjualan", "sales_per_day": "Penjualan / hari", "transactions": "ID transaksi", "units": "Unit"}).round(2), hide_index=True, use_container_width=True)

st.subheader("Komposisi kategori dan produk")
cat_stats = d.groupby("product_category", as_index=False).agg(sales=("sales", "sum"), units=("transaction_qty", "sum"))
cat_stats["share"] = cat_stats["sales"] / total_sales * 100
cat_stats = cat_stats.sort_values("sales", ascending=False)
col1, col2 = st.columns(2)
with col1:
    fig = px.bar(cat_stats.sort_values("sales"), x="sales", y="product_category", orientation="h", title="Penjualan dan kontribusi kategori", labels={"sales": "Penjualan", "product_category": "Kategori"}, color="product_category", color_discrete_sequence=PALETTE, text=cat_stats.sort_values("sales")["share"].map(lambda x: f"{x:.1f}%"))
    fig.update_traces(textposition="outside", cliponaxis=False, showlegend=False)
    st.plotly_chart(polish(fig), use_container_width=True)
with col2:
    top_products = d.groupby("product_detail", as_index=False).agg(sales=("sales", "sum"), units=("transaction_qty", "sum")).nlargest(10, "sales").sort_values("sales")
    fig = px.bar(top_products, x="sales", y="product_detail", orientation="h", title="10 produk dengan penjualan tertinggi", labels={"sales": "Penjualan", "product_detail": "Produk"}, color="sales", color_continuous_scale=["#e8d8c8", COFFEE])
    st.plotly_chart(polish(fig), use_container_width=True)
st.dataframe(cat_stats.rename(columns={"product_category": "Kategori", "sales": "Penjualan", "units": "Unit", "share": "Kontribusi (%)"}).round(2), hide_index=True, use_container_width=True)

st.subheader("Pola permintaan")
st.markdown('<div class="section-note">Jumlah ID transaksi menggambarkan frekuensi catatan penjualan; penjualan per hari membantu membandingkan hari dengan jumlah kemunculan berbeda.</div>', unsafe_allow_html=True)
hour_stats = d.groupby("hour", as_index=False).agg(sales=("sales", "sum"), transactions=("transaction_id", "nunique"), units=("transaction_qty", "sum"))
day_stats = d.groupby("day_name_id", as_index=False).agg(sales=("sales", "sum"), transactions=("transaction_id", "nunique"), days=("transaction_date", lambda s: s.dt.date.nunique()))
day_stats["sales_per_day"] = day_stats["sales"] / day_stats["days"].clip(lower=1)
day_stats["day_name_id"] = pd.Categorical(day_stats["day_name_id"], categories=DAY_ORDER, ordered=True)
day_stats = day_stats.sort_values("day_name_id")
col1, col2 = st.columns(2)
with col1:
    fig = px.bar(hour_stats, x="hour", y="transactions", title="Aktivitas transaksi menurut jam", labels={"hour": "Jam", "transactions": "ID transaksi"})
    fig.update_traces(marker_color="#B67B52")
    st.plotly_chart(polish(fig), use_container_width=True)
with col2:
    fig = px.bar(day_stats, x="day_name_id", y="sales_per_day", title="Rata-rata penjualan per hari dalam minggu", labels={"day_name_id": "Hari", "sales_per_day": "Penjualan / hari"})
    fig.update_traces(marker_color="#5B7C70")
    st.plotly_chart(polish(fig), use_container_width=True)

with st.expander("Rincian jenis produk per kategori"):
    chosen_category = st.selectbox("Pilih kategori", cat_stats["product_category"].tolist())
    detail = d[d["product_category"] == chosen_category].groupby("product_type", as_index=False).agg(sales=("sales", "sum"), units=("transaction_qty", "sum")).sort_values("sales", ascending=False)
    detail["share"] = detail["sales"] / detail["sales"].sum() * 100
    st.dataframe(detail.rename(columns={"product_type": "Jenis produk", "sales": "Penjualan", "units": "Unit", "share": "Kontribusi (%)"}).round(2), hide_index=True, use_container_width=True)

st.caption(f"Analisis mencakup {start_date:%d %b %Y}–{end_date:%d %b %Y} · {len(selected_stores)} gerai dipilih · {len(selected_categories)} kategori dipilih.")
