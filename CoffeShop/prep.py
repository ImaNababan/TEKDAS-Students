import pandas as pd

# Ganti nama file sesuai file Excel kamu
df = pd.read_excel("Coffee Shop Sales.xlsx", sheet_name="Transactions")

# Cek struktur
print(df.shape)  # harus (149116, 11)
print(df.dtypes)
print(df.head())

# Nilai penjualan per baris
df["sales"] = df["transaction_qty"] * df["unit_price"]

# Rekonsiliasi dengan angka baseline di slide
print("Total penjualan :", round(df["sales"].sum(), 2))  # 698812.33
print("Transaksi unik  :", df["transaction_id"].nunique())  # 149116
print("Total unit      :", df["transaction_qty"].sum())  # 214470
print(
    "Rata2 transaksi :",
    round(df["sales"].sum() / df["transaction_id"].nunique(), 2),
)  # 4.69

# Penjualan per bulan
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df["month"] = df["transaction_date"].dt.to_period("M").astype(str)
print(df.groupby("month")["sales"].sum().round(2))