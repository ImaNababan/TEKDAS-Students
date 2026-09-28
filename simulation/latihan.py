
import pandas as pd
from pathlib import Path

data_path = Path(__file__).with_name("customer_ml_dataset.csv")
df = pd.read_csv(data_path)

print(df.shape)
print(df.dtypes)
print(df.head())
print(df.isna().mean().sort_values(ascending=False).head())
