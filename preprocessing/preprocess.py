import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "dataset" / "hostel_complaints.csv"
out = ROOT / "dataset" / "cleaned_hostel_complaints.csv"

df = pd.read_csv(src)
df = df.drop_duplicates()
numeric_cols = ["Days_Pending","Previous_Complaints","Affected_Rooms","Staff_Available","Resolution_Time_Days"]
for c in numeric_cols:
    df[c] = df[c].fillna(df[c].median())

df.to_csv(out, index=False)
print(f"Saved cleaned dataset to {out}")
