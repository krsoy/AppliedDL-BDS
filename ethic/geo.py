import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===========================
# 1. Load Excel
# ===========================
file = "202622311235604188520UHM.xlsx"
df = pd.read_excel(file, header=None, engine="openpyxl")

# ===========================
# 2. Detect time row
# ===========================
time_pattern = re.compile(r"^\d{4}M(0[1-9]|1[0-2])$")

for idx in range(len(df)):
    row = df.iloc[idx].astype(str)
    if row.str.match(time_pattern).sum() >= 6:
        time_row = idx
        break

time_cols = df.iloc[time_row].astype(str).str.match(time_pattern)
first_time_col = time_cols.idxmax()
time_labels = df.iloc[time_row, first_time_col:].astype(str).tolist()

# ===========================
# 3. Find FIRST row of each country
# ===========================
def find_first_row(key):
    for i in range(len(df)):
        if str(df.iat[i, 2]).strip() == key:
            return i
    return None

US_row = find_first_row("United States")
CN_row = find_first_row("China")

US_start, US_end = US_row, CN_row
CN_start, CN_end = CN_row, len(df)

# ===========================
# 4. Normalize block
# ===========================
def normalize(df, country, start, end):
    block = df.iloc[start:end].reset_index(drop=True)
    current_flow = None
    current_item = None
    rows = []

    for i in range(len(block)):
        row = block.iloc[i]

        # Flow
        if isinstance(row[3], str) and row[3] in ["Imports", "Exports"]:
            current_flow = row[3]

        # Item
        if isinstance(row[4], str) and row[4].strip() != "":
            current_item = row[4].strip()

        # Values
        vals = pd.to_numeric(row[first_time_col:], errors="coerce")
        if vals.notna().any():
            for t, v in zip(time_labels, vals):
                if pd.notna(v):
                    rows.append({
                        "Country": country,
                        "Flow": current_flow,
                        "Item": current_item,
                        "Time": t,
                        "Value": float(v)
                    })
    out = pd.DataFrame(rows)
    out["Time"] = pd.to_datetime(out["Time"].str.replace("M", "-"), format="%Y-%m")
    return out

# ===========================
# 5. Create final us_df / china_df
# ===========================
us_df = normalize(df, "United States", US_start, US_end)
china_df = normalize(df, "China", CN_start, CN_end)

print(len(us_df), "rows in us_df")
print(len(china_df), "rows in china_df")

# ===========================
# 6. Plot
# ===========================
plot_df = pd.concat([us_df, china_df])

for u in plot_df["Item"].unique():
    _p = "Exports"
    sub = plot_df[
        (plot_df["Flow"] == _p) &
        (plot_df["Item"] == u)
    ]

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=sub, x="Time", y="Value", hue="Country")
    plt.title(f"{_p} – {u} (US vs China)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()