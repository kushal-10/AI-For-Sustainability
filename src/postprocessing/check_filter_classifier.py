#!/usr/bin/env python3
import pandas as pd

OLD_CSV = "data/exports/company_year_hits.csv"       # from previous step
NEW_CSV = "data/exports/company_year_summary.csv"    # your new file

def norm_company(s):
    return (s or "").strip()

def main():
    # Load previous totals (one row per company-year with integer counts)
    old = pd.read_csv(OLD_CSV, dtype={"year": "Int64"})
    old.rename(columns={"company":"Company Name", "year":"Year"}, inplace=True)
    old["Company Name"] = old["Company Name"].apply(norm_company)

    # Collapse to company-year just in case (summing counts)
    sdg_cols_old  = [f"SDG{i}" for i in range(1, 18)]
    tech_cols_old = ["AI_ML", "Cloud_Computing", "Big_Data_Blockchain", "Applications_Practice"]
    keep_old = ["Company Name", "Year"] + sdg_cols_old + tech_cols_old
    old = old[keep_old].groupby(["Company Name", "Year"], as_index=False).sum(numeric_only=True)

    # Load new summary with Substantive/Symbolic split
    new = pd.read_csv(NEW_CSV, dtype={"Year": "Int64"})
    new["Company Name"] = new["Company Name"].apply(norm_company)

    # Build totals from Substantive + Symbolic for SDGs
    for i in range(1, 18):
        sub = f"SDG {i} Substantive"
        sym = f"SDG {i} Symbolic"
        tot = f"SDG{i}_TOTAL"
        new[tot] = new.get(sub, 0).fillna(0).astype(int) + new.get(sym, 0).fillna(0).astype(int)

    # Build totals from Substantive + Symbolic for TECH buckets
    tech_map = {
        "AI_ML_TOTAL": ("AI-ML Substantive", "AI-ML Symbolic"),
        "Cloud_Computing_TOTAL": ("Cloud Computing Substantive", "Cloud Computing Symbolic"),
        "Big_Data_Blockchain_TOTAL": ("Big Data/Blockchain Substantive", "Big Data/Blockchain Symbolic"),
        "Applications_Practice_TOTAL": ("Applications Practice Substantive", "Applications Practice Symbolic"),
    }
    for out_col, (sub, sym) in tech_map.items():
        new[out_col] = new.get(sub, 0).fillna(0).astype(int) + new.get(sym, 0).fillna(0).astype(int)

    # Collapse new to company-year (sum across languages if present)
    total_cols_new = [f"SDG{i}_TOTAL" for i in range(1, 18)] + list(tech_map.keys())
    new_tot = new.groupby(["Company Name", "Year"], as_index=False)[total_cols_new].sum(numeric_only=True)

    # Merge old vs new totals
    merged = old.merge(new_tot, on=["Company Name", "Year"], how="outer", indicator=False).fillna(0)

    # Build comparison booleans for each bucket
    checks = []
    for i in range(1, 18):
        checks.append(merged[f"SDG{i}"].astype(int) == merged[f"SDG{i}_TOTAL"].astype(int))
    checks.append(merged["AI_ML"].astype(int) == merged["AI_ML_TOTAL"].astype(int))
    checks.append(merged["Cloud_Computing"].astype(int) == merged["Cloud_Computing_TOTAL"].astype(int))
    checks.append(merged["Big_Data_Blockchain"].astype(int) == merged["Big_Data_Blockchain_TOTAL"].astype(int))
    checks.append(merged["Applications_Practice"].astype(int) == merged["Applications_Practice_TOTAL"].astype(int))

    # Any mismatch per row?
    import numpy as np
    all_ok = np.logical_and.reduce(checks)
    wrong_count = int((~all_ok).sum())

    # Print ONLY the number
    print(wrong_count)

if __name__ == "__main__":
    main()
