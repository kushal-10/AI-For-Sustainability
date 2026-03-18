import pandas as pd
import json
import re

FILE = "sample_1000.csv"
START = 4
END = 101

# 501 - 600

df = pd.read_csv(FILE)

LABEL_COL = "hit_labels"

if LABEL_COL not in df.columns:
    df[LABEL_COL] = ""

sdg_cols = [c for c in df.columns if c.startswith("hits_sdg")]

# terminal color
RED = "\033[91m"
RESET = "\033[0m"

def highlight_text(text, patterns):
    """Highlight regex matches inside passage"""
    highlighted = text

    for pat in patterns:
        try:
            regex = re.compile(pat, re.IGNORECASE)
            highlighted = regex.sub(lambda m: f"{RED}{m.group(0)}{RESET}", highlighted)
        except re.error:
            pass  # ignore invalid regex

    return highlighted


def get_unfinished_index():
    subset = df.loc[START:END]
    for idx,row in subset.iterrows():
        if row[LABEL_COL] == "" or pd.isna(row[LABEL_COL]):
            return idx
    return None


i = get_unfinished_index()

if i is None:
    print("All rows labeled.")
    exit()


while i is not None and i <= END:

    row = df.loc[i]

    print("\n" + "="*80)
    print(f"ROW {i}")
    print("="*80)

    # collect patterns first
    all_patterns = []
    parsed_hits = {}

    for col in sdg_cols:
        val = row[col]
        if pd.notna(val) and val != "{}":
            try:
                parsed = json.loads(val)
                parsed_hits[col] = parsed
                all_patterns.extend(parsed.keys())
            except:
                pass

    # print highlighted passage
    print("\nPASSAGE:\n")
    print(highlight_text(row["passage"], all_patterns))

    if not parsed_hits:
        print("\nNo SDG hits.")

    results = {}

    # annotate hits
    for col,patterns in parsed_hits.items():

        print(f"\n--- {col} ---")

        for pattern,label in patterns.items():

            while True:
                ans = input(f"{pattern} ({label}) → yes/no/skip/quit: ").lower().strip()

                if ans == "quit":
                    df.to_csv(FILE, index=False)
                    print("Saved.")
                    exit()

                if ans in ["yes","no","skip"]:
                    break

                print("Invalid.")

            if ans != "skip":
                results[pattern] = ans

    # save row results
    df.at[i, LABEL_COL] = json.dumps(results)
    df.to_csv(FILE, index=False)

    i += 1
    i = get_unfinished_index()

print("Done.")
