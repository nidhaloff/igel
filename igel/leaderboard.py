import os
import pandas as pd

def generate_leaderboard(results_dir="results", output_csv="leaderboard.csv", output_html="leaderboard.html"):
    rows = []
    for fname in os.listdir(results_dir):
        if fname.endswith(".json"):
            data = pd.read_json(os.path.join(results_dir, fname), typ='series')
            rows.append(data)
        elif fname.endswith(".csv"):
            data = pd.read_csv(os.path.join(results_dir, fname)).iloc[0]
            rows.append(data)
    if not rows:
        print("No result files found.")
        return
    df = pd.DataFrame(rows)
    df = df.sort_values(by=df.columns[1], ascending=False)  # Sort by first metric column
    df.to_csv(output_csv, index=False)
    df.to_html(output_html, index=False)
    print(f"Leaderboard saved to {output_csv} and {output_html}")
    print(df)
