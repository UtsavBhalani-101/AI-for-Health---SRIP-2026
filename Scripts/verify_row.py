"""
Verify whether a row from AP02.csv matches a window from the raw Flow .txt file.
Format: 'DD.MM.YYYY HH:MM:SS,mmm; value'
"""
import pandas as pd
import numpy as np
import glob
import os

# --- Load the CSV output row
csv_path = r"D:\IIT Gandhinagar\AI for health\srip\internship\Dataset\AP05.csv"
df = pd.read_csv(csv_path)

ROW_IDX = 1  # 0-indexed row to check (row 2 in the file = the pasted values)

row = df.iloc[ROW_IDX]
signal_cols = [c for c in df.columns if c not in ("participant_id", "label")]
csv_values = row[signal_cols].values.astype(int)

print(f"CSV row {ROW_IDX}: label={int(row['label'])}, window length={len(csv_values)}")
print(f"First 10 values: {csv_values[:10].tolist()}")

# --- Load the raw flow file (format: timestamp; value)
data_dir = r"D:\IIT Gandhinagar\AI for health\srip\internship\Data\AP05"
flow_files = glob.glob(os.path.join(data_dir, "Flow*.txt"))
flow_file = [f for f in flow_files if "Event" not in f][0]
print(f"\nFlow file: {os.path.basename(flow_file)}")

raw_values = []
in_data = False
with open(flow_file, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if line == "Data:":
            in_data = True
            continue
        if in_data and ";" in line:
            val_str = line.split(";", 1)[1].strip()
            try:
                raw_values.append(int(float(val_str)))
            except ValueError:
                continue

raw_signal = np.array(raw_values)
print(f"Raw flow signal length: {len(raw_signal)} samples")

# --- Search for the matching window
window_len = len(csv_values)
print(f"\nSearching for exact matching window of length {window_len}...")

found = False
for start in range(0, len(raw_signal) - window_len + 1):
    window = raw_signal[start:start + window_len]
    if np.array_equal(window, csv_values):
        time_sec_start = start / 32
        time_sec_end = (start + window_len) / 32
        print(f"\n✅ EXACT MATCH FOUND!")
        print(f"   Raw signal index : {start} → {start + window_len - 1}")
        print(f"   Approx. time     : {time_sec_start:.2f}s → {time_sec_end:.2f}s from recording start")
        print(f"   (@ 32 Hz sample rate, window = {window_len/32:.1f} seconds)")
        found = True
        break

if not found:
    print("❌ No exact match found. Searching for closest partial match...")
    first10 = csv_values[:10].tolist()
    for start in range(len(raw_signal) - window_len):
        if raw_signal[start:start+10].tolist() == first10:
            candidate = raw_signal[start:start+window_len]
            diff_count = int(np.sum(candidate != csv_values))
            print(f"\nPartial match at index {start}, mismatches: {diff_count}/{window_len}")
            if diff_count < 30:
                diffs = np.where(candidate != csv_values)[0]
                print("First differing positions:")
                for d in diffs[:10]:
                    print(f"  pos {d}: raw={candidate[d]}, csv={csv_values[d]}")
