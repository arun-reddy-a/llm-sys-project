import csv
import sys
import os

def analyze(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return

    metrics = {}
    tokens = [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
    ]

    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            # Find the header row (starts after 'ID', 'Process', etc.)
            for row in reader:
                if not row: continue
                if row[0] == "ID": # This is the header of the metrics
                    header = row
                    continue
                # Assuming simple_vadd is the first/only kernel
                if "simple_vadd" in row[4]: # Kernel Name usually at index 4 or similar
                    for i, h in enumerate(header):
                        if h in tokens or any(t in h for t in tokens):
                            metrics[h] = row[i]
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return

    print("\n🔍 Analysis for simple_vadd on Blackwell B200:")
    print("-" * 45)
    for k, v in metrics.items():
        print(f"{k:40} : {v:>8}%")
    print("-" * 45)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        print("Usage: python analyze_ncu.py <filename.csv>")
