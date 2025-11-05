import matplotlib.pyplot as plt
import argparse
import re

def parse_and_plot(filename, output_file):
    x_vals = []  # record index
    y_vals = []  # differences
    
    count = 0
    pattern = re.compile(r"prox_score_before prox_score_after\s+(\d+),(\d+):")
    check_str = "queued_paths + unique_crashes > temp_total_found"
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        match = pattern.search(line)
        if match and i + 2 < len(lines):
            next_line = lines[i + 2]
            if check_str in next_line:
                before = int(match.group(1))
                after = int(match.group(2))
                if before != after:
                    count += 1
                    diff = after - before
                    x_vals.append(count)
                    y_vals.append(diff)
    
    if not x_vals:
        print("No valid records found.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-')
    plt.xlabel('Record Index')
    plt.ylabel('After - Before')
    plt.title('Difference of prox_score_after - prox_score_before (filtered)')
    plt.grid(True)
    
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot filtered prox_score differences from log file.")
    parser.add_argument("filename", help="Path to the log file")
    parser.add_argument("-o", "--output", required=True, help="Output PNG file to save the plot")
    args = parser.parse_args()
    
    parse_and_plot(args.filename, args.output)
