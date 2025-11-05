import matplotlib.pyplot as plt
import argparse

def parse_and_plot(filename, output_file=None):
    x_vals = []  # time values (seconds)
    y_vals = []  # hit count values
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        print("File is empty.")
        return
    
    # First line values
    first_line = lines[0].strip().split()
    if len(first_line) < 3:
        print("Invalid file format in the first line.")
        return
    
    base_time = int(first_line[2])
    prev_y, prev_x = None, None

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        
        y = int(parts[1])
        x = int(parts[2]) - base_time  # relative time in seconds
        
        # Skip duplicates (same x and y as previous)
        if prev_y == y and prev_x == x:
            continue
        
        x_vals.append(x)
        y_vals.append(y)
        prev_y, prev_x = y, x
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Hit Count')
    plt.title('Hit Count over Time')
    plt.grid(True)

    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot hit count over time from log file.")
    parser.add_argument("filename", help="Path to the log file")
    parser.add_argument("-o", "--output", help="Output PNG file to save the plot")
    args = parser.parse_args()
    
    parse_and_plot(args.filename, args.output)
