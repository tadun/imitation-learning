import sys, csv
from pathlib import Path
import matplotlib.pyplot as plt

def load_xy(path: Path):
    xs, ys = [], []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                xs.append(float(row["x_pf"])); ys.append(float(row["y_pf"]))
            except: pass
    return xs, ys

if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else max(Path("logs").glob("perception_*.csv"))
    x, y = load_xy(p)
    plt.figure()
    plt.plot(x, y)
    plt.gca().set_aspect('equal', 'box')
    plt.title(f"PF trajectory: {p.name}")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.show()
