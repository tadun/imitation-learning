import csv, sys, statistics as stats
from pathlib import Path

def eval_csv(p: Path):
    vis_flags, bearings = [], []
    with p.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            vis_flags.append(int(row["visible"]))
            try:
                bearings.append(float(row["bearing_rad"]))
            except: pass
    n = max(1, len(vis_flags))
    return sum(vis_flags)/n, (stats.pstdev(bearings) if len(bearings) > 1 else 0.0)

if __name__ == "__main__":
    logs = Path("logs")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else max(logs.glob("perception_*.csv"))
    vr, jit = eval_csv(path)
    print(f"{path.name}: visible={vr*100:.1f}%  bearing_jitter={jit:.3f} rad")
