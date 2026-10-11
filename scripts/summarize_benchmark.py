"""Create a compact Markdown report and a F1/FPS comparison chart."""

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/benchmark.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("results/demo_benchmark.md"))
    parser.add_argument("--plot", type=Path, default=Path("results/benchmark.png"))
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Benchmark CSV is empty")
    frame_count = int(float(rows[0].get("frames") or 0))
    lines = ["# Benchmark summary", "", f"Input table: `{args.input}`", "",
             f"Results use a synthetic {frame_count}-frame, 320x240 clip. They validate the pipeline and are not a claim about surveillance accuracy.", "",
             "| Method | Pixel F1 | Pixel IoU | Frame F1 | Processing FPS | Events |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        def show(field: str, digits: int = 3) -> str:
            value = row.get(field, "")
            return f"{float(value):.{digits}f}" if value else "—"
        lines.append(f"| {row['method']} | {show('pixel_f1')} | {show('pixel_iou')} | {show('frame_f1')} | {show('fps', 1)} | {show('events', 0)} |")
    lines += ["", "A crossing event is generated from a confirmed track crossing the configured line. The synthetic foreground is deliberately bright and simple. For CDnet or camera footage, replace the demo labels and tune the ROI, area, learning rate and shadow settings.", ""]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    if not args.skip_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
        colors = {"mog2": "#2471a3", "knn": "#17a589", "diff": "#d68910", "running_average": "#884ea0"}
        for row in rows:
            f1, fps = float(row.get("pixel_f1") or 0), float(row.get("fps") or 0)
            ax.scatter(fps, f1, s=75, color=colors.get(row["method"], "#555555"), label=row["method"])
        ax.legend(title="Method", frameon=False)
        ax.set(xlabel="Processing FPS", ylabel="Aggregate pixel F1", title="Background model comparison on demo clip")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot)
        plt.close(fig)
    print(f"Wrote {args.markdown}" + (f" and {args.plot}" if not args.skip_plot else ""))


if __name__ == "__main__":
    main()
