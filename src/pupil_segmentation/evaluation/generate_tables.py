"""Aggregate benchmark results into comparison tables.

Scans a results directory for benchmark JSON files and produces:
- CSV comparison table
- Markdown table (for README / reports)
- Optional LaTeX table

Usage:
    uv run python -m src.pupil_segmentation.evaluation.generate_tables \
        --results_dir results --output_dir tables
"""

import argparse
import json
import sys
from pathlib import Path

# Columns to include in the comparison table, in order.
# Each entry is (display_name, json_key, format_spec).
TABLE_COLUMNS = [
    ("Experiment", "_experiment", ""),
    ("Pupil IoU", "iou_pupil", ".4f"),
    ("Iris IoU", "iou_iris", ".4f"),
    ("Mean IoU", "mean_iou", ".4f"),
    ("Pupil Axis Err (px)", "pupil_axis_error", ".2f"),
    ("Iris Axis Err (px)", "iris_axis_error", ".2f"),
    ("P/I Ratio MAE", "pupil_iris_ratio_mae", ".4f"),
    ("FPS", "fps", ".1f"),
    ("Size (MB)", "model_size_mb", ".2f"),
]


def load_results(results_dir: Path) -> list[dict]:
    """Load all benchmark JSON files from a directory."""
    results = []
    json_files = sorted(results_dir.glob("*.json"))
    for path in json_files:
        # Skip non-benchmark files (robustness, summaries, etc.)
        with open(path) as f:
            data = json.load(f)
        # Benchmark JSONs have "mean_iou" at the top level
        if "mean_iou" not in data:
            continue
        data["_experiment"] = path.stem
        results.append(data)
    return results


def format_value(value, fmt: str) -> str:
    """Format a value with the given format spec."""
    if not fmt:
        return str(value)
    try:
        return f"{value:{fmt}}"
    except (TypeError, ValueError):
        return str(value)


def to_csv(results: list[dict], output_path: Path) -> None:
    """Write results as CSV."""
    headers = [col[0] for col in TABLE_COLUMNS]
    with open(output_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in results:
            values = [format_value(row.get(col[1], ""), col[2]) for col in TABLE_COLUMNS]
            f.write(",".join(values) + "\n")


def to_markdown(results: list[dict]) -> str:
    """Generate a markdown table string."""
    headers = [col[0] for col in TABLE_COLUMNS]
    rows = []
    for row in results:
        rows.append([format_value(row.get(col[1], ""), col[2]) for col in TABLE_COLUMNS])

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    # Build table
    lines = []
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |")

    return "\n".join(lines)


def to_latex(results: list[dict]) -> str:
    """Generate a LaTeX tabular string."""
    headers = [col[0] for col in TABLE_COLUMNS]
    n_cols = len(headers)

    lines = [
        r"\begin{tabular}{" + "l" + "r" * (n_cols - 1) + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in results:
        values = [format_value(row.get(col[1], ""), col[2]) for col in TABLE_COLUMNS]
        lines.append(" & ".join(values) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results into tables")
    parser.add_argument(
        "--results_dir", type=Path, default=Path("results"), help="Directory with benchmark JSONs"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("tables"), help="Output directory for tables"
    )
    parser.add_argument("--latex", action="store_true", help="Also generate LaTeX table")
    parser.add_argument(
        "--sort_by",
        type=str,
        default="iou_pupil",
        help="Sort results by this metric (descending)",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    results = load_results(args.results_dir)
    if not results:
        print(f"No benchmark JSON files found in {args.results_dir}")
        sys.exit(1)

    # Sort by the chosen metric (descending = best first)
    results.sort(key=lambda r: r.get(args.sort_by, 0), reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = args.output_dir / "comparison.csv"
    to_csv(results, csv_path)
    print(f"CSV:      {csv_path}")

    # Markdown
    md_table = to_markdown(results)
    md_path = args.output_dir / "comparison.md"
    md_path.write_text(md_table + "\n")
    print(f"Markdown: {md_path}")

    # Print to stdout
    print(f"\n{md_table}\n")

    # LaTeX (optional)
    if args.latex:
        latex_table = to_latex(results)
        tex_path = args.output_dir / "comparison.tex"
        tex_path.write_text(latex_table + "\n")
        print(f"LaTeX:    {tex_path}")

    print(f"\n{len(results)} experiments compared, sorted by {args.sort_by} (descending)")


if __name__ == "__main__":
    main()
