import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple, Any, Optional, Callable
import math
import re

import matplotlib.pyplot as plt


DEFAULT_PSNRCOL = "psnr_total"
DEFAULT_CODEBOOK_COL = "codebook_usage"
DEFAULT_CODEBOOK_FILENAME = "codebook_usage.csv"


def _to_number(value: Any):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def read_csv_columns(path: str) -> Dict[str, List[Any]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                columns[name].append(_to_number(row.get(name)))
    if not any(columns.values()):
        raise ValueError(f"CSV has no rows: {path}")
    return columns


def simplify_label(label: str) -> str:
    name = re.sub(r"^run[_-]*", "", label, flags=re.IGNORECASE)
    lower = name.lower()

    codebook_match = re.search(r"(?:codebook[_-]?|c)(\d+)", lower)
    if codebook_match:
        return f"Codebook Size {codebook_match.group(1)}"

    latent_size_match = re.search(r"latent[_-]?size[_-]?(?:only[_-]?)?(\d+)", lower)
    if latent_size_match:
        return f"Latent size {latent_size_match.group(1)}"

    latent_res_match = re.search(r"latent[_-]?resolution[_-]?(\d+x\d+)", lower)
    if latent_res_match:
        return f"Latent resolution {latent_res_match.group(1)}"

    tokens = re.split(r"[_-]+", name)
    drop = {
        "run",
        "vq",
        "vqinr",
        "relu",
        "siren",
        "set",
        "newset",
        "codebook",
    }
    cleaned = [t for t in tokens if t and t.lower() not in drop]
    if cleaned:
        return " ".join(cleaned)
    return name


def default_label_from_path(path: str, style: str) -> str:
    base = os.path.basename(path)
    if base in {"psnr_history.csv", "codebook_usage.csv", "training_metrics.csv"}:
        parent = os.path.basename(os.path.dirname(path))
        label = parent or base
    else:
        label = os.path.splitext(base)[0]
    if style == "simple":
        return simplify_label(label)
    return label


def select_psnr_column(columns: Dict[str, List[Any]], preferred: Optional[str]) -> str:
    if preferred:
        if preferred not in columns:
            raise ValueError(
                f"PSNR column '{preferred}' not found. Available: {list(columns.keys())}"
            )
        return preferred
    if DEFAULT_PSNRCOL in columns:
        return DEFAULT_PSNRCOL
    psnr_cols = [c for c in columns if c.startswith("psnr")]
    if len(psnr_cols) == 1:
        return psnr_cols[0]
    if psnr_cols:
        return psnr_cols[0]
    raise ValueError(
        f"No PSNR column found. Available: {list(columns.keys())}"
    )


def select_codebook_column(columns: Dict[str, List[Any]], preferred: Optional[str]) -> str:
    if preferred:
        if preferred not in columns:
            raise ValueError(
                f"Codebook column '{preferred}' not found. Available: {list(columns.keys())}"
            )
        return preferred
    if DEFAULT_CODEBOOK_COL in columns:
        return DEFAULT_CODEBOOK_COL
    candidates = [c for c in columns if "codebook" in c or "usage" in c]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return candidates[0]
    raise ValueError(
        f"No codebook usage column found. Available: {list(columns.keys())}"
    )


def column_values(columns: Dict[str, List[Any]], column: str) -> List[float]:
    return [float(v) for v in columns.get(column, []) if v is not None]


def psnr_values(
    columns: Dict[str, List[Any]],
    preferred: Optional[str],
    use_average: bool,
) -> List[float]:
    if not use_average:
        col = select_psnr_column(columns, preferred)
        return column_values(columns, col)
    psnr_cols = [c for c in columns if c.startswith("psnr")]
    if not psnr_cols:
        raise ValueError(
            f"No PSNR column found. Available: {list(columns.keys())}"
        )
    if "psnr_total" in psnr_cols and len(psnr_cols) > 1:
        psnr_cols = [c for c in psnr_cols if c != "psnr_total"]
    max_len = max(len(columns[c]) for c in psnr_cols)
    y_vals: List[float] = []
    for i in range(max_len):
        vals = []
        for col in psnr_cols:
            if i < len(columns[col]):
                v = columns[col][i]
                if v is not None:
                    vals.append(float(v))
        if vals:
            y_vals.append(sum(vals) / len(vals))
    return y_vals


def codebook_values(
    columns: Dict[str, List[Any]],
    preferred: Optional[str],
) -> List[float]:
    col = select_codebook_column(columns, preferred)
    return column_values(columns, col)


def sample_xy(
    x_vals: List[float],
    y_vals: List[float],
    sample_every: int,
    max_points: int,
) -> Tuple[List[float], List[float]]:
    if not x_vals or not y_vals:
        return x_vals, y_vals
    step = max(1, sample_every)
    if max_points and max_points > 0 and len(x_vals) > max_points:
        step = max(step, math.ceil(len(x_vals) / max_points))
    if step <= 1:
        return x_vals, y_vals
    out_x: List[float] = []
    out_y: List[float] = []
    for i in range(0, len(x_vals), step):
        x_chunk = x_vals[i : i + step]
        y_chunk = y_vals[i : i + step]
        if not x_chunk or not y_chunk:
            continue
        out_x.append(sum(x_chunk) / len(x_chunk))
        out_y.append(sum(y_chunk) / len(y_chunk))
    return out_x, out_y


def build_series(
    files: List[str],
    labels: Optional[List[str]],
    y_selector: Callable[[Dict[str, List[Any]]], List[float]],
    sample_every: int,
    max_points: int,
    label_style: str,
) -> List[Tuple[List[float], List[float], str]]:
    if labels and len(labels) != len(files):
        raise ValueError("Label count must match file count")
    series = []
    for idx, path in enumerate(files):
        columns = read_csv_columns(path)
        y_vals = y_selector(columns)
        x = columns.get("epoch")
        if not x or all(v is None for v in x):
            x = list(range(1, len(y_vals) + 1))
        else:
            x = [float(v) for v in x if v is not None]
        label = labels[idx] if labels else default_label_from_path(path, label_style)
        min_len = min(len(x), len(y_vals))
        x_vals = x[:min_len]
        y_vals = y_vals[:min_len]
        x_vals, y_vals = sample_xy(x_vals, y_vals, sample_every, max_points)
        series.append((x_vals, y_vals, label))
    return series


def plot_series(
    series: List[Tuple[List[float], List[float], str]],
    title: str,
    ylabel: str,
    output_path: str,
):
    if not series:
        return
    def sort_key_label(label: str):
        match = re.search(r"(\d+)", label)
        return int(match.group(1)) if match else float("inf")

    series = sorted(series, key=lambda item: sort_key_label(item[2]))
    plt.figure(figsize=(9, 5))
    for x_vals, y_vals, label in series:
        plt.plot(x_vals, y_vals, label=label, linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        ordered = sorted(zip(handles, labels), key=lambda hl: sort_key_label(hl[1]))
        handles, labels = zip(*ordered)
        plt.legend(handles, labels)
    else:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def find_codebook_files(root: str, filename: str) -> List[str]:
    matches = []
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            matches.append(os.path.join(dirpath, filename))
    return sorted(matches)


def dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Plot PSNR curves and codebook utilization from CSV files."
    )

    parser.add_argument("--psnr-files", nargs="*", default=[], help="PSNR CSV files")
    parser.add_argument("--psnr-labels", nargs="*", help="Labels for PSNR curves")
    parser.add_argument(
        "--psnr-column",
        default=None,
        help=f"PSNR column name (default: {DEFAULT_PSNRCOL})",
    )
    parser.add_argument(
        "--psnr-avg",
        action="store_true",
        help="Average all psnr_* columns per epoch (exclude psnr_total if present)",
    )

    parser.add_argument(
        "--codebook-files", nargs="*", default=[], help="Codebook CSV files"
    )
    parser.add_argument(
        "--codebook-dir",
        default=None,
        help="Directory to scan for codebook usage CSV files",
    )
    parser.add_argument(
        "--codebook-filename",
        default=DEFAULT_CODEBOOK_FILENAME,
        help="Filename to match when scanning for codebook files",
    )
    parser.add_argument(
        "--codebook-labels", nargs="*", help="Labels for codebook curves"
    )
    parser.add_argument(
        "--codebook-column",
        default=None,
        help=f"Codebook usage column name (default: {DEFAULT_CODEBOOK_COL})",
    )

    parser.add_argument("--out-dir", default="res", help="Output directory")
    parser.add_argument("--psnr-out", default=None, help="PSNR output filename")
    parser.add_argument(
        "--codebook-out", default=None, help="Codebook output filename"
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Average every N epochs into one point (applies to both plots)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=300,
        help="Cap the number of plotted points (0 disables)",
    )
    parser.add_argument(
        "--label-style",
        choices=["simple", "raw"],
        default="simple",
        help="Legend label style for auto-generated labels",
    )

    args = parser.parse_args()

    psnr_files = list(args.psnr_files)
    codebook_files = list(args.codebook_files)

    if args.codebook_dir:
        codebook_files.extend(
            find_codebook_files(args.codebook_dir, args.codebook_filename)
        )

    psnr_files = dedupe(psnr_files)
    codebook_files = dedupe(codebook_files)

    if not psnr_files and not codebook_files:
        parser.print_help(sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    if psnr_files:
        series = build_series(
            psnr_files,
            args.psnr_labels,
            lambda cols: psnr_values(cols, args.psnr_column, args.psnr_avg),
            args.sample_every,
            args.max_points,
            args.label_style,
        )
        psnr_out = args.psnr_out or os.path.join(args.out_dir, "psnr_compare.png")
        psnr_title = "PSNR Curves"
        psnr_ylabel = "Mean PSNR (dB)" if args.psnr_avg else "PSNR (dB)"
        plot_series(series, psnr_title, psnr_ylabel, psnr_out)

    if codebook_files:
        series = build_series(
            codebook_files,
            args.codebook_labels,
            lambda cols: codebook_values(cols, args.codebook_column),
            args.sample_every,
            args.max_points,
            args.label_style,
        )
        codebook_out = args.codebook_out or os.path.join(
            args.out_dir, "codebook_compare.png"
        )
        plot_series(
            series,
            "Codebook Utilization",
            "Active Codes (%)",
            codebook_out,
        )


if __name__ == "__main__":
    main()
