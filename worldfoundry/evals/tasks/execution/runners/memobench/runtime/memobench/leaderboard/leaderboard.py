import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Automated metrics (0-100 composite scores from run_eval.py)
AUTOMATED_METRICS = [
    "VisualQuality",
    "MotionSmoothness",
    "ObjIdentityConsistency",
    "Geo3DConsistency",
    "CameraControllability",
    "ImageRewardScore",
]

# Pixel-fidelity metrics (raw scale, NOT 0-100)
FIDELITY_METRICS = [
    "GT_ALL_PSNR",
    "GT_ALL_SSIM",
    "GT_ALL_LPIPS",
]

# ORS (0-1 raw → displayed as 0-100)
ORS_METRIC = "ORS"

# VQA dimensions (0-100)
VQA_METRICS = [
    "InstructionFollowing",
    "ObjectBackground",
    "ContinuityOfMemory",
    "PhysicsAdherence",
]

# Higher is better for all except LPIPS
LOWER_IS_BETTER = {"GT_ALL_LPIPS"}

# Display order
DISPLAY_ORDER = [*AUTOMATED_METRICS, ORS_METRIC, *FIDELITY_METRICS, *VQA_METRICS]

# Pretty names for display
PRETTY_NAMES = {
    "VisualQuality": "VisQual",
    "MotionSmoothness": "MotSmooth",
    "ObjIdentityConsistency": "ObjConsist",
    "Geo3DConsistency": "3DConsist",
    "CameraControllability": "CamCtrl",
    "ImageRewardScore": "ImgReward",
    "ORS": "ORS",
    "GT_ALL_PSNR": "PSNR",
    "GT_ALL_SSIM": "SSIM",
    "GT_ALL_LPIPS": "LPIPS",
    "InstructionFollowing": "VQA-IF",
    "ObjectBackground": "VQA-OB",
    "ContinuityOfMemory": "VQA-CM",
    "PhysicsAdherence": "VQA-PA",
}

MODEL_DISPLAY_NAMES = {
    "lingbot-world": "LingBot-World",
    "wan2.2": "Wan2.2",
    "fantasy-world": "FantasyWorld",
    "matrix-game2": "Matrix-Game2",
    "stable-virtual-camera": "SVC",
    "open-sora": "Open-SoRA",
    "ltx-video": "LTX-Video",
    "cogvideo": "CogVideoX",
}

# Map variant directory names to a canonical key (lowercase)
# so that ORS dirs like "LingBot-World" merge with eval dirs like "lingbot-world"
_CANONICAL_KEY = {
    "lingbot-world": "lingbot-world",
    "LingBot-World": "lingbot-world",
    "Lingbot": "lingbot-world",
    "wan2.2": "wan2.2",
    "Wan2.2": "wan2.2",
    "fantasy-world": "fantasy-world",
    "FantasyWorld": "fantasy-world",
    "matrix-game2": "matrix-game2",
    "Matrix-Game2": "matrix-game2",
    "stable-virtual-camera": "stable-virtual-camera",
    "StableVirtualCamera": "stable-virtual-camera",
    "open-sora": "open-sora",
    "Open-SoRA": "open-sora",
    "ltx-video": "ltx-video",
    "LTX-Video": "ltx-video",
    "cogvideo": "cogvideo",
    "CogVideoX": "cogvideo",
}


def _canonicalize(name: str) -> str:
    return _CANONICAL_KEY.get(name, name.lower())


def _latest_csv(directory: str, pattern: str = "eval_*.csv") -> str | None:
    """Return the most recently modified CSV matching pattern."""
    csvs = sorted(glob.glob(os.path.join(directory, pattern)), key=os.path.getmtime)
    return csvs[-1] if csvs else None


def load_eval_results(eval_dir: str) -> dict[str, pd.DataFrame]:
    """Load per-model eval CSVs from eval_dir/{model}/eval_*.csv."""
    results = {}
    for model_dir in sorted(Path(eval_dir).iterdir()):
        if not model_dir.is_dir():
            continue
        csv_path = _latest_csv(str(model_dir))
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path)
        key = _canonicalize(model_dir.name)
        results[key] = df
    return results


def load_ors_results(ors_dir: str) -> dict[str, pd.DataFrame]:
    """Load per-model ORS CSVs from ors_dir/{model}/ors_scores.csv."""
    results = {}
    for model_dir in sorted(Path(ors_dir).iterdir()):
        if not model_dir.is_dir():
            continue
        csv_path = os.path.join(str(model_dir), "ors_scores.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        key = _canonicalize(model_dir.name)
        results[key] = df
    return results


def load_vqa_results(vqa_dir: str) -> dict[str, pd.DataFrame]:
    """Load per-model VQA scores from vqa_dir/{model}/*.csv.

    Each CSV has columns: scene, video_id, Evaluation, score
    where `score` is a JSON dict like:
      {"Instruction Following": 0.67, "Object and Background": 0.33,
       "Continuity of Memory": 0.60, "Physics Adherence": 0.67}
    Scores are 0-1 fractions; we convert to 0-100.
    Directories ending with '_old' are skipped.
    """
    # Map from VQA JSON keys to our internal metric names
    VQA_KEY_MAP = {
        "Instruction Following": "InstructionFollowing",
        "Object and Background": "ObjectBackground",
        "Continuity of Memory": "ContinuityOfMemory",
        "Physics Adherence": "PhysicsAdherence",
    }

    results = {}
    for model_dir in sorted(Path(vqa_dir).iterdir()):
        if not model_dir.is_dir():
            continue
        # Skip _old directories and stale aliases
        if model_dir.name.endswith("_old") or model_dir.name == "Lingbot":
            continue
        key = _canonicalize(model_dir.name)

        clip_csvs = sorted(model_dir.glob("*.csv"))
        if not clip_csvs:
            continue

        rows = []
        for csv_path in clip_csvs:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if "score" not in df.columns:
                continue
            for _, r in df.iterrows():
                try:
                    scores = json.loads(r["score"])
                except (json.JSONDecodeError, TypeError):
                    scores = r["score"]
                    if isinstance(scores, str):
                        continue
                row = {"scene": r.get("scene", ""), "video_id": r.get("video_id", "")}
                for vqa_key, metric_name in VQA_KEY_MAP.items():
                    if vqa_key in scores:
                        row[metric_name] = float(scores[vqa_key]) * 100  # 0-1 → 0-100
                    else:
                        row[metric_name] = np.nan
                rows.append(row)

        if rows:
            results[key] = pd.DataFrame(rows)
    return results


def aggregate_model(
    model_name: str,
    eval_df: pd.DataFrame | None = None,
    ors_df: pd.DataFrame | None = None,
    vqa_df: pd.DataFrame | None = None,
) -> dict:
    """Compute mean scores for one model across all clips."""
    row = {"Model": MODEL_DISPLAY_NAMES.get(model_name, model_name)}

    # Automated metrics (already 0-100 composite scores)
    if eval_df is not None:
        for m in AUTOMATED_METRICS:
            # ImageReward is stored raw (can be negative); prefer the 0-100 _pct
            # column so the displayed value is on the same scale as the paper table.
            if m == "ImageRewardScore" and "ImageRewardScore_pct" in eval_df.columns:
                row[m] = eval_df["ImageRewardScore_pct"].mean(skipna=True)
                continue
            if m in eval_df.columns:
                val = eval_df[m].mean(skipna=True)
                # CameraControllability: camfix files store as 0-1, others as 0-100
                if m == "CameraControllability" and val < 1:
                    val *= 100
                row[m] = val
            else:
                row[m] = np.nan

        # Pixel fidelity (raw scale)
        for m in FIDELITY_METRICS:
            if m in eval_df.columns:
                row[m] = eval_df[m].mean(skipna=True)
            else:
                row[m] = np.nan

    # ORS (raw 0-1 → display 0-100)
    if ors_df is not None and "ors" in ors_df.columns:
        row[ORS_METRIC] = ors_df["ors"].mean(skipna=True) * 100
    else:
        row[ORS_METRIC] = np.nan

    # VQA dimensions (already 0-100)
    if vqa_df is not None:
        for m in VQA_METRICS:
            if m in vqa_df.columns:
                row[m] = vqa_df[m].mean(skipna=True)
            else:
                row[m] = np.nan
    else:
        for m in VQA_METRICS:
            row[m] = np.nan

    return row


def build_leaderboard(
    eval_dir: str | None = None,
    ors_dir: str | None = None,
    vqa_dir: str | None = None,
) -> pd.DataFrame:
    """Build the full leaderboard DataFrame."""
    eval_data = load_eval_results(eval_dir) if eval_dir else {}
    ors_data = load_ors_results(ors_dir) if ors_dir else {}
    vqa_data = load_vqa_results(vqa_dir) if vqa_dir else {}

    # Union of all model keys
    all_models = sorted(set(eval_data) | set(ors_data) | set(vqa_data))
    if not all_models:
        print("No model results found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for model in all_models:
        row = aggregate_model(
            model,
            eval_df=eval_data.get(model),
            ors_df=ors_data.get(model),
            vqa_df=vqa_data.get(model),
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    # Reorder columns
    cols = ["Model"] + [c for c in DISPLAY_ORDER if c in df.columns]
    df = df[cols]
    return df


def _bold_best(series: pd.Series, lower_is_better: bool = False) -> list[str]:
    """Return formatted strings with best bolded and second-best underlined (markdown)."""
    valid = series.dropna()
    if valid.empty:
        return ["—"] * len(series)
    sorted_unique = sorted(valid.unique(), reverse=not lower_is_better)
    best = sorted_unique[0]
    second = sorted_unique[1] if len(sorted_unique) > 1 else None
    worst = sorted_unique[-1] if len(sorted_unique) > 1 else None
    out = []
    for v in series:
        if pd.isna(v):
            out.append("—")
        elif np.isclose(v, best):
            out.append(f"**{v:.2f}**")
        elif second is not None and np.isclose(v, second):
            out.append(f"<u>{v:.2f}</u>")
        elif worst is not None and np.isclose(v, worst):
            out.append('<font color="red">' + f"{v:.2f}" + "</font>")
        else:
            out.append(f"{v:.2f}")
    return out


def format_markdown(df: pd.DataFrame) -> str:
    """Render leaderboard as a GitHub-flavored Markdown table."""
    fmt = df.copy()
    for col in fmt.columns:
        if col == "Model":
            continue
        lib = col in LOWER_IS_BETTER
        fmt[col] = _bold_best(fmt[col], lower_is_better=lib)

    # Rename columns to pretty names
    fmt = fmt.rename(columns=PRETTY_NAMES)

    lines = []
    header = "| " + " | ".join(fmt.columns) + " |"
    sep = "|" + "|".join("---:" if c != fmt.columns[0] else ":---" for c in fmt.columns) + "|"
    lines.append(header)
    lines.append(sep)
    for _, row in fmt.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def format_latex(df: pd.DataFrame) -> str:
    """Render leaderboard as a LaTeX table."""
    ncols = len(df.columns)
    col_spec = "l" + "r" * (ncols - 1)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{MemoBench Leaderboard.} Per-model mean scores. "
        r"Automated metrics and ORS are on a 0--100 scale; PSNR/SSIM/LPIPS "
        r"are in their native units. Best in \textbf{bold}.}",
        r"\label{tab:leaderboard}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{adjustbox}{width=\linewidth}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Add arrows
    arrows = []
    for c in df.columns:
        if c == "Model":
            arrows.append(r"\textbf{Model}")
        elif c in LOWER_IS_BETTER:
            arrows.append(rf"\textbf{{{PRETTY_NAMES.get(c, c)}}} $\downarrow$")
        elif c in FIDELITY_METRICS and c != "GT_ALL_LPIPS":
            arrows.append(rf"\textbf{{{PRETTY_NAMES.get(c, c)}}} $\uparrow$")
        else:
            arrows.append(rf"\textbf{{{PRETTY_NAMES.get(c, c)}}} $\uparrow$")
    lines.append(" & ".join(arrows) + r" \\")
    lines.append(r"\midrule")

    # Find best, second-best, and worst per column
    best = {}
    second = {}
    worst = {}
    for col in df.columns:
        if col == "Model":
            continue
        valid = df[col].dropna()
        if valid.empty:
            continue
        sorted_unique = sorted(valid.unique(), reverse=col not in LOWER_IS_BETTER)
        best[col] = sorted_unique[0]
        if len(sorted_unique) > 1:
            second[col] = sorted_unique[1]
            worst[col] = sorted_unique[-1]

    # Rows
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            v = row[col]
            if col == "Model":
                cells.append(str(v))
            elif pd.isna(v):
                cells.append("—")
            elif col in best and np.isclose(v, best[col]):
                cells.append(rf"\textbf{{{v:.2f}}}")
            elif col in second and np.isclose(v, second[col]):
                cells.append(rf"\underline{{{v:.2f}}}")
            elif col in worst and np.isclose(v, worst[col]):
                cells.append(rf"\textcolor{{red}}{{{v:.2f}}}")
            else:
                cells.append(f"{v:.2f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def format_csv(df: pd.DataFrame) -> str:
    """Render leaderboard as CSV."""
    return df.to_csv(index=False, float_format="%.2f")


def format_html(df: pd.DataFrame) -> str:
    """Render leaderboard as a self-contained HTML page with sortable columns."""
    # Prepare data as JSON for JavaScript
    pretty = {c: PRETTY_NAMES.get(c, c) for c in df.columns}
    columns = list(df.columns)
    pretty_columns = [pretty[c] for c in columns]
    lower_is_better_indices = [i for i, c in enumerate(columns) if c in LOWER_IS_BETTER]

    # Build rows as list of lists
    rows_data = []
    for _, row in df.iterrows():
        r = []
        for col in columns:
            v = row[col]
            if col == "Model":
                r.append(str(v))
            elif pd.isna(v):
                r.append(None)
            else:
                r.append(round(float(v), 2))
        rows_data.append(r)

    # Find best, second-best, and worst values per column
    best_vals = {}
    second_vals = {}
    worst_vals = {}
    for i, col in enumerate(columns):
        if col == "Model":
            continue
        vals = [r[i] for r in rows_data if r[i] is not None]
        if vals:
            sorted_unique = sorted(set(vals), reverse=col not in LOWER_IS_BETTER)
            best_vals[i] = sorted_unique[0]
            if len(sorted_unique) > 1:
                second_vals[i] = sorted_unique[1]
                worst_vals[i] = sorted_unique[-1]

    import json as _json

    data_json = _json.dumps(
        {
            "columns": pretty_columns,
            "columnKeys": columns,
            "rows": rows_data,
            "lowerIsBetter": lower_is_better_indices,
            "best": {str(k): v for k, v in best_vals.items()},
            "second": {str(k): v for k, v in second_vals.items()},
            "worst": {str(k): v for k, v in worst_vals.items()},
        }
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MemoBench Leaderboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f8f9fa; color: #1a1a2e; padding: 2rem;
  }}
  h1 {{ text-align: center; margin-bottom: 0.3rem; font-size: 1.8rem; }}
  .subtitle {{ text-align: center; color: #666; margin-bottom: 1.5rem; font-size: 0.95rem; }}
  .table-wrap {{
    overflow-x: auto; background: #fff; border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 0;
  }}
  table {{ border-collapse: collapse; width: 100%; min-width: 900px; }}
  th, td {{ padding: 10px 14px; text-align: right; white-space: nowrap; }}
  td:first-child {{ text-align: left; position: sticky; left: 0; background: inherit; z-index: 1; }}
  th {{
    background: #1a1a2e; color: #fff; cursor: pointer; user-select: none;
    font-weight: 600; font-size: 0.85rem; letter-spacing: 0.02em;
    position: sticky; top: 0; z-index: 2;
  }}
  th:first-child {{ text-align: left; position: sticky; left: 0; z-index: 3; background: #1a1a2e; }}
  th:hover {{ background: #2d2d5e; }}
  th .arrow {{ display: inline-block; width: 1em; text-align: center; margin-left: 4px; opacity: 0.5; }}
  th.sorted .arrow {{ opacity: 1; }}
  th.sorted {{ background: #2d2d5e; }}
  tr:nth-child(even) {{ background: #f4f6f9; }}
  tr:hover {{ background: #e8ecf3; }}
  td.best {{ color: #16a34a; font-weight: 700; }}
  td.second {{ text-decoration: underline; text-underline-offset: 3px; }}
  td.worst {{ color: #dc2626; }}
  td.na {{ color: #aaa; font-style: italic; }}
  .rank {{ color: #888; font-weight: 400; margin-right: 6px; }}
  .legend {{
    text-align: center; margin-top: 1rem; font-size: 0.82rem; color: #888;
  }}
</style>
</head>
<body>
<h1>MemoBench Leaderboard</h1>
<p class="subtitle">Click any column header to sort. Default: highest average composite score.</p>
<div class="table-wrap">
  <table id="lb"><thead><tr id="hdr"></tr></thead><tbody id="bdy"></tbody></table>
</div>
<p class="legend">
  <span style="color:#16a34a;font-weight:700;">Green bold</span> = best in column &nbsp;|&nbsp;
  <span style="text-decoration:underline;">Underline</span> = second best &nbsp;|&nbsp;
  <span style="color:#dc2626;">Red</span> = worst in column &nbsp;|&nbsp;
  All composite metrics on 0&ndash;100 scale &nbsp;|&nbsp;
  LPIPS: lower is better; all others: higher is better
</p>
<script>
const D = {data_json};
const state = {{ col: -1, asc: false }};

function avgScore(row) {{
  let sum = 0, n = 0;
  for (let i = 1; i < row.length; i++) {{
    if (row[i] === null) continue;
    let v = row[i];
    if (D.lowerIsBetter.includes(i)) v = 100 - v;
    sum += v; n++;
  }}
  return n ? sum / n : -Infinity;
}}

function render(sortCol, sortAsc) {{
  const rows = [...D.rows];
  if (sortCol <= 0) {{
    rows.sort((a, b) => sortAsc ? avgScore(a) - avgScore(b) : avgScore(b) - avgScore(a));
  }} else {{
    rows.sort((a, b) => {{
      if (a[sortCol] === null && b[sortCol] === null) return 0;
      if (a[sortCol] === null) return 1;
      if (b[sortCol] === null) return -1;
      return sortAsc ? a[sortCol] - b[sortCol] : b[sortCol] - a[sortCol];
    }});
  }}

  // Header
  const hdr = document.getElementById("hdr");
  hdr.innerHTML = "";
  D.columns.forEach((name, i) => {{
    const th = document.createElement("th");
    let arrow = "";
    if (i === 0) {{
      th.className = (sortCol <= 0) ? "sorted" : "";
      arrow = (sortCol <= 0) ? (sortAsc ? "&#9650;" : "&#9660;") : "&#9660;";
    }} else if (i === sortCol) {{
      th.className = "sorted";
      arrow = sortAsc ? "&#9650;" : "&#9660;";
    }} else {{
      arrow = D.lowerIsBetter.includes(i) ? "&#9660;" : "&#9650;";
    }}
    th.innerHTML = name + '<span class="arrow">' + arrow + '</span>';
    th.onclick = () => {{
      if (i === 0) {{
        if (state.col <= 0) {{
          state.asc = !state.asc;
        }} else {{
          state.col = -1;
          state.asc = false;
        }}
      }} else if (state.col === i) {{
        state.asc = !state.asc;
      }} else {{
        state.col = i;
        state.asc = D.lowerIsBetter.includes(i);
      }}
      render(state.col, state.asc);
    }};
    hdr.appendChild(th);
  }});

  // Body
  const bdy = document.getElementById("bdy");
  bdy.innerHTML = "";
  rows.forEach((row, rank) => {{
    const tr = document.createElement("tr");
    row.forEach((v, i) => {{
      const td = document.createElement("td");
      if (i === 0) {{
        td.innerHTML = '<span class="rank">' + (rank + 1) + '.</span>' + v;
      }} else if (v === null) {{
        td.textContent = "—";
        td.className = "na";
      }} else {{
        td.textContent = v.toFixed(2);
        if (D.best[String(i)] !== undefined && Math.abs(v - D.best[String(i)]) < 0.005) {{
          td.className = "best";
        }} else if (D.second[String(i)] !== undefined && Math.abs(v - D.second[String(i)]) < 0.005) {{
          td.className = "second";
        }} else if (D.worst[String(i)] !== undefined && Math.abs(v - D.worst[String(i)]) < 0.005) {{
          td.className = "worst";
        }}
      }}
      tr.appendChild(td);
    }});
    bdy.appendChild(tr);
  }});
}}

render(-1, false);
</script>
</body>
</html>"""


def print_terminal(df: pd.DataFrame, sort_by: str | None = None):
    """Pretty-print leaderboard to terminal with ANSI colors."""
    if sort_by:
        asc = sort_by in LOWER_IS_BETTER
        key = sort_by
        # Try pretty name reverse lookup
        for k, v in PRETTY_NAMES.items():
            if v == sort_by:
                key = k
                break
        if key in df.columns:
            df = df.sort_values(key, ascending=asc, na_position="last")

    # Find best, second-best, and worst per column
    best = {}
    second = {}
    worst = {}
    for col in df.columns:
        if col == "Model":
            continue
        valid = df[col].dropna()
        if valid.empty:
            continue
        sorted_unique = sorted(valid.unique(), reverse=col not in LOWER_IS_BETTER)
        best[col] = sorted_unique[0]
        if len(sorted_unique) > 1:
            second[col] = sorted_unique[1]
            worst[col] = sorted_unique[-1]

    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

    # Column widths
    pretty = {c: PRETTY_NAMES.get(c, c) for c in df.columns}
    widths = {}
    for col in df.columns:
        w = len(pretty[col])
        for v in df[col]:
            if col == "Model":
                w = max(w, len(str(v)))
            elif pd.isna(v):
                w = max(w, 1)
            else:
                w = max(w, len(f"{v:.2f}"))
        widths[col] = w + 2

    # Header
    header = ""
    for col in df.columns:
        name = pretty[col]
        if col == "Model":
            header += f"{BOLD}{name:<{widths[col]}}{RESET}"
        else:
            header += f"{CYAN}{name:>{widths[col]}}{RESET}"
    print(header)
    print("─" * sum(widths.values()))

    # Rows
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        line = ""
        for col in df.columns:
            v = row[col]
            w = widths[col]
            if col == "Model":
                label = f"{rank}. {v}"
                line += f"{BOLD}{label:<{w}}{RESET}"
            elif pd.isna(v):
                line += f"{'—':>{w}}"
            elif col in best and np.isclose(v, best[col]):
                line += f"{GREEN}{BOLD}{v:>{w}.2f}{RESET}"
            elif col in second and np.isclose(v, second[col]):
                line += f"{UNDERLINE}{v:>{w}.2f}{RESET}"
            elif col in worst and np.isclose(v, worst[col]):
                line += f"{RED}{v:>{w}.2f}{RESET}"
            else:
                line += f"{v:>{w}.2f}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="MemoBench Leaderboard — aggregate and display model scores.")
    parser.add_argument(
        "--eval-dir", type=str, default=None, help="Directory with per-model eval CSVs (from run_eval.py)"
    )
    parser.add_argument(
        "--ors-dir", type=str, default=None, help="Directory with per-model ORS CSVs (from compute_ors.py)"
    )
    parser.add_argument("--vqa-dir", type=str, default=None, help="Directory with per-model VQA CSVs")
    parser.add_argument(
        "--load", type=str, default=None, help="Load a previously saved leaderboard CSV instead of computing"
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Save leaderboard to CSV file")
    parser.add_argument(
        "--format",
        choices=["terminal", "markdown", "latex", "csv", "html"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    parser.add_argument(
        "--sort", type=str, default=None, help="Sort by this metric (use pretty name, e.g. ORS, CamCtrl)"
    )
    parser.add_argument(
        "--subset",
        choices=["synthetic", "real", "all"],
        default="all",
        help="Filter clips by data type before aggregation",
    )
    args = parser.parse_args()

    if args.load:
        df = pd.read_csv(args.load)
    else:
        if not any([args.eval_dir, args.ors_dir, args.vqa_dir]):
            parser.error("Provide at least one of --eval-dir, --ors-dir, --vqa-dir")
        df = build_leaderboard(args.eval_dir, args.ors_dir, args.vqa_dir)

    # Sort
    if args.sort:
        key = args.sort
        for k, v in PRETTY_NAMES.items():
            if v == args.sort:
                key = k
                break
        if key in df.columns:
            asc = key in LOWER_IS_BETTER
            df = df.sort_values(key, ascending=asc, na_position="last")

    # Save
    if args.output:
        df.to_csv(args.output, index=False, float_format="%.4f")
        print(f"Saved to {args.output}", file=sys.stderr)

    # Display
    if args.format == "terminal":
        print_terminal(df, sort_by=args.sort)
    elif args.format == "markdown":
        print(format_markdown(df))
    elif args.format == "latex":
        print(format_latex(df))
    elif args.format == "csv":
        print(format_csv(df))
    elif args.format == "html":
        html_out = args.output.replace(".csv", ".html") if args.output else "leaderboard.html"
        Path(html_out).write_text(format_html(df))
        print(f"HTML leaderboard saved to {html_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
