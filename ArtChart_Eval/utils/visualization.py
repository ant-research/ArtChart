import argparse
import os
import re
import tempfile
import warnings
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


DEFAULT_SCORE_FIELDS: Sequence[str] = (
    "following_score",
    "readability_score",
    "ocr_text_score",
    "aes_score",
    "text_position_score",
    "math_score",
)

SCORE_LABELS: Dict[str, str] = {
    "following_score": "Instruction",
    "readability_score": "Readability",
    "ocr_text_score": "OCR Text",
    "aes_score": "Aesthetics",
    "text_position_score": "Text Position",
    "math_score": "Math",
}

TASK_TYPE_LABELS: Dict[str, str] = {
    "bar": "Vertical Bar",
    "hbar": "Horizontal Bar",
    "pie": "Pie",
    "area": "Area",
}


def translate_text(text_to_translate: str, mapping_dict: Dict[str, str]) -> str:
    return mapping_dict.get(text_to_translate, text_to_translate)


def _read_csv(csv_file: Union[str, PathLike]) -> pd.DataFrame:
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path, encoding="utf-8")


def _normalize_task_type(df: pd.DataFrame) -> pd.Series:
    if "task_type" not in df.columns:
        return pd.Series(["all"] * len(df), index=df.index, dtype="object")
    return df["task_type"].fillna("Unlabeled Type").astype(str).replace({"hor_bar": "hbar"})


def _available_score_fields(df: pd.DataFrame, score_fields: Optional[Sequence[str]] = None) -> List[str]:
    if score_fields:
        available = [field for field in score_fields if field in df.columns]
        missing = [field for field in score_fields if field not in df.columns]
        if missing:
            warnings.warn(f"Score fields not found and skipped: {missing}")
        return available

    ordered_fields = [field for field in DEFAULT_SCORE_FIELDS if field in df.columns]
    extra_fields = sorted(
        field
        for field in df.columns
        if field.endswith("_score") and field not in ordered_fields and field != "overall_score"
    )
    return ordered_fields + extra_fields


def _numeric_scores(df: pd.DataFrame, score_fields: Sequence[str]) -> pd.DataFrame:
    scores = df.loc[:, list(score_fields)].apply(pd.to_numeric, errors="coerce")
    return scores.where(np.isfinite(scores) & (scores >= 0))


def _safe_filename(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return safe or "unknown"


def analyze_by_task_type(csv_file: Union[str, PathLike]) -> Dict[str, Any]:
    """
    Summarize evaluation success and failure counts by chart type.

    This function is used by BaseEvaluator after writing the evaluation CSV.
    It selects the first available score field from DEFAULT_SCORE_FIELDS and
    treats rows with a finite numeric score as successful.
    """
    df = _read_csv(csv_file)
    task_types = _normalize_task_type(df)
    score_fields = _available_score_fields(df)

    results: Dict[str, Any] = {}
    if not score_fields:
        for task_type, group in df.groupby(task_types):
            results[str(task_type)] = {
                "Success": 0,
                "Failure": int(len(group)),
                "Total": int(len(group)),
                "Score Field": None,
            }
        return results

    primary_score = score_fields[0]
    numeric_score = pd.to_numeric(df[primary_score], errors="coerce")
    valid_score = np.isfinite(numeric_score)

    for task_type, indices in task_types.groupby(task_types).groups.items():
        task_valid = valid_score.loc[list(indices)]
        success_count = int(task_valid.sum())
        total_count = int(len(task_valid))
        results[str(task_type)] = {
            "Success": success_count,
            "Failure": total_count - success_count,
            "Total": total_count,
            "Score Field": primary_score,
        }

    return results


def summarize_scores(
    csv_file: Union[str, PathLike],
    score_fields: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return mean scores by task_type for the current ArtChart evaluation fields.
    """
    df = _read_csv(csv_file)
    available_fields = _available_score_fields(df, score_fields)
    if not available_fields:
        return pd.DataFrame()

    task_types = _normalize_task_type(df)
    scores = _numeric_scores(df, available_fields)
    score_table = pd.concat([task_types.rename("task_type"), scores], axis=1)
    return score_table.groupby("task_type")[available_fields].mean().round(2)


def _load_model_stats(
    csv_paths: Sequence[Union[str, PathLike]],
    names: Sequence[str],
    score_fields: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    stats_data: List[Dict[str, Any]] = []

    for csv_path, name in zip(csv_paths, names):
        df = _read_csv(csv_path)
        available_fields = _available_score_fields(df, score_fields)
        if not available_fields:
            warnings.warn(f"No score fields found in {csv_path}; skipping.")
            continue

        task_types = _normalize_task_type(df)
        scores = _numeric_scores(df, available_fields)
        task_table = pd.concat([task_types.rename("task_type"), scores], axis=1)
        task_stats = task_table.groupby("task_type")[available_fields].mean().round(2)
        overall_scores = scores[available_fields].mean().round(2)

        stats_data.append(
            {
                "name": name,
                "csv_path": str(csv_path),
                "score_fields": available_fields,
                "task_stats": task_stats,
                "overall_scores": overall_scores,
            }
        )

    return stats_data


def _common_score_fields(stats_data: Sequence[Dict[str, Any]]) -> List[str]:
    if not stats_data:
        return []

    common = set(stats_data[0]["score_fields"])
    for item in stats_data[1:]:
        common.intersection_update(item["score_fields"])

    return [field for field in DEFAULT_SCORE_FIELDS if field in common] + sorted(
        field for field in common if field not in DEFAULT_SCORE_FIELDS
    )


def _common_task_types(stats_data: Sequence[Dict[str, Any]]) -> List[str]:
    if not stats_data:
        return []

    common = set(stats_data[0]["task_stats"].index)
    for item in stats_data[1:]:
        common.intersection_update(item["task_stats"].index)

    return sorted(common)


def _write_figure(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(fig, "write_image"):
        try:
            fig.write_image(str(output_path), width=900, height=700, scale=2)
            print(f"Saved chart to: {output_path}")
        except Exception as exc:
            html_path = output_path.with_suffix(".html")
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            warnings.warn(
                f"Could not save PNG to {output_path}. Saved interactive HTML instead: {html_path}. "
                f"Install kaleido to enable PNG export. Original error: {exc}"
            )
        return

    try:
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
        print(f"Saved chart to: {output_path}")
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass


def _values_for_item(
    item: Dict[str, Any],
    score_fields: Sequence[str],
    values_key: str,
    task_type: Optional[str] = None,
) -> List[float]:
    if values_key == "overall_scores":
        values = item["overall_scores"].reindex(score_fields)
    else:
        values = item["task_stats"].loc[task_type, list(score_fields)]
    return [0.0 if pd.isna(value) else float(value) for value in values.tolist()]


def _make_matplotlib_radar_figure(
    stats_data: Sequence[Dict[str, Any]],
    score_fields: Sequence[str],
    title: str,
    values_key: str,
    task_type: Optional[str] = None,
) -> Any:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib.pyplot as plt

    labels = [SCORE_LABELS.get(field, field) for field in score_fields]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"polar": True})
    for item in stats_data:
        r_values = _values_for_item(item, score_fields, values_key, task_type)
        r_closed = r_values + r_values[:1]
        ax.plot(angles_closed, r_closed, linewidth=2, label=item["name"])
        ax.fill(angles_closed, r_closed, alpha=0.06)

    ax.set_title(title, pad=24)
    ax.set_ylim(0, 10)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=min(3, max(1, len(stats_data))))
    fig.tight_layout()
    return fig


def _make_radar_figure(
    stats_data: Sequence[Dict[str, Any]],
    score_fields: Sequence[str],
    title: str,
    values_key: str,
    task_type: Optional[str] = None,
) -> Any:
    try:
        return _make_matplotlib_radar_figure(
            stats_data=stats_data,
            score_fields=score_fields,
            title=title,
            values_key=values_key,
            task_type=task_type,
        )
    except ModuleNotFoundError:
        import plotly.graph_objects as go

    theta = [SCORE_LABELS.get(field, field) for field in score_fields]
    theta_closed = theta + [theta[0]]

    fig = go.Figure()
    for item in stats_data:
        r_values = _values_for_item(item, score_fields, values_key, task_type)
        r_closed = r_values + [r_values[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=theta_closed,
                fill="none",
                name=item["name"],
                hovertemplate="%{theta}: %{r:.2f}<extra>" + item["name"] + "</extra>",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 10],
                "tickmode": "linear",
                "dtick": 2,
            }
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5},
        margin={"l": 80, "r": 80, "t": 90, "b": 100},
    )
    return fig


def _write_summary_csv(
    stats_data: Sequence[Dict[str, Any]],
    score_fields: Sequence[str],
    output_path: Path,
) -> None:
    rows = []
    for item in stats_data:
        overall_row = {"model": item["name"], "task_type": "overall"}
        for field in score_fields:
            overall_row[field] = item["overall_scores"].get(field, np.nan)
        rows.append(overall_row)

        task_stats = item["task_stats"].reindex(columns=score_fields)
        for task_type, row in task_stats.iterrows():
            task_row = {"model": item["name"], "task_type": task_type}
            for field in score_fields:
                task_row[field] = row.get(field, np.nan)
            rows.append(task_row)

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved summary CSV to: {output_path}")


def radar_visualization_text(
    csv_paths: List[str],
    names: List[str],
    save_dir: str,
    score_fields: Optional[Sequence[str]] = None,
    per_task: bool = True,
) -> None:
    """
    Generate radar charts for one or more ArtChart evaluation CSV files.

    Outputs:
        eval.png or eval.html: overall mean scores across all chart types.
        eval_<task_type>.png or .html: per-chart-type radar charts.
        eval_summary.csv: numeric summary used by the charts.
    """
    if len(csv_paths) != len(names):
        raise ValueError("csv_paths and names must have the same length.")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    stats_data = _load_model_stats(csv_paths, names, score_fields)
    if not stats_data:
        warnings.warn("No valid CSV files were loaded. Nothing to visualize.")
        return

    common_fields = _common_score_fields(stats_data)
    if not common_fields:
        warnings.warn("No common score fields found across the provided CSV files.")
        return

    _write_summary_csv(stats_data, common_fields, save_path / "eval_summary.csv")

    overall_fig = _make_radar_figure(
        stats_data=stats_data,
        score_fields=common_fields,
        title="ArtChart Evaluation",
        values_key="overall_scores",
    )
    _write_figure(overall_fig, save_path / "eval.png")

    if not per_task:
        return

    for task_type in _common_task_types(stats_data):
        task_label = TASK_TYPE_LABELS.get(str(task_type), str(task_type))
        task_fig = _make_radar_figure(
            stats_data=stats_data,
            score_fields=common_fields,
            title=f"ArtChart Evaluation - {task_label}",
            values_key="task_stats",
            task_type=str(task_type),
        )
        _write_figure(task_fig, save_path / f"eval_{_safe_filename(str(task_type))}.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ArtChart evaluation summaries and radar charts.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--csv-paths",
        nargs="+",
        required=True,
        help="One or more evaluation CSV files.",
    )
    parser.add_argument(
        "-n",
        "--names",
        nargs="+",
        required=True,
        help="Legend names corresponding to --csv-paths.",
    )
    parser.add_argument(
        "-d",
        "--save-dir",
        type=str,
        default="output_charts",
        help="Directory for output charts and summary CSV.",
    )
    parser.add_argument(
        "--score-fields",
        nargs="+",
        default=None,
        help="Optional score fields to visualize. Defaults to all current ArtChart score fields found in the CSV.",
    )
    parser.add_argument(
        "--no-per-task",
        action="store_true",
        help="Only generate the overall radar chart.",
    )
    args = parser.parse_args()

    radar_visualization_text(
        csv_paths=args.csv_paths,
        names=args.names,
        save_dir=args.save_dir,
        score_fields=args.score_fields,
        per_task=not args.no_per_task,
    )


if __name__ == "__main__":
    main()
