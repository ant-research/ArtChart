"""
Generate grayscale benchmark control images from ArtBench JSON metadata.

The default output layout is:
    <output_dir>/<chart_type>/<sample_id>/
        gray.png
        prompt.txt

The ArtBench JSON files use "hbar" for horizontal bar charts.

Example:
    python ArtChart_Bench/generate_gray_from_json.py \
        --json-path ArtChart_Bench/ArtBench-200.json \
        --output-dir ArtChart_Bench/ArtBench-200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


CHART_TYPE_ALIASES = {
    "bar": "bar",
    "vertical_bar": "bar",
    "hbar": "hbar",
    "hor_bar": "hbar",
    "horizontal_bar": "hbar",
    "pie": "pie",
    "area": "area",
}


def parse_resolution(resolution: str) -> Tuple[int, int]:
    normalized = resolution.lower().replace("*", "x").replace("×", "x")
    match = re.fullmatch(r"(\d+)x(\d+)", normalized)
    if not match:
        raise ValueError(f"Invalid resolution: {resolution}. Expected format like 1024x1024.")

    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid resolution: {resolution}. Width and height must be positive.")
    return width, height


def normalize_chart_type(chart_type: str) -> str:
    key = str(chart_type).strip().lower()
    if key not in CHART_TYPE_ALIASES:
        raise ValueError(f"Unsupported chart_type: {chart_type}")
    return CHART_TYPE_ALIASES[key]


def output_chart_type_name(source_chart_type: str, hbar_dir_name: str) -> str:
    internal_type = normalize_chart_type(source_chart_type)
    if internal_type == "hbar":
        return hbar_dir_name
    return internal_type


class ChartImageCreator:
    """Create grayscale chart-structure images for one benchmark sample."""

    def __init__(
        self,
        chart_type: str,
        resolution: str,
        font_path: Optional[str] = None,
        font_size: int = 20,
        dpi: int = 100,
    ) -> None:
        self.chart_type = normalize_chart_type(chart_type)
        self.target_w, self.target_h = parse_resolution(resolution)
        self.canvas_w, self.canvas_h = self._get_canvas_size()
        self.font_path = font_path
        self.font_size = font_size
        self.dpi = dpi
        self._setup_chart_layout()

    def _get_canvas_size(self) -> Tuple[int, int]:
        if self.chart_type == "hbar" and self.target_w != self.target_h:
            return self.target_h, self.target_w
        return self.target_w, self.target_h

    def _setup_chart_layout(self) -> None:
        is_square = self.target_w == self.target_h
        if self.chart_type == "pie":
            self.chart_width_ratio = 0.60 if is_square else 0.50
            self.bottom_offset_ratio = 0.80
            self.top_margin_ratio = 0.70
        elif self.chart_type == "bar":
            self.chart_width_ratio = 0.80 if is_square else 0.70
            self.bottom_offset_ratio = 0.30
            self.top_margin_ratio = 0.40
        elif self.chart_type == "hbar":
            self.chart_width_ratio = 0.70 if is_square else 0.75
            self.bottom_offset_ratio = 0.30
            self.top_margin_ratio = 0.40
        else:
            self.chart_width_ratio = 0.70 if is_square else 0.40
            self.bottom_offset_ratio = 0.50 if is_square else 0.80
            self.top_margin_ratio = 0.50 if is_square else 0.70

    @staticmethod
    def _load_plotting_modules():
        cache_root = Path(tempfile.gettempdir()) / "artchart_plot_cache"
        mpl_config_dir = cache_root / "matplotlib"
        xdg_cache_dir = cache_root / "xdg"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        xdg_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.font_manager as fm
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import numpy as np

        try:
            from scipy.interpolate import PchipInterpolator
        except ImportError:
            PchipInterpolator = None

        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return np, plt, mpatches, fm, PchipInterpolator

    def _font_properties(self, fm):
        if self.font_path and Path(self.font_path).exists():
            return fm.FontProperties(fname=self.font_path)
        return None

    def _figure(self, plt, area: bool = False):
        width_inches = self.canvas_w / self.dpi
        height_inches = self.canvas_h / self.dpi
        fig = plt.figure(figsize=(width_inches, height_inches), dpi=self.dpi)
        if area:
            ax = fig.add_axes([0.01, 0.15, 0.98, 0.6])
            ax.set_frame_on(False)
        else:
            left_margin = (1 - self.chart_width_ratio) / 2
            ax = fig.add_axes([left_margin, 0, self.chart_width_ratio, 1])
        return fig, ax

    @staticmethod
    def _scale_y_values(np, y):
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            return y
        max_y = max(y)
        scale_factor = max_y / 10 if max_y > 10 else 1
        return y / scale_factor

    @staticmethod
    def _numeric_x(np, x_values: List[str]):
        x_numeric = np.arange(len(x_values))
        x_map = dict(zip(x_numeric, x_values))
        return x_numeric, x_map

    @staticmethod
    def _rounded_bar(ax, mpatches, x, y, width, height, color, alpha: float = 1.0) -> None:
        radius = width / 2
        if height <= 0:
            return
        if height <= radius:
            ellipse = mpatches.Ellipse((x + width / 2, y), width, 2 * height, facecolor=color, alpha=alpha)
            ax.add_patch(ellipse)
            return
        rect_height = height - radius
        rect = mpatches.Rectangle((x, y), width, rect_height, facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        circle = mpatches.Wedge((x + width / 2, y + rect_height), width / 2, 0, 180, facecolor=color, alpha=alpha)
        ax.add_patch(circle)

    def _draw_bars(self, ax, mpatches, x, y_scaled):
        bar_width = 0.6
        y_max_data = max(y_scaled) if len(y_scaled) else 1
        if y_max_data <= 0:
            y_max_data = 1
        bottom_offset = y_max_data * self.bottom_offset_ratio
        top_margin = y_max_data * self.top_margin_ratio
        y_max = y_max_data + bottom_offset + top_margin

        for x_val, y_val in zip(x, y_scaled):
            self._rounded_bar(ax, mpatches, x_val - bar_width / 2, bottom_offset, bar_width, y_val, "gray", alpha=0.8)

        ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
        ax.set_ylim(0, y_max)
        ax.margins(0, 0)
        ax.axis("off")
        return y_max, bottom_offset

    def _draw_bar_labels(self, ax, x, y_scaled, x_map, y_original, y_max, bottom_offset, font_kwargs) -> None:
        rotation = 90 if self.chart_type == "hbar" else 0
        for x_val in x:
            ax.text(
                x_val,
                bottom_offset * 0.5,
                f"{x_map[x_val]}",
                ha="center",
                va="center",
                fontsize=self.font_size,
                alpha=0.8,
                rotation=rotation,
                **font_kwargs,
            )

        for x_val, y_val, display_value in zip(x, y_scaled, y_original):
            ax.text(
                x_val,
                bottom_offset + y_val + y_max * 0.02,
                f"{display_value:g}",
                ha="center",
                va="bottom",
                fontsize=self.font_size,
                alpha=0.8,
                rotation=rotation,
                **font_kwargs,
            )

    def _smooth_area(self, np, PchipInterpolator, x, y_scaled):
        x_min, x_max = min(x), max(x)
        data_range = x_max - x_min or 1
        padding = data_range * 0.1
        x_extended = np.concatenate([[x_min - padding], x, [x_max + padding]])
        y_extended = np.concatenate([[0], y_scaled, [0]])
        x_smooth = np.linspace(x_min - padding, x_max + padding, 500)
        if PchipInterpolator is None:
            y_smooth = np.interp(x_smooth, x_extended, y_extended)
        else:
            y_smooth = PchipInterpolator(x_extended, y_extended)(x_smooth)
        return x_smooth, y_smooth

    def _draw_area(self, ax, np, PchipInterpolator, x, y_scaled):
        x_smooth, y_smooth = self._smooth_area(np, PchipInterpolator, x, y_scaled)
        ax.plot(x_smooth, y_smooth, color="gray", linewidth=0)
        ax.fill_between(x_smooth, 0, y_smooth, color="gray", alpha=0.8)
        ax.scatter(x, y_scaled, color="black", s=150, zorder=5)
        ax.scatter(x, [0] * len(x), color="black", s=150, zorder=5)
        y_max = max(y_smooth) * 1.1 if max(y_smooth) > 0 else 1
        ax.set_ylim(0, y_max)
        ax.set_xlim(min(x_smooth), max(x_smooth))
        ax.margins(0, 0)
        ax.axis("off")
        return y_max

    def _draw_area_labels(self, ax, x, y_scaled, x_map, y_original, y_max, font_kwargs) -> None:
        for x_val in x:
            ax.text(
                x_val,
                -0.08 * y_max,
                f"{x_map[x_val]}",
                ha="center",
                va="center",
                fontsize=self.font_size,
                alpha=0.8,
                **font_kwargs,
            )
        for x_val, y_val, display_value in zip(x, y_scaled, y_original):
            ax.text(
                x_val,
                y_val + y_max * 0.03,
                f"{display_value:g}",
                ha="center",
                va="bottom",
                fontsize=self.font_size,
                alpha=0.8,
                **font_kwargs,
            )

    def _draw_pie(self, ax, np, y, x_map, with_text: bool, font_kwargs) -> None:
        wedges, _ = ax.pie(
            y,
            labels=None,
            colors=["gray"] * len(y),
            autopct=None,
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 0, "edgecolor": "none"},
        )
        separator_linewidth = 8.0
        inner_radius = 0.0
        outer_radius = 1.0
        boundary_angles = []
        for wedge in wedges:
            for angle in (wedge.theta1, wedge.theta2):
                normalized_angle = angle % 360
                if not any(abs(normalized_angle - existing) < 1e-6 for existing in boundary_angles):
                    boundary_angles.append(normalized_angle)

        for angle in boundary_angles:
            angle_rad = np.radians(angle)
            ax.plot(
                [inner_radius * np.cos(angle_rad), outer_radius * np.cos(angle_rad)],
                [inner_radius * np.sin(angle_rad), outer_radius * np.sin(angle_rad)],
                color="#FFFFFF",
                linewidth=separator_linewidth,
                solid_capstyle="butt",
                zorder=3,
            )

        circle_angles = np.linspace(0, 2 * np.pi, 512)
        ax.plot(
            outer_radius * np.cos(circle_angles),
            outer_radius * np.sin(circle_angles),
            color="#FFFFFF",
            linewidth=separator_linewidth,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.axis("equal")
        if not with_text:
            return

        for wedge, label, value in zip(wedges, x_map.values(), y):
            angle = (wedge.theta1 + wedge.theta2) / 2
            angle_rad = np.radians(angle)
            label_x = 1.25 * np.cos(angle_rad)
            label_y = 1.25 * np.sin(angle_rad)
            value_x = 1.45 * np.cos(angle_rad)
            value_y = 1.45 * np.sin(angle_rad)
            ax.text(label_x, label_y, f"{label}", ha="center", va="center", fontsize=self.font_size, alpha=0.8, **font_kwargs)
            ax.text(
                value_x,
                value_y,
                f"{value:g}%",
                ha="center",
                va="center",
                fontsize=self.font_size * 0.9,
                alpha=0.8,
                **font_kwargs,
            )

    def create(self, data: Dict[str, List[Any]], save_path: Path, with_text: bool = False) -> None:
        np, plt, mpatches, fm, PchipInterpolator = self._load_plotting_modules()
        x, x_map = self._numeric_x(np, data["X"])
        y = np.asarray(data["Y"], dtype=float)
        y_scaled = self._scale_y_values(np, y)
        font_prop = self._font_properties(fm)
        font_kwargs = {"fontproperties": font_prop} if font_prop else {}

        if self.chart_type == "pie":
            fig, ax = self._figure(plt, area=False)
            self._draw_pie(ax, np, y, x_map, with_text, font_kwargs)
        elif self.chart_type == "area":
            fig, ax = self._figure(plt, area=True)
            y_max = self._draw_area(ax, np, PchipInterpolator, x, y_scaled)
            if with_text:
                self._draw_area_labels(ax, x, y_scaled, x_map, y, y_max, font_kwargs)
        else:
            fig, ax = self._figure(plt, area=False)
            y_max, bottom_offset = self._draw_bars(ax, mpatches, x, y_scaled)
            if with_text:
                self._draw_bar_labels(ax, x, y_scaled, x_map, y, y_max, bottom_offset, font_kwargs)

        plt.savefig(save_path, dpi=self.dpi, transparent=False, pad_inches=0, facecolor="white")
        plt.close(fig)

        with Image.open(save_path) as image:
            image = image.convert("RGB")
            if self.chart_type == "hbar":
                image = image.rotate(-90, expand=True)
            image.save(save_path)


def parse_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(value))
    if not matches:
        raise ValueError(f"Could not parse numeric value from {value!r}")
    return float(matches[-1])


def sample_to_chart_data(sample: Dict[str, Any]) -> Dict[str, List[Any]]:
    rows = sample.get("data")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Sample is missing a non-empty data list")

    labels: List[str] = []
    values: List[float] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid data row: {row!r}")
        label = str(row.get("category", "")).strip()
        if not label:
            raise ValueError(f"Data row is missing category: {row!r}")
        value = row.get("raw_value", row.get("value"))
        labels.append(label)
        values.append(parse_number(value))

    if len(labels) < 2:
        raise ValueError("A chart sample needs at least two data points")
    return {"X": labels, "Y": values}


def normalize_pie_values(data: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    total = sum(float(value) for value in data["Y"])
    if total > 0 and abs(total - 100) > 1:
        normalized = [round(float(value) / total * 100, 1) for value in data["Y"]]
        normalized[-1] = round(100 - sum(normalized[:-1]), 1)
        return {"X": data["X"], "Y": normalized}
    return data


def load_samples(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{json_path} must contain a JSON list of samples")
    return payload


def default_output_dir(json_path: Path) -> Path:
    return json_path.with_suffix("")


def filter_samples(
    samples: Iterable[Dict[str, Any]],
    chart_types: Iterable[str],
    limit: Optional[int],
    limit_per_type: Optional[int],
) -> List[Dict[str, Any]]:
    wanted = {normalize_chart_type(chart_type) for chart_type in chart_types}
    selected: List[Dict[str, Any]] = []
    per_type_count: Dict[str, int] = defaultdict(int)

    for sample in samples:
        internal_type = normalize_chart_type(sample.get("chart_type", ""))
        if internal_type not in wanted:
            continue
        if limit_per_type is not None and per_type_count[internal_type] >= limit_per_type:
            continue

        selected.append(sample)
        per_type_count[internal_type] += 1
        if limit is not None and len(selected) >= limit:
            break

    return selected


def write_sample(
    sample: Dict[str, Any],
    output_dir: Path,
    resolution: str,
    hbar_dir_name: str,
    font_path: Optional[str],
    font_size: int,
    dpi: int,
    write_text: bool,
    overwrite: bool,
) -> Tuple[str, bool]:
    sample_id = str(sample.get("id", "")).strip()
    if not sample_id:
        raise ValueError("Sample is missing id")

    source_chart_type = str(sample.get("chart_type", "")).strip()
    internal_chart_type = normalize_chart_type(source_chart_type)
    chart_dir_name = output_chart_type_name(source_chart_type, hbar_dir_name)
    sample_dir = output_dir / chart_dir_name / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    prompt = str(sample.get("prompt", "")).strip()
    if prompt:
        prompt_path = sample_dir / "prompt.txt"
        if overwrite or not prompt_path.exists():
            prompt_path.write_text(prompt + "\n", encoding="utf-8")

    gray_path = sample_dir / "gray.png"
    text_path = sample_dir / "text.png"
    if gray_path.exists() and not overwrite and (not write_text or text_path.exists()):
        return chart_dir_name, False

    data = sample_to_chart_data(sample)
    if internal_chart_type == "pie":
        data = normalize_pie_values(data)

    creator = ChartImageCreator(
        chart_type=internal_chart_type,
        resolution=resolution,
        font_path=font_path,
        font_size=font_size,
        dpi=dpi,
    )
    if overwrite or not gray_path.exists():
        creator.create(data, gray_path, with_text=False)
    if write_text and (overwrite or not text_path.exists()):
        creator.create(data, text_path, with_text=True)
    return chart_dir_name, True


def generate_from_json(
    json_path: Path,
    output_dir: Path,
    chart_types: Iterable[str],
    resolution: str,
    hbar_dir_name: str,
    font_path: Optional[str],
    font_size: int,
    dpi: int,
    write_text: bool,
    overwrite: bool,
    limit: Optional[int],
    limit_per_type: Optional[int],
) -> Counter:
    samples = load_samples(json_path)
    selected = filter_samples(samples, chart_types, limit=limit, limit_per_type=limit_per_type)

    counts: Counter = Counter()
    skipped: Counter = Counter()
    errors: List[str] = []
    total = len(selected)

    for index, sample in enumerate(selected, start=1):
        try:
            chart_dir_name, generated = write_sample(
                sample=sample,
                output_dir=output_dir,
                resolution=resolution,
                hbar_dir_name=hbar_dir_name,
                font_path=font_path,
                font_size=font_size,
                dpi=dpi,
                write_text=write_text,
                overwrite=overwrite,
            )
            if generated:
                counts[chart_dir_name] += 1
            else:
                skipped[chart_dir_name] += 1
        except Exception as exc:
            sample_id = sample.get("id", f"#{index}")
            errors.append(f"{sample_id}: {exc}")

        if index == total or index % 100 == 0:
            print(f"Processed {index}/{total}")

    for message in errors[:20]:
        print(f"Warning: {message}")
    if len(errors) > 20:
        print(f"Warning: omitted {len(errors) - 20} additional errors")

    counts["skipped_existing"] = sum(skipped.values())
    counts["errors"] = len(errors)
    return counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate four chart-type grayscale control images from ArtBench JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json-path", type=Path, required=True, help="Path to ArtBench JSON metadata.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output benchmark directory. Defaults to the JSON path without the .json suffix.",
    )
    parser.add_argument(
        "--chart-types",
        nargs="+",
        default=["bar", "hbar", "pie", "area"],
        help="Chart types to generate. Supports bar, hbar, pie, and area. hor_bar is accepted as a legacy alias.",
    )
    parser.add_argument("--resolution", default="1024x1024", help="Output image resolution, e.g. 1024x1024 or 1024x768.")
    parser.add_argument("--hbar-dir-name", default="hbar", help="Directory name to use for horizontal bar samples.")
    parser.add_argument("--font-path", default=None, help="Optional font file path used only when --write-text is set.")
    parser.add_argument("--font-size", type=int, default=20, help="Font size used only when --write-text is set.")
    parser.add_argument("--dpi", type=int, default=100, help="Matplotlib DPI used to render exact pixel dimensions.")
    parser.add_argument("--write-text", action="store_true", help="Also generate text.png with labels and values.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate images even if they already exist.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum total number of samples to process.")
    parser.add_argument("--limit-per-type", type=int, default=None, help="Maximum number of samples to process per chart type.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.json_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"JSON       : {args.json_path}")
    print(f"Output dir : {output_dir}")
    print(f"Chart types: {args.chart_types}")
    print(f"Resolution : {args.resolution}")

    counts = generate_from_json(
        json_path=args.json_path,
        output_dir=output_dir,
        chart_types=args.chart_types,
        resolution=args.resolution,
        hbar_dir_name=args.hbar_dir_name,
        font_path=args.font_path,
        font_size=args.font_size,
        dpi=args.dpi,
        write_text=args.write_text,
        overwrite=args.overwrite,
        limit=args.limit,
        limit_per_type=args.limit_per_type,
    )

    print("Done.")
    for chart_type in sorted(k for k in counts if k not in {"errors", "skipped_existing"}):
        print(f"  {chart_type}: {counts[chart_type]} generated")
    print(f"  skipped_existing: {counts['skipped_existing']}")
    print(f"  errors: {counts['errors']}")


if __name__ == "__main__":
    main()
