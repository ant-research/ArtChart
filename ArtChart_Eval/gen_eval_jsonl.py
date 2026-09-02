"""
Build an evaluation jsonl file from generated ArtChart images and benchmark data.

Expected layout:
    OUTPUT_DIR/<chart_type>/<sample_id>.png
    EVAL_DATA_DIR/<chart_type>/<sample_id>/prompt.txt
    EVAL_DATA_DIR/<chart_type>/<sample_id>/gray.png

Each jsonl entry contains:
    output_image, instruction, task_type, lang, title, data_str, data
"""

import argparse
import json
import os
import re
from pathlib import Path


def parse_values_from_text(text):
    """Parse numeric chart values from a data text segment."""
    values = []
    for item in re.split(r"[\n,，、;；|]", str(text)):
        item = item.strip()
        if not item:
            continue
        numbers = re.findall(r"-?\d+(?:\.\d+)?", item)
        if numbers:
            # Use the last number to avoid category-side numbers such as 2024/Q1.
            value = float(numbers[-1])
            values.append(int(value) if value.is_integer() else value)
    return values


def extract_quoted_segments(prompt):
    segments = []
    segments.extend(re.findall(r"“([^”]*)”", prompt))
    segments.extend(re.findall(r'"([^"]*)"', prompt))
    segments.extend(re.findall(r"'([^']*)'", prompt))
    return segments


def extract_data_from_prompt(prompt):
    """Extract the chart value list from quoted data text in the prompt."""
    for segment in extract_quoted_segments(prompt):
        values = parse_values_from_text(segment)
        if len(values) >= 2:
            return values, segment

    values = parse_values_from_text(prompt)
    if len(values) >= 2:
        return values, prompt

    return None, None


def extract_title_from_prompt(prompt, data_segment):
    """Extract the chart title from quoted prompt text as a fallback metadata source."""
    candidates = [segment for segment in extract_quoted_segments(prompt) if segment != data_segment]
    return candidates[-1] if candidates else ""


def jsonl_output_path(jsonl_dir):
    path = Path(jsonl_dir)
    if path.suffix == ".jsonl":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / "eval.jsonl"


def build_jsonl(artchart_dir, eval_data_dir, output_jsonl_dir, chart_types, lang="zh"):
    artchart_dir = Path(artchart_dir)
    eval_data_dir = Path(eval_data_dir)
    output_path = jsonl_output_path(output_jsonl_dir)
    entries = []

    for chart_type in chart_types:
        if chart_type == "hor_bar":
            chart_type = "hbar"
        gen_img_dir = artchart_dir / chart_type
        bench_chart_dir = eval_data_dir / chart_type

        if not gen_img_dir.exists():
            print(f"Warning: generated image directory not found: {gen_img_dir}")
            continue
        if not bench_chart_dir.exists():
            print(f"Warning: benchmark chart directory not found: {bench_chart_dir}")
            continue

        for image_path in sorted(gen_img_dir.glob("*.png")):
            sample_id = image_path.stem
            sample_dir = bench_chart_dir / sample_id
            prompt_path = sample_dir / "prompt.txt"
            gray_path = sample_dir / "gray.png"

            if not prompt_path.exists():
                print(f"Warning: prompt not found: {prompt_path}")
                continue
            if not gray_path.exists():
                print(f"Warning: gray image not found: {gray_path}")

            prompt = prompt_path.read_text(encoding="utf-8").strip()
            data_values, data_segment = extract_data_from_prompt(prompt)
            if data_values is None:
                print(f"Warning: failed to extract data list for {chart_type}/{sample_id}")
                print(f"  prompt: {prompt}")
                continue
            title = extract_title_from_prompt(prompt, data_segment)

            entry = {
                "output_image": str(image_path.resolve()),
                "instruction": prompt,
                "task_type": chart_type,
                "lang": lang,
                "title": title,
                "data_str": data_segment,
                "data": data_values,
            }
            entries.append(entry)

    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Created jsonl: {output_path}")
    print(f"Total entries: {len(entries)}")
    return output_path, len(entries)


def parse_args():
    parser = argparse.ArgumentParser(description="Build ArtChart evaluation jsonl.")
    parser.add_argument("--artchart-dir", type=str, required=True, help="Generated ArtChart image root directory.")
    parser.add_argument("--eval-data-dir", type=str, required=True, help="Benchmark data root directory.")
    parser.add_argument("--output-jsonl-dir", type=str, required=True, help="Output jsonl directory or .jsonl path.")
    parser.add_argument(
        "--chart-types",
        type=str,
        nargs="+",
        default=["pie", "bar", "hbar", "area"],
        help="Chart types to include. hor_bar is accepted as a legacy alias for hbar.",
    )
    parser.add_argument("--lang", type=str, default="zh", help="Language field written to each jsonl entry.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Build ArtChart evaluation JSONL")
    print("=" * 60)
    print(f"ARTCHART_DIR      : {args.artchart_dir}")
    print(f"EVAL_DATA_DIR     : {args.eval_data_dir}")
    print(f"OUTPUT_JSONL_DIR  : {args.output_jsonl_dir}")
    print(f"Chart types       : {args.chart_types}")
    print("=" * 60)

    build_jsonl(
        artchart_dir=args.artchart_dir,
        eval_data_dir=args.eval_data_dir,
        output_jsonl_dir=args.output_jsonl_dir,
        chart_types=args.chart_types,
        lang=args.lang,
    )


if __name__ == "__main__":
    main()
