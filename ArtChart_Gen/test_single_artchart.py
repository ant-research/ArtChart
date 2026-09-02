"""
Generate one ArtChart sample end to end.

This script creates:
    prompt.txt
    gray.png
    text.png
    artchart.png

Example:
    python test_single_artchart.py \
        --title "2026年最受欢迎旅游住宿场地" \
        --data "民宿 35%，酒店 28%，露营 20%，青旅 12%" \
        --chart_type hbar \
        --resolution 1024x1024 \
        --output_dir "$save_paths" \
        --base_model_path "$base_model" \
        --artchart_controlnet_path "$controlnet_path" \
        --lightning_lora_path "$lightning_lora_path" \
        --grpo_lora_path "$grpo_lora_path" \
        --api_key "$VLM_API_KEY" \
        --api_url "$VLM_API_URL"

If --prompt is provided, --title is ignored and the LLM prompt expansion step is skipped.
"""

import argparse
import json
import os
import re
from pathlib import Path

import requests
from PIL import Image
import torch
from diffusers import (
    QwenImageControlNetModel,
    QwenImageControlNetPipeline,
    QwenImageTransformer2DModel,
)



CHART_TYPE_TO_CN = {
    "bar": "柱状",
    "hbar": "横向条形",
    "pie": "饼",
    "area": "面积",
}


class ArtChartGenerator:
    def __init__(
        self,
        controlnet_path,
        base_model_path,
        lightning_lora_path,
        grpo_lora_path=None,
    ):
        """Load all models and LoRA adapters.

        Args:
            controlnet_path: Directory of the trained ControlNet checkpoint.
            base_model_path: Directory of the Qwen-Image base model.
            lightning_lora_path: Path to the Lightning LoRA weights file. This LoRA accelerates inference so only 4 steps are needed.
            grpo_lora_path: Optional path to a GRPO-trained LoRA weights file.If given, it is loaded on top of the Lightning LoRA and both adapters are active at equal weight.
        """

        print(f"Loading ControlNet from {controlnet_path}...")
        controlnet = QwenImageControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            base_model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.pipe = QwenImageControlNetPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        if lightning_lora_path is not None:
            self.pipe.load_lora_weights(lightning_lora_path, adapter_name="lightning")
            self.num_inference_steps=4
            self.cfg=1
        else:
            self.num_inference_steps=40
            self.cfg=3.5
        if grpo_lora_path is not None:
            self.pipe.load_lora_weights(grpo_lora_path, adapter_name="grpo")
        if lightning_lora_path is not None and grpo_lora_path is not None:
            self.pipe.set_adapters(["lightning", "grpo"], adapter_weights=[1.0, 1.0])

        print("Model loaded successfully!")

    def generate(self, prompt, control_image_path, save_path, controlnet_conditioning_scale=1.0):
        """Generate a single chart image and save it.

        Args:
            prompt: Text prompt describing the desired chart.
            control_image_path: Path to the grayscale ControlNet control image.
            save_path: Where to write the generated PNG.
            controlnet_conditioning_scale: How strongly the control image guides generation. 1.0 means full strength.

        Returns:
            The save_path the image was written to.
        """
        control_image = Image.open(control_image_path).convert("RGB")

        image = self.pipe(
            prompt=prompt,
            negative_prompt=" ",
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            width=control_image.size[0],
            height=control_image.size[1],
            num_inference_steps=self.num_inference_steps,
            true_cfg_scale=self.cfg,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]

        image.save(save_path)
        return save_path


def parse_resolution(resolution):
    normalized = resolution.lower().replace("*", "x").replace("×", "x")
    if normalized not in {"1024x1024", "1024x768"}:
        return 1024,1024
    width, height = normalized.split("x")
    return int(width), int(height)


def vlm_llm(api_key, model_name, prompt, api_url):
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=120)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def build_prompt_instruction(title, data, chart_type):
    type_cn = CHART_TYPE_TO_CN[chart_type]
    if chart_type == "hbar":
        layout = "图表中从上到下的类目数据分别是"
        example = (
            "制作一张热门婚礼场地排行的横向条形图。图表采用浪漫唯美的水彩手绘风格，"
            "横条设计成延伸的粉色丝带与盛开的花藤交织形态，背景点缀玫瑰花瓣和柔和光斑。"
            "图表中从上到下的类目数据分别是\"草坪 35%，酒店 28%，海岛 20%，民宿 12%，教堂 5%\"。"
            "图表顶部写着醒目的艺术标题\"2024年最受欢迎婚礼场地TOP5\"。"
        )
    elif chart_type == "bar":
        layout = "图表中从左到右的类目数据分别是"
        example = (
            "生成一张天文馆参观量的柱状图。图表采用未来感太空科幻矢量插画风格，"
            "柱状条被设计成一排准备发射的火箭，背景是星空和发光星云。"
            "图表中从左到右的类目数据分别是\"一月 1.2w，二月 1.6w，三月 1.8w，四月 2.2w，五月 3.0w\"。"
            "图表正上方写着发光大标题\"天文馆上半年参观人数统计\"。"
        )
    elif chart_type == "pie":
        layout = "图表中类目数据按照顺时针依次排布"
        example = (
            "设计一张员工学历结构的饼图。图表采用精美的2.5D等距微立体插画风格，"
            "饼图被设计成悬浮的立体圆环切片，周围漂浮博士帽、证书和办公元素。"
            "图表中类目数据按照顺时针依次排布\"大学 55%，硕士 25%，专科 15%，博士 5%\"。"
            "图片上方写着硬朗标题\"公司年度员工学历分布统计\"。"
        )
    else:
        layout = "图表中从左到右的类目数据分别是"
        example = (
            "生成一张年度游客人次累计的面积图。图表采用清新的水彩手绘风格，"
            "将面积图的数据波动转化为连绵青山和层叠梯田，背景点缀传统农家小院。"
            "图表中从左到右的类目数据分别是\"春季 1.5万，夏季 4.2万，秋季 5.8万，冬季 2.1万\"。"
            "图表顶部写着醒目的标题\"2023年美丽乡村旅游接待量趋势\"。"
        )

    return f"""
你是一位创意数据可视化设计师。请为我创作一张独特的{type_cn}图设计提示词(prompt)。

要求：
1. 基于给定标题"{title}"和数据"{data}"扩写出一个艺术图表生图prompt。
2. 自由发挥视觉风格、背景环境、图表创意形态、颜色搭配和装饰元素，要求风格与标题主题高度相关。
3. 图表的数据承载形状必须保留为清晰可读的{type_cn}图结构，同时进行艺术化发挥。
4. 整个prompt必须是一个连续自然段落，不要分点列举。
5. prompt必须包含图表数据"{data}"和主标题"{title}"，这两段文字必须原封不动出现，并且用双引号包围。
6. 输出文字总数量不超过250字。
7. 必须以"生成一张"为开头，必须包含"{layout}"这句话，这句话后面紧跟双引号包围的数据。

输出格式：
[生成一张xxx主题的{type_cn}图]，[详细描述背景、图表创意形态、元素等图像细节]，[{layout}"{data}"]，[xx位置和xx字体标题内容"{title}"]

参考示例，请勿重复：
{example}

现在请直接输出完整prompt段落，不要有任何额外说明。
""".strip()


def generate_prompt(title, data, chart_type, api_key, api_url, model_name):
    instruction = build_prompt_instruction(title=title, data=data, chart_type=chart_type)
    return vlm_llm(api_key=api_key, model_name=model_name, prompt=instruction, api_url=api_url)


def parse_chart_data(data_str):
    items = [item.strip() for item in re.split(r"[，,]", data_str) if item.strip()]
    x_values = []
    y_values = []

    for item in items:
        parts = item.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        label, value_text = parts[0].strip(), parts[1].strip()
        numeric_match = re.search(r"-?\d+(?:\.\d+)?", value_text)
        if not label or numeric_match is None:
            continue
        x_values.append(label)
        y_values.append(float(numeric_match.group()))

    if len(x_values) < 2:
        raise ValueError(
            "Failed to parse --data. Expected format like: "
            "\"草坪 35%，酒店 28%，海岛 20%\""
        )

    return {"X": x_values, "Y": y_values}


class ChartImageCreator:
    """Create gray.png and text.png for one chart sample."""

    def __init__(
        self,
        chart_type,
        resolution,
        font_path=None,
        font_size=20,
        dpi=100,
    ):
        self.chart_type = chart_type
        self.target_w, self.target_h = parse_resolution(resolution)
        self.canvas_w, self.canvas_h = self._get_canvas_size()
        self.font_path = font_path
        self.font_size = font_size
        self.dpi = dpi
        self._setup_chart_layout()

    def _get_canvas_size(self):
        if self.chart_type == "hbar" and self.target_w != self.target_h:
            return self.target_h, self.target_w
        return self.target_w, self.target_h

    def _setup_chart_layout(self):
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
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.font_manager as fm
        from scipy.interpolate import PchipInterpolator

        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return np, plt, mpatches, fm, PchipInterpolator

    def _font_properties(self, fm):
        if self.font_path and os.path.exists(self.font_path):
            return fm.FontProperties(fname=self.font_path)
        return None

    def _figure(self, plt, area=False):
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
    def _numeric_x(np, x_values):
        x_numeric = np.arange(len(x_values))
        x_map = dict(zip(x_numeric, x_values))
        return x_numeric, x_map

    @staticmethod
    def _rounded_bar(ax, mpatches, x, y, width, height, color, alpha=1.0):
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
        y_max_data = max(y_scaled)
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

    def _draw_bar_labels(self, ax, x, y_scaled, x_map, y_original, y_max, bottom_offset, font_kwargs):
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

    def _draw_area_labels(self, ax, x, y_scaled, x_map, y_original, y_max, font_kwargs):
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

    def _draw_pie(self, ax, np, y, x_map, with_text, font_kwargs):
        n = len(y)
        wedges, _ = ax.pie(
            y,
            labels=None,
            colors=["gray"] * n,
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

    def create(self, data, save_path, with_text=False):
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

        if self.chart_type == "hbar":
            image = Image.open(save_path)
            rotated = image.rotate(-90, expand=True)
            rotated.save(save_path)


def normalize_pie_values(data):
    total = sum(data["Y"])
    if total > 0 and abs(total - 100) > 1:
        normalized = [round(value / total * 100, 1) for value in data["Y"]]
        normalized[-1] = round(100 - sum(normalized[:-1]), 1)
        data = {"X": data["X"], "Y": normalized}
    return data


def generate_gray_and_text(data, chart_type, resolution, output_dir):
    if chart_type == "pie":
        data = normalize_pie_values(data)

    creator = ChartImageCreator(chart_type=chart_type, resolution=resolution)
    gray_path = output_dir / "gray.png"
    text_path = output_dir / "text.png"
    creator.create(data, gray_path, with_text=False)
    creator.create(data, text_path, with_text=True)

    return gray_path, text_path


def generate_artchart(args, prompt, gray_path, output_dir):
    #if args.grpo_lora_path and not args.lightning_lora_path:
    #    raise ValueError("--grpo_lora_path currently requires --lightning_lora_path.")

    #from generate_bench_artchart import ArtChartGenerator

    generator = ArtChartGenerator(
        controlnet_path=args.artchart_controlnet_path,
        base_model_path=args.base_model_path,
        lightning_lora_path=args.lightning_lora_path or None,
        grpo_lora_path=args.grpo_lora_path or None,
    )
    artchart_path = output_dir / "artchart.png"
    generator.generate(
        prompt=prompt,
        control_image_path=gray_path,
        save_path=artchart_path,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )
    return artchart_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a single ArtChart sample from title/data or a provided prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--title", type=str, default=None, help="Chart title. Ignored when --prompt is provided.")
    parser.add_argument("--data", type=str, required=True, help='Chart data, e.g. "草坪 35%%，酒店 28%%，海岛 20%%".')
    parser.add_argument("--prompt", type=str, default=None, help="Use this prompt directly and skip LLM expansion.")
    parser.add_argument("--chart_type", type=str, required=True, choices=["bar", "hbar", "hor_bar", "pie", "area"])
    parser.add_argument(
        "--resolution",
        type=str,
        default="1024x1024",
        help="Output resolution. Supported: 1024x1024/1024*1024/1024x768/1024*768.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for prompt.txt/gray.png/text.png/artchart.png.")

    parser.add_argument("--api_key", type=str, default=None, help="LLM API key. Falls back to VLM_API_KEY.")
    parser.add_argument("--api_url", type=str, default=None, help="LLM API URL. Falls back to VLM_API_URL.")
    parser.add_argument("--llm_model_name", type=str, default="Qwen3.5-397B-A17B")

    parser.add_argument("--base_model_path", type=str, required=True, help="Directory of the Qwen-Image base model.")
    parser.add_argument("--artchart_controlnet_path", type=str, required=True, help="Directory of the ArtChart ControlNet checkpoint.")
    parser.add_argument("--lightning_lora_path", type=str, default=None, help="Optional Lightning LoRA path.")
    parser.add_argument("--grpo_lora_path", type=str, default=None, help="Optional ArtChart GRPO LoRA path.")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.chart_type == "hor_bar":
        args.chart_type = "hbar"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = parse_chart_data(args.data)

    if args.prompt:
        prompt = args.prompt.strip()
    else:
        if not args.title:
            raise ValueError("--title is required when --prompt is not provided.")
        api_key = args.api_key or os.getenv("VLM_API_KEY")
        if not api_key:
            raise ValueError("LLM prompt expansion requires --api_key or VLM_API_KEY.")
        api_url = args.api_url or os.getenv("VLM_API_URL")
        if not api_url:
            raise ValueError("LLM prompt expansion requires --api_url or VLM_API_URL.")
        prompt = generate_prompt(
            title=args.title,
            data=args.data,
            chart_type=args.chart_type,
            api_key=api_key,
            api_url=api_url,
            model_name=args.llm_model_name,
        )

    prompt_path = output_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    gray_path, text_path = generate_gray_and_text(
        data=data,
        chart_type=args.chart_type,
        resolution=args.resolution,
        output_dir=output_dir,
    )
    artchart_path = generate_artchart(args=args, prompt=prompt, gray_path=gray_path, output_dir=output_dir)

    print("Single ArtChart generation complete:")
    print(f"  prompt  : {prompt_path}")
    print(f"  gray    : {gray_path}")
    print(f"  text    : {text_path}")
    print(f"  artchart: {artchart_path}")


if __name__ == "__main__":
    main()
