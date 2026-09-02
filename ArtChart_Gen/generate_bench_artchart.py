"""
A sample folder in the benchmark looks like:
    <bench_dir>/<chart_type>/<sample_id>/
        gray.png      # grayscale control image (sketch of the chart)
        prompt.txt     # text prompt describing the chart

Run from the command line, e.g.:
    python generate_controlnet_images.py \
        --base_model_path /path/to/base \
        --artchart_controlnet_path /path/to/controlnet \
        --artchart_bench_dir /path/to/benchmark \
        --output_dir eval_imgs/run \
        --lightning_lora_path /path/to/light \
        --grpo_lora_path /path/to/lora
"""

import argparse
import gc
import os

import torch
from PIL import Image
from tqdm import tqdm
from diffusers import (
    QwenImageControlNetModel,
    QwenImageControlNetPipeline,
    QwenImageTransformer2DModel,
)
# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

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

        has_lightning_lora = bool(lightning_lora_path)
        has_grpo_lora = bool(grpo_lora_path)

        if has_lightning_lora:
            self.pipe.load_lora_weights(lightning_lora_path, adapter_name="lightning")
            self.num_inference_steps=4
            self.cfg=1
        else:
            self.num_inference_steps=40
            self.cfg=3.5
        if has_grpo_lora:
            self.pipe.load_lora_weights(grpo_lora_path, adapter_name="grpo")
        if has_lightning_lora and has_grpo_lora:
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


# ---------------------------------------------------------------------------
# Per-checkpoint batch generation
# ---------------------------------------------------------------------------

def generate_for_checkpoint(
    checkpoint_path,
    output_base_dir,
    bench_dir,
    chart_types,
    base_model_path,
    lightning_lora_path,
    grpo_lora_path=None,
):
    """Generate evaluation images for one checkpoint across all chart types.

    Args:
        checkpoint_path: Directory of the ControlNet checkpoint to evaluate.
        output_base_dir: Root directory where generated images are written. One subfolder per chart type is created underneath it.
        bench_dir: Root directory of the benchmark dataset.
        chart_types: List of chart type folder names to process (e.g. ["pie", "bar", "hbar", "area"]).
        base_model_path: Directory of the Qwen-Image base model.
        lightning_lora_path: Path to the Lightning LoRA weights file.
        grpo_lora_path: Optional path to a GRPO LoRA weights file.
    """
    checkpoint_name = os.path.basename(checkpoint_path)
    print(f"\n{'=' * 60}")
    print(f"Processing checkpoint: {checkpoint_name}")
    print(f"{'=' * 60}")

    generator = ArtChartGenerator(
        controlnet_path=checkpoint_path,
        base_model_path=base_model_path,
        lightning_lora_path=lightning_lora_path,
        grpo_lora_path=grpo_lora_path,
    )

    # Process each chart type one at a time.
    for chart_type in chart_types:
        if chart_type == "hor_bar":
            chart_type = "hbar"
        chart_data_dir = os.path.join(bench_dir, chart_type)
        chart_output_dir = os.path.join(output_base_dir, chart_type)
        os.makedirs(chart_output_dir, exist_ok=True)

        sample_dirs = sorted(
            d for d in os.listdir(chart_data_dir)
            if os.path.isdir(os.path.join(chart_data_dir, d))
        )
        print(f"\nGenerating {chart_type} images ({len(sample_dirs)} samples)...")

        for sample_id in tqdm(sample_dirs, desc=chart_type):
            sample_dir = os.path.join(chart_data_dir, sample_id)
            gray_path = os.path.join(sample_dir, "gray.png")
            prompt_path = os.path.join(sample_dir, "prompt.txt")
            output_path = os.path.join(chart_output_dir, f"{sample_id}.png")

            # Skip samples we have already generated, so the run can resume.
            if os.path.exists(output_path):
                print(f"Skipping {sample_id}, already exists")
                continue

            if not os.path.exists(prompt_path):
                print(f"Warning: prompt.txt not found for {sample_id}")
                continue

            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

            try:
                generator.generate(prompt, gray_path, output_path)
            except Exception as e:
                print(f"Error generating {sample_id}: {e}")

        print(f"Finished {chart_type}: {chart_output_dir}")


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation images with an ArtChart Method.",
    )

    # --- What to evaluate -------------------------------------------------
    
    parser.add_argument(
        "--artchart_bench_dir", type=str, required=True,
        help="Root directory of the benchmark dataset. Expected layout: "
             "<bench_dir>/<chart_type>/<sample_id>/{gray.png, prompt.txt}.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="eval_imgs",
        help="Root directory where generated images are written.",
    )
    parser.add_argument(
        "--chart_types", type=str, nargs="+",
        default=["pie", "bar", "hbar", "area"],
        help="Chart type folder names to process. hor_bar is accepted as a legacy alias for hbar.",
    )

    # --- Model paths ------------------------------------------------------
    parser.add_argument(
        "--base_model_path", type=str,
        default="",
        help="Directory of the Qwen-Image base model.",
    )
    parser.add_argument(
        "--artchart_controlnet_path", type=str, required=True,
        help="Directory of the trained ControlNet checkpoint to evaluate.",
    )
    parser.add_argument(
        "--lightning_lora_path", type=str,
        default="",
        help="Path to the Lightning LoRA weights file (enables 4-step inference).",
    )
    parser.add_argument(
        "--grpo_lora_path", type=str, default=None,
        help="Optional path to a GRPO-trained LoRA weights file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("ControlNet evaluation image generation")
    print("=" * 60)
    print(f"ControlNet checkpoint: {args.artchart_controlnet_path}")
    print(f"Benchmark directory : {args.artchart_bench_dir}")
    print(f"Output directory    : {args.output_dir}")
    print(f"Base model          : {args.base_model_path}")
    print(f"Chart types         : {args.chart_types}")
    if args.grpo_lora_path:
        print(f"GRPO LoRA           : {args.grpo_lora_path}")
    print("=" * 60)

    if not os.path.exists(args.artchart_controlnet_path):
        print(f"Error: checkpoint not found: {args.artchart_controlnet_path}")
        return

    generate_for_checkpoint(
        checkpoint_path=args.artchart_controlnet_path,
        output_base_dir=args.output_dir,
        bench_dir=args.artchart_bench_dir,
        chart_types=args.chart_types,
        base_model_path=args.base_model_path,
        lightning_lora_path=args.lightning_lora_path,
        grpo_lora_path=args.grpo_lora_path,
    )

    print("\n" + "=" * 60)
    print("Image generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()