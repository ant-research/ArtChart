from utils.misc import *
from api import *
from PIL import Image, ImageDraw
from evaluator.base_evaluator import BaseEvaluator
from os import PathLike
from pathlib import Path
from prompts import prompts_pool
from prompts.prompts_pool import bar_eval_prompt, horbar_eval_prompt, area_eval_prompt, pie_eval_prompt, TEXT_POSITION_PROMPT_MAP
import math
import time
import ast
import tempfile
import json
from paddleocr import PaddleOCR
import numpy as np
import re


class TextEvaluator(BaseEvaluator):
    def __init__(self,
                 api_handler: BaseAPIHandler,
                 aes_predictor_path: Optional[Union[str, PathLike]] = None,
                 aes_encoder_path: Optional[Union[str, PathLike]] = None,
                 ocr_det_model_dir: Optional[Union[str, PathLike]] = None,
                 ocr_rec_model_dir: Optional[Union[str, PathLike]] = None,
                 device: str = "cuda",
                 aesthetic_method: str = "vlm"):
        """
        Initialize TextEvaluator with an API handler.
        Args:
            api_handler (BaseAPIHandler): API handler for making requests.
        """
        super().__init__(api_handler)

        self.aesthetic_method = aesthetic_method
        self.aes_predictor_path = aes_predictor_path
        self.aes_encoder_path = aes_encoder_path
        self.aes_model = None
        self.aes_preprocesser = None
        self.device = device
        if self.aesthetic_method == "aes":
            self._load_aes_model()

        ocr_kwargs = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if ocr_det_model_dir:
            ocr_kwargs["text_detection_model_dir"] = str(ocr_det_model_dir)
        if ocr_rec_model_dir:
            ocr_kwargs["text_recognition_model_dir"] = str(ocr_rec_model_dir)
        self.ocr = PaddleOCR(**ocr_kwargs)

    def _load_aes_model(self) -> None:
        if not self.aes_predictor_path or not self.aes_encoder_path:
            raise ValueError(
                "Local AES evaluation requires AES_PREDICTOR_PATH and AES_ENCODER_PATH "
                "in text_evaluator.py, or --aes-predictor-path and --aes-encoder-path."
            )
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        import torch

        self.aes_model, self.aes_preprocesser = convert_v2_5_from_siglip(
            predictor_name_or_path=self.aes_predictor_path,
            encoder_model_name=self.aes_encoder_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.aes_model = self.aes_model.to(torch.bfloat16).to(self.device)

    def evaluate_instruction_following(self, output_image: Union[str, PathLike], instruction: str,
                                       model_name: str = "Qwen3.5-397B-A17B", **kwargs) -> Dict[str, Any]:
        following_prompt = prepare_prompt(prompts_pool._Chart_instruction_follow, instruction=instruction)
        messages = self.api_handler.prepare_messages(image_links=[output_image], text_prompt=following_prompt,force_same_size=True)
        try:
            following_result = self.api_handler.submit_messages(messages, model_name=model_name)["response"]
            following_result = parse_llm_output(following_result)
        except Exception as e:
            following_result = None

        if isinstance(following_result, dict):
            following_score = min(following_result["score"]) if isinstance(following_result["score"], list) else following_result["score"]
        else:
            following_score = following_result

        return dict(
            following_score=following_score,
            following_reason=following_result["reasoning"] if isinstance(following_result, dict) else "",
        )

    def evaluate_readability(self, output_image: Union[str, PathLike], task_type: str,
                             model_name: str = "Qwen3.5-397B-A17B", **kwargs) -> Dict[str, Any]:
        readability_prompt = prepare_prompt(prompts_pool._Chart_Readability,instruction=task_type)
        messages = self.api_handler.prepare_messages(image_links=[output_image], text_prompt=readability_prompt,force_same_size=True)
        try:
            readability_result = self.api_handler.submit_messages(messages, model_name=model_name)["response"]
            readability_result = parse_llm_output(readability_result)
        except Exception as e:
            readability_result = None

        if isinstance(readability_result, dict):
            readability_score = min(readability_result["score"]) if isinstance(readability_result["score"], list) else readability_result["score"]
        else:
            readability_score = readability_result

        return dict(
            readability_score=readability_score,
            readability_reason=readability_result["reasoning"] if isinstance(readability_result, dict) else "",
        )

    def evaluate_aes(self, output_image: Union[str, PathLike], instruction: str = "",
                     model_name: str = "Qwen3.5-397B-A17B", **kwargs) -> Dict[str, Any]:
        if self.aesthetic_method != "aes":
            aesthetic_prompt = prepare_prompt(prompts_pool._Chart_Aesthetic, instruction=instruction)
            messages = self.api_handler.prepare_messages(image_links=[output_image], text_prompt=aesthetic_prompt,force_same_size=True)
            try:
                aesthetic_result = self.api_handler.submit_messages(messages, model_name=model_name)["response"]
                aesthetic_result = parse_llm_output(aesthetic_result)
            except Exception:
                aesthetic_result = None

            if isinstance(aesthetic_result, dict):
                aesthetic_score = min(aesthetic_result["score"]) if isinstance(aesthetic_result["score"], list) else aesthetic_result["score"]
            else:
                aesthetic_score = aesthetic_result

            return dict(
                aes_score=aesthetic_score,
                aes_reason=aesthetic_result["reasoning"] if isinstance(aesthetic_result, dict) else "",
            )

        import torch

        image = Image.open(output_image).convert("RGB")
        pixel_values = (
            self.aes_preprocesser(images=image, return_tensors="pt")
                .pixel_values.to(torch.bfloat16)
                .to(self.device)
        )

        with torch.inference_mode():
            score = self.aes_model(pixel_values).logits.squeeze().float().cpu().numpy()

        return dict(aes_score=float(score))


    def evaluate_text_position_vlm(self, output_image: Union[str, PathLike], instruction: str,
                                   task_type: str, model_name: str = "Qwen3.5-397B-A17B", **kwargs) -> Dict[str, Any]:
        """
        Evaluate text-position correctness in a chart with a VLM.

        Args:
            output_image: Path to the generated chart image.
            instruction: Generation instruction containing the reference answer in quotes.
            task_type: Chart type, supports bar, hbar, pie, and area.
            model_name: VLM model name.

        Returns:
            dict: Contains text_position_score and text_position_reason.
        """
        import re

        # Extract the first quoted segment from the instruction as the reference answer.
        pattern1 = r'"([^"]*)"'  # en Double quotes.
        pattern2 = r'“([^”]*)”'  # zh Double quotes.

        quoted_list1 = re.findall(pattern1, instruction)
        quoted_list2 = re.findall(pattern2, instruction)

        # Prefer the first pattern, then fall back to the second pattern.
        if quoted_list1:
            text = quoted_list1[0]
        elif quoted_list2:
            text = quoted_list2[0]
        else:
            return {
                "text_position_score": None,
                "text_position_reason": "无法从instruction中提取标准答案"
            }

        # Select the prompt builder for the chart type.
        prompt_func = TEXT_POSITION_PROMPT_MAP.get(task_type)
        if prompt_func is None:
            return {
                "text_position_score": None,
                "text_position_reason": f"不支持的图表类型: {task_type}，支持的类型: {list(TEXT_POSITION_PROMPT_MAP.keys())}"
            }

        # Build the evaluation prompt.
        eval_prompt = prompt_func(text)

        # Run VLM evaluation.
        messages = self.api_handler.prepare_messages(
            image_links=[output_image],
            text_prompt=eval_prompt,
            force_same_size=True
        )

        try:
            response = self.api_handler.submit_messages(messages, model_name=model_name)["response"]
            # Parse the JSON response.
            result = parse_llm_output(response)
            if isinstance(result, dict) and "score" in result:
                score = result["score"]
                reason = result.get("reason", "")
                return {
                    "text_position_score": float(score) if score is not None else None,
                    "text_position_reason": reason
                }
            else:
                # Try to parse the response directly as a score.
                try:
                    score = float(response.strip())
                    return {
                        "text_position_score": score,
                        "text_position_reason": ""
                    }
                except:
                    return {
                        "text_position_score": None,
                        "text_position_reason": f"无法解析VLM响应: {response[:200]}"
                    }
        except Exception as e:
            return {
                "text_position_score": None,
                "text_position_reason": f"VLM调用失败: {str(e)}"
            }



    def evaluate_ocr_text(self,output_image: Union[str, PathLike], instruction: str,eval_type:str,
                          title: Optional[str] = None, data_str: Optional[str] = None, **kwargs)-> Dict[str, Any]:
        def calculate_text_unit_f1_score(gt, pred, similarity_threshold=0.6):
            def normalize_text(text):
                """Normalize whitespace while preserving punctuation."""
                text = re.sub(r'[ \t]+', ' ', text)  # Collapse repeated spaces.
                text = re.sub(r'\n+', '\n', text)  # Collapse repeated newlines.
                return text.strip()

            def extract_phrases(text):
                """Extract a list of phrases."""
                # Split on line breaks and common separators.
                # lines = text.split('\n')
                # lines = text.split_multi(text, ['\n', ',','，', '、', ';', '|'])
                lines = re.split(r'[\n,，、;；:：。.|]', text)
                phrases = []
                for line in lines:
                    # Each line may contain multiple phrases separated by spaces.
                    parts = [p.strip() for p in line.split() if p.strip()]
                    phrases.extend(parts)
                return [p for p in phrases if len(p) > 0]

            def normalize_unit(text):
                text = str(text).strip().casefold()
                text = re.sub(r'\s+', '', text)
                return (
                    text.replace('（', '(')
                    .replace('）', ')')
                    .replace('％', '%')
                    .replace('，', ',')
                )

            def edit_distance(str1, str2):
                if not str1:
                    return len(str2)
                if not str2:
                    return len(str1)

                previous = list(range(len(str2) + 1))
                for i, char1 in enumerate(str1, 1):
                    current = [i]
                    for j, char2 in enumerate(str2, 1):
                        current.append(
                            min(
                                previous[j] + 1,
                                current[j - 1] + 1,
                                previous[j - 1] + (char1 != char2),
                            )
                        )
                    previous = current
                return previous[-1]

            def normalized_edit_similarity(str1, str2):
                str1 = normalize_unit(str1)
                str2 = normalize_unit(str2)
                max_len = max(len(str1), len(str2))
                if max_len == 0:
                    return 1.0
                return max(0.0, 1.0 - edit_distance(str1, str2) / max_len)

            def greedy_matching(weights):
                candidates = []
                for gt_idx, row in enumerate(weights):
                    for pred_idx, sim in enumerate(row):
                        if sim >= similarity_threshold:
                            candidates.append((sim, gt_idx, pred_idx))
                candidates.sort(reverse=True)

                matches = []
                used_gt = set()
                used_pred = set()
                for sim, gt_idx, pred_idx in candidates:
                    if gt_idx in used_gt or pred_idx in used_pred:
                        continue
                    matches.append((gt_idx, pred_idx, sim))
                    used_gt.add(gt_idx)
                    used_pred.add(pred_idx)
                return matches

            def max_weight_matching(weights):
                num_gt = len(weights)
                num_pred = len(weights[0]) if weights else 0
                if num_gt == 0 or num_pred == 0:
                    return []
                if min(num_gt, num_pred) > 18:
                    return greedy_matching(weights)

                if num_gt <= num_pred:
                    dp = {0: (0.0, [])}
                    for pred_idx in range(num_pred):
                        next_dp = dict(dp)
                        for mask, (score, matches) in dp.items():
                            for gt_idx in range(num_gt):
                                if mask & (1 << gt_idx):
                                    continue
                                sim = weights[gt_idx][pred_idx]
                                if sim < similarity_threshold:
                                    continue
                                new_mask = mask | (1 << gt_idx)
                                new_score = score + sim
                                if new_score > next_dp.get(new_mask, (-1.0, []))[0]:
                                    next_dp[new_mask] = (new_score, matches + [(gt_idx, pred_idx, sim)])
                        dp = next_dp
                    return max(dp.values(), key=lambda item: item[0])[1]

                dp = {0: (0.0, [])}
                for gt_idx in range(num_gt):
                    next_dp = dict(dp)
                    for mask, (score, matches) in dp.items():
                        for pred_idx in range(num_pred):
                            if mask & (1 << pred_idx):
                                continue
                            sim = weights[gt_idx][pred_idx]
                            if sim < similarity_threshold:
                                continue
                            new_mask = mask | (1 << pred_idx)
                            new_score = score + sim
                            if new_score > next_dp.get(new_mask, (-1.0, []))[0]:
                                next_dp[new_mask] = (new_score, matches + [(gt_idx, pred_idx, sim)])
                    dp = next_dp
                return max(dp.values(), key=lambda item: item[0])[1]

            gt_units = extract_phrases(normalize_text(gt))
            pred_units = extract_phrases(normalize_text(pred))
            details = {
                "gt_units": gt_units,
                "pred_units": pred_units,
                "threshold": similarity_threshold,
            }

            if not gt_units:
                details.update({"precision": None, "recall": None, "f1": None, "matches": []})
                return None, details
            if not pred_units:
                details.update({"precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": []})
                return 0.0, details

            weights = [
                [normalized_edit_similarity(gt_unit, pred_unit) for pred_unit in pred_units]
                for gt_unit in gt_units
            ]
            matches = max_weight_matching(weights)
            true_positive = sum(match[2] for match in matches)
            precision = true_positive / len(pred_units) if pred_units else 0.0
            recall = true_positive / len(gt_units) if gt_units else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            score = round(10 * f1, 2)

            details.update({
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "matches": [
                    {
                        "gt": gt_units[gt_idx],
                        "pred": pred_units[pred_idx],
                        "similarity": round(sim, 4),
                    }
                    for gt_idx, pred_idx, sim in matches
                ],
            })
            return score, details

        gt_parts = [str(item).strip() for item in (title, data_str) if item]
        if gt_parts:
            gt_text = ' '.join(gt_parts)
        else:
            pattern1 = r'“([^”]*)”'  # Chinese double quotes.
            quoted_list1 = re.findall(pattern1, instruction)
            pattern2 = r'"([^"]*)"'  # English double quotes.
            quoted_list2 = re.findall(pattern2, instruction)
            pattern3 = r'\'([^\']*)\''  # English single quotes.
            quoted_list3 = re.findall(pattern3, instruction)
            gt_text = ' '.join(quoted_list1 + quoted_list2 + quoted_list3)

        if eval_type=='ocr':
            try:
                result = self.ocr.predict(output_image)
                first = result[0] if result else {}
                rec_texts = first.get('rec_texts') if hasattr(first, 'get') else first['rec_texts']
                pred_recognized_text = ' '.join(rec_texts)
            except Exception:
                pred_recognized_text = ''

            ocr_global_score, details = calculate_text_unit_f1_score(gt_text, pred_recognized_text)
            ocr_text = str('gt_text: ' + gt_text + ';  ocr_pred_text: ' + pred_recognized_text)
            res = {'ocr_text_score': ocr_global_score, 'ocr_reason': ocr_text}
            return res

        else:# 'ocr_vl
            try:
                messages = self.api_handler.prepare_messages(image_links=[output_image], text_prompt='提取图像中的所有文字',force_same_size=True)
                paddleocr_vl_pred_text = self.api_handler.submit_messages(messages, model_name='PaddleOCR-VL')["response"]
            except:
                paddleocr_vl_pred_text=None

            if paddleocr_vl_pred_text is not None:
                ocrVL_global_score, details = calculate_text_unit_f1_score(gt_text, paddleocr_vl_pred_text)
                ocr_text = str('gt_text: ' + gt_text + ';  ocrVL_pred_text: ' + paddleocr_vl_pred_text)
            else:
                ocrVL_global_score = None
                ocr_text=str('gt_text: ' + gt_text + ';  ocrVL_pred_text: None' )

            res = {'ocr_text_score':ocrVL_global_score,'ocrvl_reason':ocr_text}

        return res

    def evaluate_math_vlm(self, output_image: Union[str, PathLike], instruction: str, task_type: str,
                          data: Optional[str] = None, model_name: str = "Qwen3.5-397B-A17B", **kwargs) -> Dict[str, Any]:
        prompt = prompts_pool.MATH_RATIO_PROMPT_MAP.get(task_type)
        if prompt is None:
            return {
                "math_score": None,
                "math_reason": f"Unsupported chart type for math_vlm: {task_type}.",
            }

        def extract_numeric_values_from_data_text(data_text: str):
            values = []
            for item in re.split(r'[\n,，、;；|]', str(data_text)):
                item = item.strip()
                if not item:
                    continue
                numbers = re.findall(r'-?\d+(?:\.\d+)?', item)
                if numbers:
                    # Use the last number to avoid category numbers such as Q1 or 2024.
                    values.append(float(numbers[-1]))
            return values

        def extract_gt_values_for_math():
            if data:
                if isinstance(data, list):
                    try:
                        values = [float(x) for x in data]
                    except Exception:
                        values = []
                    if len(values) >= 1:
                        return values, json.dumps(data, ensure_ascii=False)
                values = extract_numeric_values_from_data_text(data)
                if len(values) >= 1:
                    return values, str(data)

            quoted_segments = []
            quoted_segments.extend(re.findall(r'“([^”]*)”', instruction))
            quoted_segments.extend(re.findall(r'"([^"]*)"', instruction))
            quoted_segments.extend(re.findall(r"'([^']*)'", instruction))

            for segment in quoted_segments:
                values = extract_numeric_values_from_data_text(segment)
                if len(values) >= 1:
                    return values, segment

            values = extract_numeric_values_from_data_text(instruction)
            if len(values) >= 1:
                return values, instruction
            return [], ""

        def rect_from_ocr_box(box):
            try:
                arr = np.asarray(box, dtype=float)
            except Exception:
                return None

            if arr.size < 4:
                return None
            if arr.ndim == 1:
                if arr.size == 4:
                    x1, y1, x2, y2 = arr.tolist()
                    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                if arr.size >= 8:
                    arr = arr[:8].reshape(-1, 2)
            if arr.ndim >= 1 and arr.shape[-1] >= 1:
                points = arr.reshape(-1, arr.shape[-1])[:, :2]
                xs = points[:, 0]
                ys = points[:, 1]
                return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            return None

        def normalize_ocr_boxes(raw_boxes):
            boxes = []
            if raw_boxes is None:
                return boxes
            try:
                iterable = list(raw_boxes)
            except Exception:
                iterable = []
            for box in iterable:
                rect = rect_from_ocr_box(box)
                if rect is not None:
                    boxes.append(rect)
            return boxes

        def extract_ocr_boxes():
            boxes = []

            try:
                result = self.ocr.predict(str(output_image))
                if result:
                    first = result[0]
                    raw_boxes = first.get('rec_boxes') if hasattr(first, 'get') else first['rec_boxes']
                    boxes = normalize_ocr_boxes(raw_boxes)
                    if boxes:
                        return boxes
            except Exception:
                pass

            return boxes

        def mask_text_regions():
            image = Image.open(output_image).convert("RGB")
            draw = ImageDraw.Draw(image)
            width, height = image.size
            padding = max(2, int(round(max(width, height) * 0.003)))
            boxes = extract_ocr_boxes()

            for x1, y1, x2, y2 in boxes:
                left = max(0, int(math.floor(x1)) - padding)
                top = max(0, int(math.floor(y1)) - padding)
                right = min(width, int(math.ceil(x2)) + padding)
                bottom = min(height, int(math.ceil(y2)) + padding)
                draw.rectangle([left, top, right, bottom], fill=(255, 255, 255))

            output_path_str = str(output_image)
            output_path_parts = output_path_str.split('/')
            if len(output_path_parts) >= 3:
                output_path_parts[-3] = f"{output_path_parts[-3]}_mask"
                masked_path = Path('/'.join(output_path_parts))
            else:
                temp_file = tempfile.NamedTemporaryFile(prefix="artchart_math_masked_", suffix=".png", delete=False)
                masked_path = Path(temp_file.name)
                temp_file.close()

            masked_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(masked_path)
            return str(masked_path), len(boxes)

        def parse_vlm_ratio_list(response: str):
            if response is None:
                return []
            text = str(response).strip()
            match = re.search(r'\[[^\]]*\]', text, flags=re.S)
            candidate = match.group(0) if match else text

            parsed = None
            try:
                parsed = ast.literal_eval(candidate)
            except Exception:
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    parsed = None

            if isinstance(parsed, (int, float)):
                parsed = [parsed]
            if not isinstance(parsed, list):
                numbers = re.findall(r'-?\d+(?:\.\d+)?', candidate)
                parsed = [float(x) for x in numbers] if numbers else []

            ratios = []
            for item in parsed:
                try:
                    value = float(item)
                except Exception:
                    continue
                if math.isfinite(value):
                    ratios.append(value)
            return ratios

        def compute_math_score(gt_values, pred_values):
            result = {
                "math_score": None,
                "gt_norm": [],
                "pred_norm": [],
            }

            if len(gt_values) == 0:
                return result
            if len(pred_values) != len(gt_values):
                result["math_score"] = 0.0

            gt = np.asarray(gt_values, dtype=float)
            pred = np.asarray(pred_values, dtype=float)

            if task_type == 'pie':
                if np.sum(gt) > 0:
                    gt_ratios = gt / np.sum(gt) * 100
                    result["gt_norm"] = [round(float(x), 4) for x in gt_ratios]
                if len(pred) > 0 and np.sum(pred) > 0:
                    pred_ratios = pred / np.sum(pred) * 100
                    result["pred_norm"] = [round(float(x), 4) for x in pred_ratios]
                if len(pred_values) != len(gt_values) or np.sum(gt) <= 0 or np.sum(pred) <= 0:
                    result["math_score"] = 0.0
                    return result
                gt_ratios = gt / np.sum(gt) * 100
                pred_ratios = pred / np.sum(pred) * 100
                mae = float(np.mean(np.abs(pred_ratios - gt_ratios)) / 100)
                rank_tol = 5.0
            else:
                if len(gt) > 0 and np.max(gt) > 0:
                    gt_ratios = gt / np.max(gt)
                    result["gt_norm"] = [round(float(x), 4) for x in gt_ratios]
                if len(pred) > 0 and np.max(pred) > 0:
                    pred_ratios = np.clip(pred / np.max(pred), 0, 1)
                    result["pred_norm"] = [round(float(x), 4) for x in pred_ratios]
                if len(pred_values) != len(gt_values) or np.max(gt) <= 0 or np.max(pred) <= 0:
                    result["math_score"] = 0.0
                    return result
                gt_ratios = gt / np.max(gt)
                pred_ratios = pred / np.max(pred)
                pred_ratios = np.clip(pred_ratios, 0, 1)
                mae = float(np.mean(np.abs(pred_ratios - gt_ratios)))
                rank_tol = 0.05

            free_tol = 0.05
            zero_score_error = 0.45
            ratio_score = 10 * max(0, 1 - max(0, mae - free_tol) / zero_score_error)

            correct = 0
            total = 0
            for i in range(len(gt_ratios)):
                for j in range(i + 1, len(gt_ratios)):
                    if abs(gt_ratios[i] - gt_ratios[j]) <= rank_tol:
                        continue
                    total += 1
                    if (gt_ratios[i] - gt_ratios[j]) * (pred_ratios[i] - pred_ratios[j]) > 0:
                        correct += 1

            rank_score = 10.0 if total == 0 else 10.0 * correct / total
            math_score = 0.8 * ratio_score + 0.2 * rank_score

            result["math_score"] = round(float(math_score), 2)
            result["gt_norm"] = [round(float(x), 4) for x in gt_ratios]
            result["pred_norm"] = [round(float(x), 4) for x in pred_ratios]
            return result

        gt_values, _ = extract_gt_values_for_math()
        if len(gt_values) < 2:
            return {
                "math_score": None,
                "math_reason": json.dumps({
                    "gt-data-real": [],
                    "gt-data-nor": [],
                    "vlm-data-nor": [],
                    "masked-image": "",
                }, ensure_ascii=False),
            }

        masked_image, _ = mask_text_regions()
        messages = self.api_handler.prepare_messages(
            image_links=[masked_image],
            text_prompt=prompt,
            force_same_size=True,
        )

        try:
            response = self.api_handler.submit_messages(messages, model_name=model_name)["response"]
            pred_values = parse_vlm_ratio_list(response)
        except Exception:
            return {
                "math_score": None,
                "math_reason": json.dumps({
                    "gt-data-real": [float(x) for x in gt_values],
                    "gt-data-nor": [],
                    "vlm-data-nor": [],
                    "masked-image": masked_image,
                }, ensure_ascii=False),
            }

        score_details = compute_math_score(gt_values=gt_values, pred_values=pred_values)
        return {
            "math_score": score_details["math_score"],
            "math_reason": json.dumps({
                "gt-data-real": [float(x) for x in gt_values],
                "gt-data-nor": score_details["gt_norm"],
                "vlm-data-nor": score_details["pred_norm"],
                "masked-image": masked_image,
            }, ensure_ascii=False),
        }

    def evaluate_single(self, output_image: Union[str, PathLike], instruction: str,
                        task_type: str, eval_type: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a single image editing result and return scores.

        Args:
            input_img: Path or object of the input (original) image.
            output_img: Path or object of the output (edited) image.
            instruction: The editing instruction string.
            task_type: The type of editing task.

        Returns:
            dict: A dictionary with keys:
               - following_score: float, score for instruction following
               - following_reason: str, reasoning for following score
               - readability_score: float, score for chart readability
               - readability_reason: str, reasoning for readability score
        """
        task_type = "hbar" if task_type == "hor_bar" else task_type
        result = dict(task_type=task_type)
        if eval_type is None or eval_type=='None':
            eval_type=['instruction_following','readability','ocr','aes','text_position_vlm','math_vlm']# ,'ocr_vl',
        else:
            eval_type=[eval_type]

        if 'instruction_following' in eval_type:
            following_result = self.evaluate_instruction_following(
                output_image=output_image,
                instruction=instruction,
                **kwargs,
            )
            result.update(following_result)
        if 'readability' in eval_type:
            readability_result = self.evaluate_readability(
                output_image=output_image,
                task_type=task_type,
                **kwargs,
            )
            result.update(readability_result)
        if 'aes' in eval_type:
            # aes score
            aes_result = self.evaluate_aes(output_image=output_image, instruction=instruction, **kwargs)
            result.update(aes_result)
        if 'ocr' in eval_type:
            ocr_result = self.evaluate_ocr_text(
                output_image=output_image,
                instruction=instruction,
                eval_type='ocr',
                **kwargs,
            )
            result.update(ocr_result)
        if 'ocr_vl' in eval_type:
            ocr_result = self.evaluate_ocr_text(
                output_image=output_image,
                instruction=instruction,
                eval_type='ocr_vl',
                **kwargs,
            )
            result.update(ocr_result)
        if 'text_position_vlm' in eval_type:
            text_position_result = self.evaluate_text_position_vlm(
                output_image=output_image,
                instruction=instruction,
                task_type=task_type,
                **kwargs,
            )
            result.update(text_position_result)
        if 'math_vlm' in eval_type:
            math_result = self.evaluate_math_vlm(
                output_image=output_image,
                instruction=instruction,
                task_type=task_type,
                **kwargs,
            )
            result.update(math_result)
        return result
