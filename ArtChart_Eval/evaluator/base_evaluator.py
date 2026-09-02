from abc import ABC, abstractmethod
import json
import os
import threading

from api import *
from tqdm import tqdm
from typing import List, Dict, Any, Union
from os import PathLike
from utils.visualization import analyze_by_task_type

class BaseEvaluator(ABC):
    """
    BaseEvaluator provides common dataset loading and utility methods for evaluation classes.
    Assumes a dataset in jsonl format, where each line is a dict with keys:
        input_image, output_image, instruction, lang, task_type
    """
    def __init__(self, api_handler: BaseAPIHandler):
        """
        Initialize BaseEvaluator with an API handler.
        Args:
            api_handler (BaseAPIHandler): Instance of a class that implements the BaseAPIHandler interface.
        """
        self.api_handler = api_handler

    @abstractmethod
    def evaluate_single(self, input_image: Union[str, PathLike], output_image: Union[str, PathLike], instruction: str, task_type: str, eval_type:str,**kwargs) -> Dict[str, Any]:
        """
        Evaluate a single image editing result and return scores.
        Args:
            input_image (str or PathLike): Path or object of the input (original) image.
            output_image (str or PathLike): Path or object of the output (edited) image.
            instruction (str): The editing instruction string.
            task_type (str): The type of editing task.
        Returns:
            dict: A dictionary with evaluation results.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _load_dataset(self, dataset_path: Union[str, PathLike]) -> List[Dict[str, Any]]:
        """
        Load the dataset from the jsonl file.
        Args:
            dataset_path (str or PathLike): Path to the jsonl file containing the evaluation dataset.
        Returns:
            List[Dict[str, Any]]: List of data entries, each a dict with keys: input_image, output_image, instruction, lang, task_type.
        Raises:
            AssertionError: If any entry is missing required keys.
        """
        import os
        data = []
        dataset_path = os.path.abspath(str(dataset_path))
        base_dir = os.path.dirname(dataset_path)
        #required_keys = ["input_image", "output_image", "instruction", "lang", "task_type"]
        required_keys = ["output_image", "instruction"]
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    entry = json.loads(line)
                    for key in required_keys:
                        assert key in entry, f"Missing key '{key}' in line {idx+1} of {dataset_path}"
                    #for img_key in ["input_image", "output_image"]:
                    for img_key in ["output_image"]:
                        img_path = entry[img_key]
                        if img_path and not os.path.isabs(img_path):
                            entry[img_key] = os.path.abspath(os.path.join(base_dir, img_path))
                    data.append(entry)
        return data

    def _log(self, msg, style="info", end="\n", flush=True):
        """
        Print a styled log message to the console, including current time.
        style: info, success, warning, error
        """
        import datetime
        color = {
            "info": "\033[36m",
            "success": "\033[32m",
            "warning": "\033[33m",
            "error": "\033[31m"
        }.get(style, "\033[0m")
        reset = "\033[0m"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{color}[{now}] {msg}{reset}", end=end, flush=flush)

    def run(self, data_path: Union[str, PathLike], output_path: Union[str, PathLike], n_process: int = 8, model_name: str = "Qwen3.5-397B-A17B", eval_type: str=None, **kwargs) -> None:
        """
        Run the evaluation process. Supports both batch and single inference.
        Output is a CSV file.
        """
        import csv

        dataset = self._load_dataset(data_path)
        total = len(dataset)
        results = [None] * total
        threads = []
        lock = threading.Lock()
        pbar = tqdm(total=total, desc="Evaluating")

        def worker(start_idx, end_idx):
            for idx in range(start_idx, end_idx):
                row = dataset[idx]
                row['eval_type'] = eval_type
                try:
                    eval_result = self.evaluate_single(model_name=model_name, **row)
                    merged = dict(row)
                    if eval_result:
                        merged.update(eval_result)
                    with lock:
                        results[idx] = merged
                        pbar.update(1)
                except Exception as e:
                    import traceback
                    err_str = str(e)
                    if ("openai.BadRequestError" in err_str and "data_inspection_failed" in err_str) or (hasattr(e, 'error') and getattr(e, 'error', None) and 'data_inspection_failed' in str(getattr(e, 'error'))):
                        self._log(f"Skip idx {idx} due to openai.BadRequestError: {err_str}", style="warning")
                    else:
                        self._log(f"Skip idx {idx} due to error: {traceback.format_exc()[-400:]}", style="warning")
                    with lock:
                        results[idx] = None
                        pbar.update(1)

        chunk_size = (total + n_process - 1) // n_process
        for i in range(n_process):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total)
            if start >= end:
                break
            t = threading.Thread(target=worker, args=(start, end))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        pbar.close()

        # Gather all keys for CSV header
        all_keys = set()
        for item in results:
            if item:
                all_keys.update(item.keys())
        all_keys = list(all_keys)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, mode='w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_keys)
            writer.writeheader()
            for item in results:
                if item:
                    writer.writerow(item)

        # Analyze results
        results = analyze_by_task_type(output_path)
        print("Task Type Statistics:")
        print("-" * 60)
        print(f"{'Task Type':<15} {'Success':<7} {'Failure':<8} {'Total':<8} {'Success Rate':<8}")
        print("-" * 60)

        for task_type, counts in results.items():
            success_rate = counts["Success"] / counts["Total"] * 100 if counts["Total"] > 0 else 0
            print(f"{task_type:<15} {counts['Success']:<10} {counts['Failure']:<10} {counts['Total']:<10} {success_rate:.2f}%")

        print("-" * 60)

        # Calculate overall totals
        total_success = sum(r["Success"] for r in results.values())
        total_fail = sum(r["Failure"] for r in results.values())
        total_all = total_success + total_fail
        total_success_rate = total_success / total_all * 100 if total_all > 0 else 0

        print(f"{'Total':<15} {total_success:<10} {total_fail:<10} {total_all:<10} {total_success_rate:.2f}%")
