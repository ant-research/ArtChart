import argparse
from evaluator import TextEvaluator
from api import *


def main():
    """
    Main function to parse command-line arguments and start the evaluation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="ArtChart Evaluation Pipeline Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the input data file to be evaluated (e.g., merged_output.jsonl)")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to the output CSV file for storing evaluation results")
    parser.add_argument("--api-key", type=str, required=True,
                        help="API key for the LLM provider")
    parser.add_argument("--api-url", type=str, required=True,
                        help="Chat completions API URL for the LLM provider")
    parser.add_argument("--n-process", type=int, default=8,
                        help="Number of concurrent processes/threads for evaluation")
    parser.add_argument("--model-name", type=str, default="qwen-vl-max-latest",
                        help="The name of the Large Language Model to use for scoring")
    parser.add_argument("--aesthetic-method", type=str, default="vlm", choices=["vlm", "aes"],
                        help="Use VLM aesthetic scoring by default, or local aesthetic_predictor_v2_5 when set to aes")
    parser.add_argument("--aes-predictor-path", type=str, default=None,
                        help="Optional path to the local aesthetic scoring predictor model")
    parser.add_argument("--aes-encoder-path", type=str, default=None,
                        help="Optional path to the local aesthetic scoring encoder model")
    parser.add_argument("--ocr-det-model-dir", type=str, default=None,
                        help="Path to the PaddleOCR v5 text detection model directory")
    parser.add_argument("--ocr-rec-model-dir", type=str, default=None,
                        help="Path to the PaddleOCR v5 text recognition model directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run local models on (e.g., 'cuda', 'cuda:0', 'cpu')")
    parser.add_argument("--api-handler-type", type=str, default="VLMAPIHandler",
                        help="Specify which API handler class to use (e.g., VLMAPIHandler)")
    parser.add_argument("--eval_type", type=str, default=None, help="The option includes instruction_following, readability, ocr, ocr_vl, aes, text_position_vlm, math_vlm, None. None means all default metrics participate in the evaluation")


    args = parser.parse_args()

    # Print the configuration
    print("="*50)
    print("Starting evaluation pipeline with the following configuration:")
    for k, v in vars(args).items():
        display_value = "***" if k == "api_key" else v
        print(f"  {k:<20}: {display_value}")
    print("="*50)

    # 1. Initialize the API Handler
    # In practice, you might need to pass keys or other credentials to the handler,
    # which can also be done via argparse.
    try:
        api_handler_cls = globals()[args.api_handler_type]
        api_handler = api_handler_cls(args.api_key, args.api_url)
    except KeyError:
        raise RuntimeError(f"[Error] API handler type '{args.api_handler_type}' is not defined. "
                           f"Available handler(s): {', '.join([k for k in globals() if k.endswith('APIHandler')])}")
    except Exception as e:
        raise RuntimeError(f"[Error] Failed to instantiate API handler '{args.api_handler_type}':\n{e}")

    # 2. Initialize the Evaluator
    try:
        evaluator = TextEvaluator(
            api_handler=api_handler,
            aes_predictor_path=args.aes_predictor_path,
            aes_encoder_path=args.aes_encoder_path,
            ocr_det_model_dir=args.ocr_det_model_dir,
            ocr_rec_model_dir=args.ocr_rec_model_dir,
            device=args.device,
            aesthetic_method=args.aesthetic_method
        )
    except Exception as e:
        print(f"\n[Error] Failed to initialize TextEvaluator: {e}")
        print("Please ensure your evaluator class (TextEvaluator) and its dependencies are correctly installed and imported.")
        return

    # 3. Run the evaluation
    print("\nExecuting evaluator.run()...")
    evaluator.run(
        data_path=args.data_path,
        output_path=args.output_path,
        n_process=args.n_process,
        model_name=args.model_name,
        eval_type=args.eval_type
    )
    print("\nEvaluation script has finished.")


if __name__ == "__main__":
    main()
