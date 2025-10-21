"""Main script for batch processing mathematical problems through the verification pipeline."""

import argparse
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from dataset_loader import DatasetLoader
from pipeline import FormalVerificationPipeline
from config import FormalizerConfig, ProverConfig


class VerificationRunner:
    """Runner for batch verification of mathematical problems."""

    def __init__(
        self,
        pipeline: FormalVerificationPipeline,
        output_dir: str = "output",
        save_frequency: int = 10,
        skip_on_error: bool = True,
        batch_size: int = 1,
        num_workers: int = 1,
        resume: bool = False,
    ):
        """
        Initialize the verification runner.

        Args:
            pipeline: The verification pipeline to use
            output_dir: Directory to save results
            save_frequency: Save results every N items
            skip_on_error: If True, skip items that cause errors
            batch_size: Number of items to process in each batch
            num_workers: Number of parallel workers (1 for sequential)
            resume: If True, resume from existing results
        """
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        self.skip_on_error = skip_on_error
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resume = resume
        self.results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.processed_ids: set = set()

        # Load existing results if resuming
        if self.resume:
            self._load_existing_results()

    def process_dataset(
        self,
        dataset_loader: DatasetLoader,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        problem_name_prefix: str = "problem",
    ) -> Dict[str, Any]:
        """
        Process a dataset through the verification pipeline.

        Args:
            dataset_loader: Loader for the dataset
            start_idx: Starting index
            end_idx: Ending index (None for full dataset)
            problem_name_prefix: Prefix for theorem names

        Returns:
            Summary statistics
        """
        # Load models once at the start
        print("Loading models...")
        self.pipeline.load_models()
        print("Models loaded successfully!\n")

        # Get dataset subset
        dataset_items = list(dataset_loader.get_subset(start_idx, end_idx))
        total_items = len(dataset_items)

        # Filter out already processed items if resuming
        skipped = 0
        if self.resume:
            original_count = len(dataset_items)
            dataset_items = [
                item for item in dataset_items if item["id"] not in self.processed_ids
            ]
            skipped = original_count - len(dataset_items)
            if skipped > 0:
                print(f"Resuming: skipping {skipped} already processed items")

        print(
            f"Processing {len(dataset_items)} items from {dataset_loader.dataset_name}..."
        )
        print(f"Results will be saved to: {self.output_dir}")
        if self.batch_size > 1:
            print(f"Batch size: {self.batch_size}")
        if self.num_workers > 1:
            print(f"Parallel workers: {self.num_workers}")
        print()

        # Process in batches
        successful = 0
        failed = 0
        processed_count_before = len(self.results)

        # Use batch processing from dataset loader
        if self.batch_size > 1:
            # Get batches from dataset
            batches = list(
                dataset_loader.get_batches(
                    batch_size=self.batch_size, start=start_idx, end=end_idx
                )
            )

            # Filter out already processed items from batches
            filtered_batches = []
            for batch in batches:
                filtered_batch = [
                    item for item in batch if item["id"] not in self.processed_ids
                ]
                if filtered_batch:
                    filtered_batches.append(filtered_batch)

            total_items_to_process = sum(len(batch) for batch in filtered_batches)

            with tqdm(
                total=total_items_to_process, desc="Processing", unit="problem"
            ) as pbar:
                for batch in filtered_batches:
                    # Process batch through pipeline
                    batch_results = self._process_batch(batch, problem_name_prefix)

                    for result in batch_results:
                        if result.get("error"):
                            failed += 1
                            self.errors.append(result)
                            self.processed_ids.add(result["id"])

                            if not self.skip_on_error:
                                print(
                                    f"\nError processing item {result['id']}: {result['error']}"
                                )
                                raise Exception(result["error"])
                        else:
                            successful += 1
                            self.results.append(result)
                            self.processed_ids.add(result["id"])

                    # Update progress bar once per batch
                    pbar.update(len(batch_results))

                    # Save periodically
                    if (
                        len(self.results) - processed_count_before
                    ) % self.save_frequency == 0:
                        self._save_results()
        else:
            # Single-item processing (original behavior)
            for item in tqdm(dataset_items, desc="Processing", unit="problem"):
                # Skip if already processed
                if item["id"] in self.processed_ids:
                    continue

                try:
                    result = self._process_item(item, problem_name_prefix)
                    self.results.append(result)
                    self.processed_ids.add(item["id"])
                    successful += 1

                    # Save periodically
                    if (
                        len(self.results) - processed_count_before
                    ) % self.save_frequency == 0:
                        self._save_results()

                except Exception as e:
                    failed += 1
                    error_info = {
                        "id": item["id"],
                        "problem": item["problem"],
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    self.errors.append(error_info)
                    self.processed_ids.add(item["id"])

                    if not self.skip_on_error:
                        print(f"\nError processing item {item['id']}: {e}")
                        raise

        # Final save
        self._save_results()
        self._save_errors()

        # Unload models
        print("\nUnloading models...")
        self.pipeline.unload_models()

        # Print summary
        summary = {
            "total_items": total_items,
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "success_rate": (
                successful / len(dataset_items) if len(dataset_items) > 0 else 0
            ),
            "output_file": str(self.output_dir / "results.json"),
        }

        self._print_summary(summary)
        return summary

    def _process_item(
        self, item: Dict[str, Any], problem_name_prefix: str
    ) -> Dict[str, Any]:
        """
        Process a single dataset item.

        Args:
            item: Dataset item with 'id', 'problem', 'solution'
            problem_name_prefix: Prefix for theorem name

        Returns:
            Result dictionary
        """
        problem_name = f"{problem_name_prefix}_{item['id']}"

        solution = item["solution"].split("####")[-1].strip()
        informal_statement = f"{item["problem"]} Show that the solution is {solution}."

        # Run the pipeline
        pipeline_result = self.pipeline.run(
            informal_statement=informal_statement,
            problem_name=problem_name,
            return_full_outputs=False,
        )

        # Construct result
        result = {
            "id": item["id"],
            "problem": item["problem"],
            "solution": item["solution"],
            "formal_statement": pipeline_result.formal_statement,
            "proof": pipeline_result.proof,
            "formalization_time": pipeline_result.formalization_time,
            "proving_time": pipeline_result.proving_time,
            "total_time": pipeline_result.total_time,
        }

        return result

    def _process_batch(
        self, batch: List[Dict[str, Any]], problem_name_prefix: str
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of dataset items.

        Args:
            batch: List of dataset items with 'id', 'problem', 'solution'
            problem_name_prefix: Prefix for theorem names

        Returns:
            List of result dictionaries (including errors)
        """
        results = []

        # Prepare batch inputs
        batch_inputs = []
        for item in batch:
            problem_name = f"{problem_name_prefix}_{item['id']}"
            solution = item["solution"].split("####")[-1].strip()
            informal_statement = (
                f"{item['problem']} Show that the solution is {solution}."
            )

            batch_inputs.append(
                {
                    "informal_statement": informal_statement,
                    "problem_name": problem_name,
                    "item": item,
                }
            )

        # Process batch through pipeline
        try:
            pipeline_results = self.pipeline.run_batch(
                batch_inputs=[
                    {
                        "informal_statement": inp["informal_statement"],
                        "problem_name": inp["problem_name"],
                    }
                    for inp in batch_inputs
                ],
                return_full_outputs=False,
            )

            # Construct results
            for inp, pipeline_result in zip(batch_inputs, pipeline_results):
                item = inp["item"]
                if isinstance(pipeline_result, dict) and pipeline_result.get("error"):
                    # Error occurred
                    results.append(
                        {
                            "id": item["id"],
                            "problem": item["problem"],
                            "error": pipeline_result["error"],
                            "error_type": pipeline_result.get("error_type", "Unknown"),
                        }
                    )
                else:
                    # Success
                    results.append(
                        {
                            "id": item["id"],
                            "problem": item["problem"],
                            "solution": item["solution"],
                            "formal_statement": pipeline_result.formal_statement,
                            "proof": pipeline_result.proof,
                            "formalization_time": pipeline_result.formalization_time,
                            "proving_time": pipeline_result.proving_time,
                            "total_time": pipeline_result.total_time,
                        }
                    )

        except Exception as e:
            print(e)
            0 / 0
            # If batch processing fails entirely, fall back to individual processing
            for inp in batch_inputs:
                item = inp["item"]
                try:
                    result = self._process_item(item, problem_name_prefix)
                    results.append(result)
                except Exception as item_error:
                    results.append(
                        {
                            "id": item["id"],
                            "problem": item["problem"],
                            "error": str(item_error),
                            "error_type": type(item_error).__name__,
                        }
                    )

        return results

    def _save_results(self):
        """Save results to JSON file."""
        output_file = self.output_dir / "results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def _save_errors(self):
        """Save errors to JSON file."""
        if not self.errors:
            return

        error_file = self.output_dir / "errors.json"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(self.errors, f, indent=2, ensure_ascii=False)

    def _print_summary(self, summary: Dict[str, Any]):
        """Print processing summary."""
        print("\n" + "=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Total items:     {summary['total_items']}")
        print(f"Successful:      {summary['successful']}")
        print(f"Failed:          {summary['failed']}")
        print(f"Success rate:    {summary['success_rate']:.2%}")
        print(f"Results saved to: {summary['output_file']}")
        if self.errors:
            print(f"Errors saved to:  {self.output_dir / 'errors.json'}")
        if self.resume and summary.get("skipped", 0) > 0:
            print(f"Skipped (already processed): {summary['skipped']}")
        print("=" * 80)

    def _load_existing_results(self):
        """Load existing results from previous runs for resumption."""
        results_file = self.output_dir / "results.json"
        errors_file = self.output_dir / "errors.json"

        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                self.results = json.load(f)
                self.processed_ids = {r["id"] for r in self.results}
            print(f"Loaded {len(self.results)} existing results for resumption.")

        if errors_file.exists():
            with open(errors_file, "r", encoding="utf-8") as f:
                self.errors = json.load(f)
                self.processed_ids.update({e["id"] for e in self.errors})
            print(f"Loaded {len(self.errors)} existing errors for resumption.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process mathematical problems through formal verification pipeline"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", help="Dataset name (default: gsm8k)"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Starting index (default: 0)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending index (default: None, process all)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results (default: output)",
    )
    parser.add_argument(
        "--problem-prefix",
        type=str,
        default="problem",
        help="Prefix for theorem names (default: problem)",
    )

    # Processing arguments
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=10,
        help="Save results every N items (default: 10)",
    )
    parser.add_argument(
        "--no-skip-errors",
        action="store_true",
        help="Stop on first error instead of skipping",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of items to process per batch (default: 1)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing (default: 1, sequential)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results in output directory",
    )

    args = parser.parse_args()

    # Create output directory with timestamp (unless resuming)
    if args.resume:
        # For resume, user should provide the exact output directory path
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            print(
                f"Error: Output directory '{output_dir}' does not exist. Cannot resume."
            )
            print("To resume, provide the exact path to the existing output directory.")
            return
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"{args.dataset}_{timestamp}")

    # Initialize components
    print("Initializing components...")
    dataset_loader = DatasetLoader(dataset_name=args.dataset, split=args.split)

    pipeline = FormalVerificationPipeline()

    runner = VerificationRunner(
        pipeline=pipeline,
        output_dir=output_dir,
        save_frequency=args.save_frequency,
        skip_on_error=not args.no_skip_errors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resume=args.resume,
    )

    # Run processing
    runner.process_dataset(
        dataset_loader=dataset_loader,
        start_idx=args.start,
        end_idx=args.end,
        problem_name_prefix=args.problem_prefix,
    )


if __name__ == "__main__":
    main()
