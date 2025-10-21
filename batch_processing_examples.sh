#!/bin/bash
# Batch Processing Examples for Formal Verification Pipeline

echo "=== Batch Processing Examples ==="
echo ""

# Example 1: Basic batch processing
echo "Example 1: Process first 10 items with default settings"
echo "Command: python main.py --dataset gsm8k --split train --start 0 --end 10"
echo ""

# Example 2: Process with custom batch size
echo "Example 2: Process 50 items with batch size of 5"
echo "Command: python main.py --dataset gsm8k --start 0 --end 50 --batch-size 5"
echo ""

# Example 3: Resume interrupted run
echo "Example 3: Resume from interrupted batch processing"
echo "Command: python main.py --dataset gsm8k --resume --output-dir output/gsm8k_20250121_143022"
echo ""

# Example 4: Process with custom save frequency
echo "Example 4: Process 100 items, save every 20 items"
echo "Command: python main.py --dataset gsm8k --start 0 --end 100 --save-frequency 20"
echo ""

# Example 5: Process full test set
echo "Example 5: Process entire test set"
echo "Command: python main.py --dataset gsm8k --split test"
echo ""

# Example 6: Custom output directory
echo "Example 6: Process with custom output directory"
echo "Command: python main.py --dataset gsm8k --start 0 --end 50 --output-dir my_experiments/run1"
echo ""

# Example 7: Stop on errors
echo "Example 7: Stop on first error (useful for debugging)"
echo "Command: python main.py --dataset gsm8k --start 0 --end 10 --no-skip-errors"
echo ""

echo "=== Additional Dataset Examples ==="
echo ""

# Example 8: Process different dataset
echo "Example 8: Process MATH dataset"
echo "Command: python main.py --dataset hendrycks/math --split train --start 0 --end 20"
echo ""

echo "=== Tips ==="
echo "1. Use --batch-size to process multiple items together (improves throughput)"
echo "2. Use --resume to continue interrupted runs (specify exact output directory)"
echo "3. Use --save-frequency to control how often results are saved"
echo "4. Use --no-skip-errors when debugging to stop on first error"
echo "5. Output files include results.json (successful) and errors.json (failed items)"
