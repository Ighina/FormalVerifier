# Formal Verification Pipeline

A modular Python framework for converting natural language mathematical statements to Lean 4 formal proofs.

## Structure

The codebase is organized into the following modules:

### Core Modules

- **`config.py`**: Configuration classes for models and prompts
  - `ModelConfig`: Model settings (model_id, generation parameters)
  - `FormalizerConfig`: Statement formalization configuration
  - `ProverConfig`: Theorem proving configuration
  - Default configurations for both stages

- **`statement_formalizer.py`**: Natural language to Lean 4 conversion
  - `StatementFormalizer`: Converts informal statements to formal Lean 4 code
  - Configurable models and prompts
  - Automatic code extraction from model outputs

- **`prover.py`**: Lean 4 proof generation
  - `Prover`: Completes Lean 4 proofs given formal statements
  - Configurable models and prompts
  - Handles proof extraction and formatting

- **`pipeline.py`**: End-to-end verification pipeline
  - `FormalVerificationPipeline`: Combines formalization and proving
  - `PipelineResult`: Structured output with timing metadata
  - Methods for running full pipeline or individual stages

## Usage

### Quick Start (Default Configuration)

```python
from pipeline import FormalVerificationPipeline

pipeline = FormalVerificationPipeline()
result = pipeline.run(
    informal_statement="Prove that 3 cannot be written as the sum of two cubes.",
    problem_name="sum_of_two_cubes"
)

print(result.formal_statement)
print(result.proof)
print(f"Total time: {result.total_time:.2f}s")
```

### Custom Configuration

```python
from config import ModelConfig, FormalizerConfig, ProverConfig
from pipeline import FormalVerificationPipeline

# Customize formalizer
formalizer_config = FormalizerConfig(
    model_config=ModelConfig(
        model_id="Goedel-LM/Goedel-Formalizer-V2-32B",
        max_new_tokens=8192,
        temperature=0.7,
        do_sample=True,
        top_k=10,
        top_p=0.9
    ),
    prompt_template="Custom prompt: {problem_name}\n{informal_statement}"
)

# Customize prover
prover_config = ProverConfig(
    model_config=ModelConfig(
        model_id="Goedel-LM/Goedel-Prover-V2-32B",
        max_new_tokens=16384
    ),
    prompt_template="Prove this: {formal_statement}"
)

pipeline = FormalVerificationPipeline(
    formalizer_config=formalizer_config,
    prover_config=prover_config
)
```

### Running Stages Separately

```python
pipeline = FormalVerificationPipeline()

# Only formalize
formal_statement = pipeline.formalize_only(
    informal_statement="The square root of 2 is irrational.",
    problem_name="sqrt_two_irrational"
)

# Only prove (with existing formal statement)
proof = pipeline.prove_only(formal_statement=formal_statement)
```

### With Metadata

```python
result = pipeline.run(
    informal_statement="If x^2 = 4, then x = 2 or x = -2.",
    problem_name="square_root_theorem",
    return_full_outputs=True  # Include full model outputs
)

print(f"Formalization time: {result.formalization_time:.2f}s")
print(f"Proving time: {result.proving_time:.2f}s")
print(f"Full formalization output: {result.formalization_output}")
```

## Batch Processing

Process entire datasets through the verification pipeline:

```bash
# Process first 100 items from GSM8K training set
python main.py --dataset gsm8k --split train --start 0 --end 100

# Process full test set
python main.py --dataset gsm8k --split test

# Custom output directory and save frequency
python main.py --dataset gsm8k --output-dir my_results --save-frequency 5

# Stop on first error (default is to skip errors)
python main.py --dataset gsm8k --no-skip-errors
```

### Command Line Arguments

- `--dataset`: Dataset name (default: gsm8k)
- `--split`: Dataset split - train/test (default: train)
- `--start`: Starting index (default: 0)
- `--end`: Ending index (default: None, processes all)
- `--output-dir`: Output directory (default: output)
- `--problem-prefix`: Prefix for theorem names (default: problem)
- `--save-frequency`: Save every N items (default: 10)
- `--no-skip-errors`: Stop on first error

### Output Format

Results are saved in JSON format with the following structure:

```json
[
  {
    "id": 0,
    "problem": "Natural language problem statement",
    "solution": "Original solution from dataset",
    "formal_statement": "Lean 4 formal statement",
    "proof": "Lean 4 proof",
    "formalization_time": 12.5,
    "proving_time": 45.2,
    "total_time": 57.7
  }
]
```

Output files:
- `output/{dataset}_{timestamp}/results.json`: Successful results
- `output/{dataset}_{timestamp}/errors.json`: Failed items with error messages

## Examples

See `example_usage.py` for comprehensive examples including:
1. Default pipeline usage
2. Custom configurations
3. Running stages separately
4. Working with existing formal statements

Run the examples:
```bash
python example_usage.py
```

## Model Management

```python
# Preload models at initialization
pipeline = FormalVerificationPipeline(preload_models=True)

# Or load manually
pipeline.load_models()

# Free memory when done
pipeline.unload_models()
```

## Configuration Options

### ModelConfig Parameters
- `model_id`: HuggingFace model identifier
- `device_map`: Device mapping strategy (default: "auto")
- `torch_dtype`: Data type for model weights (default: "bfloat16")
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `do_sample`: Enable sampling
- `top_k`: Top-k sampling parameter
- `top_p`: Nucleus sampling parameter
- `seed`: Random seed for reproducibility

### Prompt Templates

Both `FormalizerConfig` and `ProverConfig` accept custom prompt templates using Python format strings:

**Formalizer template variables:**
- `{problem_name}`: Theorem name
- `{informal_statement}`: Natural language statement

**Prover template variables:**
- `{formal_statement}`: Lean 4 formal statement

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- CUDA-capable GPU (recommended)

## Original Files

The original example files are preserved:
- `prover_example.py`: Original proving example
- `statement_formalizer_example.py`: Original formalization example
