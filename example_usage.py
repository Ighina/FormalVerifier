"""Example usage of the formal verification pipeline."""

from config import (
    ModelConfig,
    FormalizerConfig,
    ProverConfig,
    DEFAULT_FORMALIZER_CONFIG,
    DEFAULT_PROVER_CONFIG
)
from pipeline import FormalVerificationPipeline


def example_default_pipeline():
    """Example using the default configurations."""
    print("=" * 80)
    print("Example 1: Using default configurations")
    print("=" * 80)

    # Create pipeline with default settings
    pipeline = FormalVerificationPipeline()

    # Run the full pipeline
    informal_statement = "Prove that 3 cannot be written as the sum of two cubes."
    result = pipeline.run(
        informal_statement=informal_statement,
        problem_name="sum_of_two_cubes"
    )

    # Display results
    print(f"\nInformal Statement:\n{result.informal_statement}\n")
    print(f"Formal Statement:\n{result.formal_statement}\n")
    print(f"Proof:\n{result.proof}\n")
    print(f"Formalization Time: {result.formalization_time:.2f}s")
    print(f"Proving Time: {result.proving_time:.2f}s")
    print(f"Total Time: {result.total_time:.2f}s")


def example_custom_configurations():
    """Example using custom configurations."""
    print("\n" + "=" * 80)
    print("Example 2: Using custom configurations")
    print("=" * 80)

    # Custom formalizer configuration
    custom_formalizer_config = FormalizerConfig(
        model_config=ModelConfig(
            model_id="Goedel-LM/Goedel-Formalizer-V2-32B",
            max_new_tokens=8192,  # Reduced from default
            temperature=0.7,  # Lower temperature for more focused output
            do_sample=True,
            top_k=10,
            top_p=0.9,
            seed=42
        ),
        prompt_template=(
            "Formalize this statement in Lean 4 with theorem name '{problem_name}':\n"
            "{informal_statement}\n"
            "Provide only the Lean 4 code."
        )
    )

    # Custom prover configuration
    custom_prover_config = ProverConfig(
        model_config=ModelConfig(
            model_id="Goedel-LM/Goedel-Prover-V2-32B",
            max_new_tokens=16384,  # Reduced from default
            seed=42
        ),
        prompt_template=(
            "Complete this Lean 4 proof:\n\n"
            "```lean4\n{formal_statement}```\n\n"
            "Provide a complete proof."
        )
    )

    # Create pipeline with custom configurations
    pipeline = FormalVerificationPipeline(
        formalizer_config=custom_formalizer_config,
        prover_config=custom_prover_config
    )

    # Run the pipeline
    informal_statement = "If x^2 + y^2 = 2x - 4y - 5, then x + y = -1."
    result = pipeline.run(
        informal_statement=informal_statement,
        problem_name="square_equation_solution",
        return_full_outputs=True
    )

    # Display results
    print(f"\nInformal Statement:\n{result.informal_statement}\n")
    print(f"Formal Statement:\n{result.formal_statement}\n")
    print(f"Proof:\n{result.proof}\n")
    print(f"Total Time: {result.total_time:.2f}s")


def example_separate_stages():
    """Example running formalization and proving separately."""
    print("\n" + "=" * 80)
    print("Example 3: Running stages separately")
    print("=" * 80)

    pipeline = FormalVerificationPipeline()

    # Step 1: Formalize only
    informal_statement = "The square root of 2 is irrational."
    print(f"\nFormalizing: {informal_statement}")

    formal_statement = pipeline.formalize_only(
        informal_statement=informal_statement,
        problem_name="sqrt_two_irrational"
    )

    print(f"\nFormal Statement:\n{formal_statement}")

    # Step 2: Prove the formal statement
    print("\nGenerating proof...")

    proof = pipeline.prove_only(formal_statement=formal_statement)

    print(f"\nProof:\n{proof}")


def example_with_existing_formal_statement():
    """Example starting with an existing formal statement."""
    print("\n" + "=" * 80)
    print("Example 4: Proving an existing formal statement")
    print("=" * 80)

    pipeline = FormalVerificationPipeline()

    # Use a pre-formalized statement (like from prover_example.py)
    formal_statement = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat


theorem square_equation_solution {x y : ℝ} (h : x^2 + y^2 = 2*x - 4*y - 5) : x + y = -1 := by
  sorry
    """.strip()

    print(f"Formal Statement:\n{formal_statement}\n")
    print("Generating proof...")

    result = pipeline.prove_only(formal_statement=formal_statement, return_metadata=True)

    print(f"\nProof:\n{result['proof']}")
    print(f"\nProving Time: {result['time_seconds']:.2f}s")


if __name__ == "__main__":
    # Run example 1 (default pipeline)
    example_default_pipeline()

    # Uncomment to run other examples:
    # example_custom_configurations()
    # example_separate_stages()
    # example_with_existing_formal_statement()
