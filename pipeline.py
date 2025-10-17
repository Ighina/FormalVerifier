"""End-to-end pipeline for formal verification from natural language to proof."""

from typing import Optional
from dataclasses import dataclass

from config import FormalizerConfig, ProverConfig, DEFAULT_FORMALIZER_CONFIG, DEFAULT_PROVER_CONFIG
from statement_formalizer import StatementFormalizer
from prover import Prover


@dataclass
class PipelineResult:
    """Result from the verification pipeline."""
    informal_statement: str
    formal_statement: str
    proof: str
    formalization_time: float
    proving_time: float
    total_time: float
    formalization_output: Optional[str] = None
    proving_output: Optional[str] = None


class FormalVerificationPipeline:
    """
    End-to-end pipeline for formal verification.

    Takes a natural language statement and produces a Lean 4 proof.
    """

    def __init__(
        self,
        formalizer_config: Optional[FormalizerConfig] = None,
        prover_config: Optional[ProverConfig] = None,
        preload_models: bool = False
    ):
        """
        Initialize the verification pipeline.

        Args:
            formalizer_config: Configuration for statement formalization.
            prover_config: Configuration for theorem proving.
            preload_models: If True, load models immediately.
        """
        self.formalizer = StatementFormalizer(formalizer_config or DEFAULT_FORMALIZER_CONFIG)
        self.prover = Prover(prover_config or DEFAULT_PROVER_CONFIG)

        if preload_models:
            self.load_models()

    def load_models(self):
        """Load both the formalizer and prover models."""
        self.formalizer.load_model()
        self.prover.load_model()

    def unload_models(self):
        """Unload both models to free memory."""
        self.formalizer.unload_model()
        self.prover.unload_model()

    def run(
        self,
        informal_statement: str,
        problem_name: str = "theorem",
        return_full_outputs: bool = False
    ) -> PipelineResult:
        """
        Run the full verification pipeline.

        Args:
            informal_statement: Natural language problem statement.
            problem_name: Name to use for the theorem.
            return_full_outputs: If True, include full model outputs in result.

        Returns:
            PipelineResult containing the proof and metadata.
        """
        # Step 1: Formalize the statement
        formalization_result = self.formalizer.formalize(
            informal_statement=informal_statement,
            problem_name=problem_name,
            return_metadata=True
        )

        formal_statement = formalization_result["formal_statement"]
        formalization_time = formalization_result["time_seconds"]
        formalization_output = formalization_result["full_output"] if return_full_outputs else None

        # Step 2: Prove the formal statement
        proving_result = self.prover.prove(
            formal_statement=formal_statement,
            return_metadata=True
        )

        proof = proving_result["proof"]
        proving_time = proving_result["time_seconds"]
        proving_output = proving_result["full_output"] if return_full_outputs else None

        # Combine results
        return PipelineResult(
            informal_statement=informal_statement,
            formal_statement=formal_statement,
            proof=proof,
            formalization_time=formalization_time,
            proving_time=proving_time,
            total_time=formalization_time + proving_time,
            formalization_output=formalization_output,
            proving_output=proving_output
        )

    def formalize_only(
        self,
        informal_statement: str,
        problem_name: str = "theorem",
        return_metadata: bool = False
    ) -> str | dict:
        """
        Run only the formalization step.

        Args:
            informal_statement: Natural language problem statement.
            problem_name: Name to use for the theorem.
            return_metadata: If True, return dict with metadata.

        Returns:
            The formalized statement, or dict with metadata.
        """
        return self.formalizer.formalize(
            informal_statement=informal_statement,
            problem_name=problem_name,
            return_metadata=return_metadata
        )

    def prove_only(
        self,
        formal_statement: str,
        return_metadata: bool = False
    ) -> str | dict:
        """
        Run only the proving step.

        Args:
            formal_statement: The Lean 4 formal statement.
            return_metadata: If True, return dict with metadata.

        Returns:
            The completed proof, or dict with metadata.
        """
        return self.prover.prove(
            formal_statement=formal_statement,
            return_metadata=return_metadata
        )
