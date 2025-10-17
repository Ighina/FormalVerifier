"""Configuration classes for the formal verification pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a language model."""
    model_id: str
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    max_new_tokens: int = 16384
    temperature: Optional[float] = None
    do_sample: bool = False
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None


@dataclass
class FormalizerConfig:
    """Configuration for statement formalization."""
    model_config: ModelConfig
    prompt_template: str = (
        "Please autoformalize the following natural language problem statement in Lean 4. "
        "Use the following theorem name: {problem_name}\n"
        "The natural language statement is: \n"
        "{informal_statement}"
        "Think before you provide the lean statement."
    )


@dataclass
class ProverConfig:
    """Configuration for theorem proving."""
    model_config: ModelConfig
    prompt_template: str = (
        "Complete the following Lean 4 code:\n\n"
        "```lean4\n{formal_statement}```\n\n"
        "Before producing the Lean 4 code to formally prove the given theorem, "
        "provide a detailed proof plan outlining the main proof steps and strategies.\n"
        "The plan should highlight key ideas, intermediate lemmas, and proof structures "
        "that will guide the construction of the final formal proof."
    )


# Default configurations
DEFAULT_FORMALIZER_CONFIG = FormalizerConfig(
    model_config=ModelConfig(
        model_id="Goedel-LM/Goedel-Formalizer-V2-32B",
        max_new_tokens=16384,
        temperature=0.9,
        do_sample=True,
        top_k=20,
        top_p=0.95,
        seed=30
    )
)

DEFAULT_PROVER_CONFIG = ProverConfig(
    model_config=ModelConfig(
        model_id="Goedel-LM/Goedel-Prover-V2-32B",
        max_new_tokens=32768,
        seed=30
    )
)
