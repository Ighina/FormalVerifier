"""Theorem proving module for completing Lean 4 proofs."""

import re
import time
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ProverConfig, DEFAULT_PROVER_CONFIG


class Prover:
    """Completes Lean 4 formal proofs given formal statements."""

    def __init__(self, config: Optional[ProverConfig] = None):
        """
        Initialize the prover.

        Args:
            config: Configuration for the prover. Uses default if None.
        """
        self.config = config or DEFAULT_PROVER_CONFIG
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is not None:
            return  # Already loaded

        model_config = self.config.model_config

        # Set random seed if specified
        if model_config.seed is not None:
            torch.manual_seed(model_config.seed)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)

        # Convert torch_dtype string to actual dtype
        dtype = (
            getattr(torch, model_config.torch_dtype)
            if isinstance(model_config.torch_dtype, str)
            else model_config.torch_dtype
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_id,
            device_map=model_config.device_map,
            torch_dtype=dtype,
            trust_remote_code=model_config.trust_remote_code,
        )

    def prove(self, formal_statement: str, return_metadata: bool = False) -> str | dict:
        """
        Generate a proof for a Lean 4 formal statement.

        Args:
            formal_statement: The Lean 4 formal statement with 'sorry'.
            return_metadata: If True, return dict with metadata including timing.

        Returns:
            The completed proof, or dict with proof and metadata.
        """
        if self.model is None:
            self.load_model()

        # Construct the prompt
        prompt = self.config.prompt_template.format(formal_statement=formal_statement)

        # Prepare chat input
        chat = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Generate
        start_time = time.time()

        generation_kwargs = {
            "max_new_tokens": self.config.model_config.max_new_tokens,
            "use_cache": True,  # 🚀 crucial for speed
            "pad_token_id": self.model.config.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.config.model_config.temperature is not None:
            generation_kwargs["temperature"] = self.config.model_config.temperature
        if self.config.model_config.do_sample:
            generation_kwargs["do_sample"] = True
        if self.config.model_config.top_k is not None:
            generation_kwargs["top_k"] = self.config.model_config.top_k
        if self.config.model_config.top_p is not None:
            generation_kwargs["top_p"] = self.config.model_config.top_p

        if isinstance(inputs, torch.Tensor):
            attention_mask = torch.ones_like(inputs)
            outputs = self.model.generate(
                inputs, attention_mask=attention_mask, **generation_kwargs
            )
        else:
            outputs = self.model.generate(**inputs, **generation_kwargs)

        elapsed_time = time.time() - start_time

        # Decode output
        model_output_text = self.tokenizer.batch_decode(outputs)[0]

        # Extract the completed proof
        completed_proof = self._extract_proof(model_output_text)

        if return_metadata:
            return {
                "proof": completed_proof,
                "full_output": model_output_text,
                "time_seconds": elapsed_time,
            }

        return completed_proof

    def prove_batch(
        self, formal_statements: list[str], return_metadata: bool = False
    ) -> list[dict]:
        """
        Generate proofs for multiple Lean 4 formal statements in batch.
        """
        if self.model is None:
            self.load_model()

        # Prepare chat prompts
        chat_templates = [
            [
                {
                    "role": "user",
                    "content": self.config.prompt_template.format(
                        formal_statement=stmt
                    ),
                }
            ]
            for stmt in formal_statements
        ]

        # Batch tokenize efficiently
        inputs = self.tokenizer.apply_chat_template(
            chat_templates,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generation configuration
        gen_cfg = self.config.model_config
        generation_kwargs = {
            "max_new_tokens": gen_cfg.max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.model.config.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if gen_cfg.temperature is not None:
            generation_kwargs["temperature"] = gen_cfg.temperature
        if gen_cfg.do_sample:
            generation_kwargs["do_sample"] = True
        if gen_cfg.top_k is not None:
            generation_kwargs["top_k"] = gen_cfg.top_k
        if gen_cfg.top_p is not None:
            generation_kwargs["top_p"] = gen_cfg.top_p

        # Generate
        start_time = time.time()
        outputs = self.model.generate(inputs, **generation_kwargs)
        elapsed_time = time.time() - start_time

        # Decode outputs
        model_output_texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        # Extract proofs
        results = []
        for text in model_output_texts:
            proof = self._extract_proof(text)
            if return_metadata:
                results.append(
                    {"proof": proof, "full_output": text, "time_seconds": elapsed_time}
                )
            else:
                results.append(proof)

        return results

    @staticmethod
    def _extract_proof(text_input: str) -> str:
        """
        Extract the Lean 4 proof from the model's output.

        Args:
            text_input: The full model output text.

        Returns:
            The extracted Lean 4 proof or the full output if no code block found.
        """
        try:
            # Try to extract code blocks
            matches = re.findall(r"```lean4\n(.*?)\n```", text_input, re.DOTALL)
            if matches:
                return matches[-1].strip()

            # If no code block found, return the full output
            # (the model might have output the proof directly)
            return text_input
        except Exception:
            return text_input

    def unload_model(self):
        """Unload the model to free memory."""
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
