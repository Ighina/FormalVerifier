"""Statement formalization module for converting natural language to Lean 4."""

import re
import time
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import FormalizerConfig, DEFAULT_FORMALIZER_CONFIG


class StatementFormalizer:
    """Converts natural language problem statements to Lean 4 formal statements."""

    def __init__(self, config: Optional[FormalizerConfig] = None):
        """
        Initialize the statement formalizer.

        Args:
            config: Configuration for the formalizer. Uses default if None.
        """
        self.config = config or DEFAULT_FORMALIZER_CONFIG
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

    def formalize(
        self,
        informal_statement: str,
        problem_name: str = "theorem",
        return_metadata: bool = False,
    ) -> str | dict:
        """
        Convert a natural language statement to Lean 4 formal statement.

        Args:
            informal_statement: The natural language problem statement.
            problem_name: Name to use for the theorem.
            return_metadata: If True, return dict with metadata including timing.

        Returns:
            The formalized Lean 4 statement, or dict with statement and metadata.
        """
        if self.model is None:
            self.load_model()

        # Construct the prompt
        prompt = self.config.prompt_template.format(
            problem_name=problem_name, informal_statement=informal_statement
        )

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

        # Extract Lean 4 code
        formal_statement = self._extract_code(model_output_text)

        if return_metadata:
            return {
                "formal_statement": formal_statement,
                "full_output": model_output_text,
                "time_seconds": elapsed_time,
            }

        return formal_statement

    def formalize_batch(
        self, batch_inputs: list[dict], return_metadata: bool = False
    ) -> list[dict]:
        """
        Convert multiple natural language statements to Lean 4 formal statements in batch.

        Args:
            batch_inputs: List of dicts with 'informal_statement' and 'problem_name'.
            return_metadata: If True, return dict with metadata including timing.

        Returns:
            List of dicts with 'formal_statement', 'full_output', and 'time_seconds'.
        """
        if self.model is None:
            self.load_model()

        # Construct prompts for all inputs
        prompts = []
        for inp in batch_inputs:
            prompt = self.config.prompt_template.format(
                problem_name=inp["problem_name"],
                informal_statement=inp["informal_statement"],
            )
            prompts.append(prompt)

        # Prepare batch chat inputs
        chats = [[{"role": "user", "content": prompt}] for prompt in prompts]

        # Batch tokenize efficiently
        inputs = self.tokenizer.apply_chat_template(
            chats,
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
            formal_statements = self._extract_code(text)
            if return_metadata:
                results.append(
                    {
                        "formal_statement": formal_statements,
                        "full_output": text,
                        "time_seconds": elapsed_time,
                    }
                )
            else:
                results.append(formal_statements)

        return results

    @staticmethod
    def _extract_code(text_input: str) -> str:
        """
        Extract the last Lean 4 code block from the model's output.

        Args:
            text_input: The full model output text.

        Returns:
            The extracted Lean 4 code or error message.
        """
        try:
            matches = re.findall(r"```lean4\n(.*?)\n```", text_input, re.DOTALL)
            return matches[-1].strip() if matches else "No Lean 4 code block found."
        except Exception:
            return "Error during code extraction."

    def unload_model(self):
        """Unload the model to free memory."""
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
