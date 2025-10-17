from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
torch.manual_seed(30)

model_id = "Goedel-LM/Goedel-Formalizer-V2-32B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)


problem_name = "test_problem"
informal_statement_content = "Prove that 3 cannot be written as the sum of two cubes."


# Construct the prompt for the model
user_prompt_content = (
    f"Please autoformalize the following natural language problem statement in Lean 4. "
    f"Use the following theorem name: {problem_name}\n"
    f"The natural language statement is: \n"
    f"{informal_statement_content}"
    f"Think before you provide the lean statement."
)



chat = [
  {"role": "user", "content": user_prompt_content},
]

inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

import time
start = time.time()
outputs = model.generate(inputs, max_new_tokens=16384, temperature = 0.9, do_sample = True, top_k=20, top_p=0.95)


model_output_text = tokenizer.batch_decode(outputs)[0]

def extract_code(text_input):
    """Extracts the last Lean 4 code block from the model's output."""
    try:
        matches = re.findall(r'```lean4\n(.*?)\n```', text_input, re.DOTALL)
        return matches[-1].strip() if matches else "No Lean 4 code block found."
    except Exception:
        return "Error during code extraction."


extracted_code = extract_code(model_output_text)

print(time.time() - start)
print("output:\n", model_output_text)
print("lean4 statement:\n", extracted_code)