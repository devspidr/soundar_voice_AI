# test_reasoning_agent.py
import os
from typing import Optional

# env-driven configs
MODEL_NAME = os.getenv("LOCAL_LLM_PATH", "gpt2")
MAX_NEW_TOKENS = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "50"))
TEMPERATURE = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.0"))
TOP_P = float(os.getenv("LOCAL_LLM_TOP_P", "0.95"))

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception as e:
    raise SystemExit(f"Missing packages: {e}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ReasoningAgent:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        print(f"Loading model {MODEL_NAME} on {DEVICE} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model.to(DEVICE)
        self.model.eval()
        print("Model loaded.")

    def generate_response(self, user_prompt: str, system_instructions: Optional[str] = None) -> str:
        prompt = (system_instructions.strip() + "\n\n") if system_instructions else ""
        prompt += f"User: {user_prompt}\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(DEVICE)
        input_len = input_ids.shape[-1]
        max_length = input_len + MAX_NEW_TOKENS
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=max_length, do_sample=(TEMPERATURE>0.0), temperature=TEMPERATURE, top_p=TOP_P, num_beams=1)
        generated_ids = outputs[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return text or "Error: model returned empty output."

if __name__ == "__main__":
    agent = ReasoningAgent()
    print("Response:", agent.generate_response("Hello! Introduce yourself in one short sentence."))
