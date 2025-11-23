# backend/agents/reasoning_agent.py
import os
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # ensure logs are visible during dev

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

class ReasoningAgent:
    def __init__(self):
        log.info("Initializing ReasoningAgent (module: %s)", __name__)
        if OpenAI is None:
            log.error("OpenAI package not installed. Install it with `pip install openai`.")
            self.client = None
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            log.warning("OPENAI_API_KEY not set; LLM requests will fail without it.")
        try:
            # NOTE: adapt initialization to your openai SDK usage
            self.client = OpenAI(api_key=api_key)
            log.info("OpenAI client created successfully.")
        except Exception as e:
            log.exception("Failed to create OpenAI client: %s", e)
            self.client = None

        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate_response(self, user_prompt: str, system_instructions: str = None):
        """
        Primary method used by app.py. Returns the assistant reply as a string.
        """
        if self.client is None:
            return "Error: LLM client not configured. Please set OPENAI_API_KEY and install the openai package."

        # Compose messages: system instructions then user content
        messages = []
        if system_instructions:
            messages.append({"role": "system", "content": system_instructions})
        else:
            messages.append({"role": "system", "content": "You are a helpful assistant. Answer succinctly in first person."})

        messages.append({"role": "user", "content": user_prompt})

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.25,
                max_tokens=400,
            )

            # Multiple SDK shapes handled:
            if hasattr(completion, "choices") and len(completion.choices) > 0:
                choice = completion.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return choice.message.content.strip()
                if isinstance(choice, dict) and "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"].strip()

            text = getattr(completion, "text", None)
            if text:
                return text.strip()
            return "Error: unexpected response shape from LLM."
        except Exception as e:
            log.exception("LLM generate error: %s", e)
            return f"Error generating response: {e}"

    # Compatibility aliases: some code might call run() or get_response()
    def run(self, *args, **kwargs):
        return self.generate_response(*args, **kwargs)

    def get_response(self, *args, **kwargs):
        return self.generate_response(*args, **kwargs)


# import os
# import logging
# import google.generativeai as genai

# log = logging.getLogger(__name__)
# log.setLevel(logging.INFO)

# class ReasoningAgent:
#     def __init__(self):
#         api_key = os.getenv("GEMINI_API_KEY")
#         if not api_key:
#             log.error("GEMINI_API_KEY is not set! Please set it before running the server.")
#             self.model = None
#             return

#         try:
#             genai.configure(api_key=api_key)
#             # Use Gemini 1.5 Flash (fast + cheap + very good)
#             self.model = genai.GenerativeModel("gemini-1.5-flash")
#             log.info("Gemini model loaded successfully.")
#         except Exception as e:
#             log.exception("Failed to initialize Gemini: %s", e)
#             self.model = None

#     def generate_response(self, user_prompt: str, system_instructions: str = None):
#         if self.model is None:
#             return "Error: Gemini client is not configured."

#         final_prompt = ""
#         if system_instructions:
#             final_prompt += f"System Instructions:\n{system_instructions}\n\n"
#         final_prompt += f"User: {user_prompt}\nAssistant:"

#         try:
#             response = self.model.generate_content(final_prompt)
#             text = response.text.strip()
#             if text.lower().startswith("assistant:"):
#                 text = text[len("assistant:"):].strip()
#             return text
#         except Exception as e:
#             log.exception("Gemini generation error: %s", e)
#             return f"Error generating response from Gemini: {e}"

























# # backend/agents/reasoning_agent.py
# """
# ReasoningAgent using a Hugging Face causal LM (Qwen or other). Drop-in ready.
# Features:
# - Uses max_new_tokens (safer than max_length).
# - Truncates long prompts to fit model's position embeddings.
# - Sends tensors to correct device.
# - Optional GenerationConfig when available.
# - Anti-repetition knobs and post-processing cleanup.
# Tune via env vars:
#   LOCAL_LLM_PATH (default: "Qwen/Qwen2.5-1.5B-Instruct")
#   LOCAL_LLM_MAX_NEW_TOKENS (default: 150)
#   LOCAL_LLM_TEMPERATURE (default: 0.0)
#   LOCAL_LLM_TOP_P (default: 0.95)
#   LOCAL_LLM_REPETITION_PENALTY (default: 1.15)
#   LOCAL_LLM_NO_REPEAT_NGRAM_SIZE (default: 3)
#   LOCAL_LLM_LENGTH_PENALTY (default: 1.0)
# """
# import os
# import logging
# import re
# import io
# import contextlib
# from typing import Optional

# log = logging.getLogger(__name__)
# log.setLevel(logging.INFO)
# if not logging.getLogger().handlers:
#     ch = logging.StreamHandler()
#     ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
#     log.addHandler(ch)

# try:
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     try:
#         from transformers import GenerationConfig
#     except Exception:
#         GenerationConfig = None
#     import torch
# except Exception as e:
#     log.exception("transformers/torch import failed: %s", e)
#     AutoTokenizer = None
#     AutoModelForCausalLM = None
#     GenerationConfig = None
#     torch = None

# # Configuration (env-driven)
# MODEL_NAME = os.getenv("LOCAL_LLM_PATH", "Qwen/Qwen2.5-1.5B-Instruct")
# MAX_NEW_TOKENS = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "150"))
# TEMPERATURE = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.0"))
# TOP_P = float(os.getenv("LOCAL_LLM_TOP_P", "0.95"))
# DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"


# class ReasoningAgent:
#     def __init__(self):
#         self.model = None
#         self.tokenizer = None

#         if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
#             log.error("Missing dependencies: please pip install transformers and torch.")
#             return

#         # Load model/tokenizer while suppressing noisy stdout/stderr from model repo code
#         try:
#             log.info("Loading model %s on device %s ...", MODEL_NAME, DEVICE)
#             with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
#                 self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
#                 self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
#             self.model.to(DEVICE)
#             self.model.eval()
#             log.info("Model loaded successfully.")
#         except Exception as e:
#             log.exception("Failed to load model %s: %s", MODEL_NAME, e)
#             self.model = None

#     def _build_prompt(self, user_prompt: str, system_instructions: Optional[str]) -> str:
#         sys_block = (system_instructions.strip() + "\n\n") if system_instructions else ""
#         prompt = f"{sys_block}User: {user_prompt.strip()}\nAssistant:"
#         if len(prompt) > 20000:
#             prompt = prompt[-20000:]
#         return prompt

#     def _collapse_repeats(self, s: str) -> str:
#         s = re.sub(r'(\b[\w\W]{20,200}?\b)(\s+\1\s+)+', r'\1 ', s, flags=re.IGNORECASE)
#         s = re.sub(r'(\b\w+\b)(\s+\1){4,}', r'\1', s, flags=re.IGNORECASE)
#         return s.strip()

#     def _trim_to_sentences(self, s: str, max_sentences: int = 3) -> str:
#         parts = re.split(r'(?<=[.!?])\s+', s)
#         if len(parts) > max_sentences:
#             return " ".join(parts[:max_sentences]).strip()
#         return s.strip()

#     def generate_response(self, user_prompt: str, system_instructions: Optional[str] = None) -> str:
#         if self.model is None or self.tokenizer is None:
#             return "Error: No local model loaded. Install transformers and set LOCAL_LLM_PATH."

#         # Sanitize system_instructions so accidental logs aren't passed
#         if system_instructions:
#             sys_lines = []
#             for ln in system_instructions.splitlines():
#                 if ln.strip().startswith(("INFO:", "DEBUG:", "WARNING:", "ERROR:", "Traceback", "[")):
#                     continue
#                 sys_lines.append(ln)
#             system_instructions = "\n".join(sys_lines).strip()

#         prompt = self._build_prompt(user_prompt, system_instructions)

#         try:
#             inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
#             input_ids = inputs["input_ids"]
#             attention_mask = inputs.get("attention_mask", None)

#             # Device placement
#             input_ids = input_ids.to(DEVICE)
#             if attention_mask is not None:
#                 attention_mask = attention_mask.to(DEVICE)

#             input_len = input_ids.shape[-1]

#             # Respect model position limit
#             model_max_pos = getattr(self.model.config, "max_position_embeddings", 4096)
#             if input_len + int(MAX_NEW_TOKENS) > model_max_pos:
#                 keep = max(1, model_max_pos - int(MAX_NEW_TOKENS) - 1)
#                 input_ids = input_ids[:, -keep:].to(DEVICE)
#                 if attention_mask is not None:
#                     attention_mask = attention_mask[:, -keep:].to(DEVICE)
#                 input_len = input_ids.shape[-1]

#             pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None) or 0
#             do_sample = float(TEMPERATURE) > 0.0

#             repetition_penalty = float(os.getenv("LOCAL_LLM_REPETITION_PENALTY", "1.15"))
#             no_repeat_ngram_size = int(os.getenv("LOCAL_LLM_NO_REPEAT_NGRAM_SIZE", "3"))
#             length_penalty = float(os.getenv("LOCAL_LLM_LENGTH_PENALTY", "1.0"))

#             gen_kwargs = {
#                 "input_ids": input_ids,
#                 "attention_mask": attention_mask,
#                 "max_new_tokens": int(MAX_NEW_TOKENS),
#                 "do_sample": do_sample,
#                 "pad_token_id": int(pad_token_id),
#                 "repetition_penalty": repetition_penalty,
#                 "no_repeat_ngram_size": no_repeat_ngram_size,
#                 "length_penalty": length_penalty,
#             }

#             if do_sample:
#                 gen_kwargs.update({"temperature": float(TEMPERATURE), "top_p": float(TOP_P)})
#             else:
#                 gen_kwargs.update({"num_beams": 1, "do_sample": False})

#             # Block common log-like phrases from being generated
#             bad_words = []
#             for phrase in ["Could not locate the custom_generate", "Could not locate", "Model loaded successfully", "INFO:", "ERROR:", "Traceback", "Started server process"]:
#                 try:
#                     token_ids = self.tokenizer(phrase, add_special_tokens=False)["input_ids"]
#                     if token_ids:
#                         bad_words.append(token_ids)
#                 except Exception:
#                     pass
#             if bad_words:
#                 gen_kwargs["bad_words_ids"] = bad_words

#             # Remove None entries
#             gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

#             log.info("Generation start (sanitized prompt). kwargs keys: %s", list(gen_kwargs.keys()))

#             # Run generation while suppressing any stdout/stderr prints from model code
#             with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
#                 with torch.no_grad():
#                     outputs = self.model.generate(**gen_kwargs)

#             # Extract generated tokens and decode
#             generated_ids = outputs[0][input_len:]
#             text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

#             # Fallback: decode full and strip prompt if needed
#             if not text:
#                 full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#                 prompt_decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
#                 if full_text.startswith(prompt_decoded):
#                     text = full_text[len(prompt_decoded):].strip()
#                 else:
#                     text = full_text

#             # SAFETY: remove that specific model debug line if present
#             text = re.sub(r'Could not locate the custom_generate[^\n\r]*', '', text, flags=re.IGNORECASE)

#             # Remove lines that look like logs/traces
#             clean_lines = []
#             for ln in text.splitlines():
#                 if ln.strip() == "":
#                     continue
#                 if ln.strip().startswith(("INFO:", "DEBUG:", "WARNING:", "ERROR:", "Traceback", "[")):
#                     continue
#                 if "Started server process" in ln or "Model loaded successfully" in ln or "Could not locate" in ln:
#                     continue
#                 clean_lines.append(ln.strip())
#             text = " ".join(clean_lines).strip()

#             # Collapse repeated phrases (simple heuristic)
#             text = re.sub(r'(\b[\w\W]{10,200}?\b)(\s+\1\s+)+', r'\1 ', text, flags=re.IGNORECASE)
#             text = re.sub(r'(\b\w+\b)(\s+\1){4,}', r'\1', text, flags=re.IGNORECASE)

#             # Trim to a few sentences for concise persona responses
#             sentences = re.split(r'(?<=[.!?])\s+', text)
#             if len(sentences) > 3:
#                 text = " ".join(sentences[:3]).strip()

#             if text.lower().startswith("assistant:"):
#                 text = text[len("assistant:"):].strip()

#             if not text:
#                 return "Error: model returned empty output. Check generation kwargs in logs."

#             return text

#         except Exception as e:
#             log.exception("Error during generation (post-filter): %s", e)
#             return f"Error generating response (local qwen): {e}"

#     def test(self, prompt: str = "Hello, what's your name?"):
#         log.info("Running quick test with prompt: %s", prompt)
#         out = self.generate_response(prompt)
#         log.info("Test output: %s", out)
#         return out


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     agent = ReasoningAgent()
#     if agent.model is None:
#         print("Model not loaded. Check logs.")
#     else:
#         print("Test generation:")
#         print(agent.test("Hello! Please introduce yourself in one short sentence."))























# # backend/agents/reasoning_agent.py

# import os
# import logging
# import requests

# log = logging.getLogger(__name__)

# class ReasoningAgent:
#     def __init__(self):
#         self.api_token = os.getenv("HF_TOKEN")
#         self.model_id = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

#         if not self.api_token:
#             raise ValueError("Missing HF_TOKEN env variable")

#         self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
#         self.headers = {
#             "Authorization": f"Bearer {self.api_token}",
#             "Content-Type": "application/json"
#         }

#     def run(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2):
#         """Send prompt to HF Inference API"""

#         payload = {
#             "inputs": prompt,
#             "parameters": {
#                 "max_new_tokens": max_tokens,
#                 "temperature": temperature,
#                 "repetition_penalty": 1.1
#             }
#         }

#         try:
#             response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)

#             if response.status_code != 200:
#                 log.error(f"[HF ERROR] {response.status_code}: {response.text}")
#                 return f"Error: {response.text}"

#             data = response.json()

#             # HF returns list of dicts -> extract text
#             if isinstance(data, list) and "generated_text" in data[0]:
#                 return data[0]["generated_text"]

#             # Some models return other formats
#             return str(data)

#         except Exception as e:
#             log.exception("Error while calling HF API")
#             return f"Exception: {str(e)}"
