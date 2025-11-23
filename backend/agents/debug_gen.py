# backend/agents/debug_gen.py
import os
import time
import logging
from backend.agents.reasoning_agent import ReasoningAgent

# Ensure environment logging verbosity
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Optional conservative defaults for debug
os.environ.setdefault("LOCAL_LLM_MAX_NEW_TOKENS", "64")   # make generation short for CPU
os.environ.setdefault("LOCAL_LLM_TEMPERATURE", "0.0")    # deterministic, faster
os.environ.setdefault("LOCAL_LLM_TOP_P", "0.95")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("debug_gen")

def run_test(prompt="Tell me about yourself in one short sentence."):
    try:
        start = time.time()
        log.info("Instantiating ReasoningAgent...")
        agent = ReasoningAgent()
        load_time = time.time() - start
        log.info("Agent created. Model loaded? %s. load_time=%.2fs", bool(getattr(agent, "model", None)), load_time)

        if agent.model is None:
            log.error("Model is not loaded. Exiting.")
            return

        # run a single generation and measure time
        gen_start = time.time()
        log.info("Starting generation for prompt: %s", prompt)
        result = agent.test(prompt)
        gen_time = time.time() - gen_start
        log.info("Generation finished in %.2f s. Output: %s", gen_time, result)
    except Exception as e:
        log.exception("Exception during test run: %s", e)

if __name__ == "__main__":
    run_test()
