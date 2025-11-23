# backend/agents/run_with_persona.py
from backend.agents.personality_agent import PersonalityAgent
from backend.agents.retrieval_agent import RetrievalAgent
from backend.agents.reasoning_agent import ReasoningAgent
import os

def main():
    # quick env-friendly defaults
    os.environ.setdefault("LOCAL_LLM_PATH", os.getenv("LOCAL_LLM_PATH", "gpt2"))
    os.environ.setdefault("LOCAL_LLM_MAX_NEW_TOKENS", os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "60"))
    os.environ.setdefault("LOCAL_LLM_TEMPERATURE", os.getenv("LOCAL_LLM_TEMPERATURE", "0.0"))
    os.environ.setdefault("LOCAL_LLM_TOP_P", os.getenv("LOCAL_LLM_TOP_P", "0.9"))

    persona = PersonalityAgent()
    retrieval = RetrievalAgent()
    agent = ReasoningAgent()

    if agent.model is None:
        print("Model not loaded. Check logs and env/transformers installation.")
        return

    # Example user question
    user_q = "Who are you?"

    # Get retrieved context (if you have memories)
    retrieved = retrieval.retrieve(user_q, k=3)

    # Build prompt + system_instructions using personality agent
    user_prompt, system_instructions = persona.apply_style(user_q, retrieved)

    # Generate
    resp = agent.generate_response(user_prompt=user_prompt, system_instructions=system_instructions)
    print("MODEL RESPONSE:\n", resp)

if __name__ == "__main__":
    main()
