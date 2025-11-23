# backend/agents/personality_agent.py
from pathlib import Path

class PersonalityAgent:
    def __init__(self):
        # Load the profile text (if available) so prompts include real specifics
        base_dir = Path(__file__).resolve().parent.parent
        profile_path = base_dir / "data" / "soundar_profile.txt"
        profile_text = ""
        if profile_path.exists():
            try:
                profile_text = profile_path.read_text(encoding="utf-8").strip()
            except Exception:
                profile_text = ""

        # Core personality instruction - short, explicit constraints
        # NOTE: Keep this strict — instructs the LLM to behave like "Soundar"
        self.system_instructions = (
            "You are Soundar. Reply exactly as Soundar would: use first-person, "
            "write concisely, naturally, and warmly in a conversational Indian tone. "
            "Important rules you MUST follow:\n"
            "1) DO NOT ask the user questions or prompt the user for more info.\n"
            "2) DO NOT add filler openings like 'That's a lovely question!' or generic praise.\n"
            "3) Answer in 1-5 short sentences unless the user requests details. Be concrete.\n"
            "4) Use short personal-style examples or anecdotes only when directly relevant.\n"
            "5) NEVER turn the answer into a list of generic tips unless asked for a list.\n"
            "6) Avoid repeating the user's question in the answer.\n"
            "7) Keep tone friendly, humble, and confident — like Soundar.\n\n"
        )

        # Append the profile content (if found) so the LLM has your real specifics
        if profile_text:
            self.system_instructions += (
                "Here is Soundar's profile and background information to draw from:\n\n"
                f"{profile_text}\n\n"
                "When possible, use these specifics in your responses (shortly and naturally)."
            )
        else:
            self.system_instructions += (
                "No profile content was loaded. If you want truly personalized answers, "
                "ingest 'backend/data/soundar_profile.txt' into the RAG database."
            )

    def apply_style(self, user_query: str, retrieved_context: str):
        """
        Return:
          - user_prompt: the user-facing text the LLM will answer
          - system_instructions: the persona + rules (to be used as system message)
        """
        # Combine retrieved context into a concise memory block (if available)
        if retrieved_context and retrieved_context.strip():
            memory_block = f"Relevant remembered context:\n{retrieved_context}\n"
        else:
            memory_block = ""

        # Build the user prompt — simple and direct: the exact thing to answer
        user_prompt = f"{memory_block}\nUser wants a direct reply to this:\n\"{user_query.strip()}\"\n\nRespond directly."
        return user_prompt, self.system_instructions
