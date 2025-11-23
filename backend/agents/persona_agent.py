# Personality agent prepares the prompt using retrieved profile chunks.
def build_persona_prompt(user_text, retrieved_chunks, conversation_history=None):
    persona_header = (
        "You are Soundar. ALWAYS speak in first person using 'I'.\n"
        "Tone: concise, thoughtful, slightly playful, technically literate.\n"
        "Use the profile context below to answer factually. If unsure, say 'I don't remember.'\n\n"
    )
    profile_ctx = "\n---PROFILE---\n" + "\n".join(retrieved_chunks) + "\n---END PROFILE---\n\n"
    convo = ""
    if conversation_history:
        convo = "\nConversation history:\n"
        for m in conversation_history[-6:]:
            convo += f"{m['role']}: {m['content']}\n"
    prompt = persona_header + profile_ctx + convo + f"User: {user_text}\nSoundar:"
    return prompt
