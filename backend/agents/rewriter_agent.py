# Simple rewriter: ensure first-person presence; basic LLM rewrite could be added.
def ensure_first_person(text):
    if text.strip().startswith('I') or ' I ' in text:
        return text.strip()
    # naive insertion
    return 'I ' + text.strip()
