"""Safe copy when retrieval or Ollama fails. Never show RDPT loops to the user."""

CRISIS_FALLBACK = (
    "If you are in immediate danger, call emergency services 112 in India now "
    "and move to a safer place if you can. I can stay with you and help you "
    "think through the next safe step. Are you somewhere you can talk?"
)

GENERIC_FALLBACK = (
    "I am here to help. I could not reach the local assistant just now, but you "
    "can still call 112 in an emergency, 181 for women in distress, or KIRAN "
    "1800-599-0019 for 24x7 mental health support. Please tell me what happened "
    "and whether you feel safe right now."
)

EMPTY_KNOWLEDGE_FALLBACK = (
    "I hear you. I do not have a matching fact sheet for this turn, so I will "
    "not invent a law or a helpline number. If you are unsafe, call 112. "
    "Can you share one more detail about what you need (police, medical help, "
    "or someone to talk to?"
)


def crisis_or_generic(user_text: str) -> str:
    lowered = (user_text or "").lower()
    crisis_tokens = (
        "kill myself",
        "suicide",
        "suicidal",
        "going to kill",
        "rape",
        "he is here",
        "he is coming",
        "beaten",
        "below 18",
        "childline",
    )
    if any(token in lowered for token in crisis_tokens):
        return CRISIS_FALLBACK
    return GENERIC_FALLBACK
