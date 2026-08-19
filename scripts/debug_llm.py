import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

import os
from ktc.triplet import Triplet
from ktc.live_config import LiveRetrievalConfig
from ktc.verbalization import _make_llm_client, _LLM_FEW_SHOT_EXAMPLES

print("LLM_API_KEY set:", bool(os.environ.get("LLM_API_KEY", "").strip()))
config = LiveRetrievalConfig.load()
print("configured llm_model:", config.llm_model)
print("configured llm_api_base:", config.llm_api_base)

client, resolved_config = _make_llm_client()
model = resolved_config.llm_model

t = Triplet(head="victim", relation="can file", tail="an online complaint")

system_prompt = (
    "Convert each knowledge triplet into exactly one fluent English sentence. "
    "Output ONLY the final sentence. No notes, no explanations, no alternative phrasing, "
    "no preamble, and no markdown. "
    "Match the style of these examples:\n\n"
    f"{_LLM_FEW_SHOT_EXAMPLES}"
)
user_prompt = f"Head: {t.head}\nRelation: {t.relation}\nTail: {t.tail}\nSentence:"

print()
print("Calling model:", model)
try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=80,
    )
    print()
    print("RAW response object:")
    print(response)
    print()
    print("finish_reason:", response.choices[0].finish_reason)
    print("message.content repr:", repr(response.choices[0].message.content))
except Exception as exc:
    print()
    print("EXCEPTION RAISED:", type(exc).__name__, "-", exc)