"""Stage-by-stage smokes for the Rakshak demo."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from demo.dialogues import PANEL_DIALOGUES
from demo.orchestrator import RakshakOrchestrator
from demo.refine import draft_should_discard, refine
from demo.retrieve import load_demo_env, retrieve

logging.basicConfig(level=logging.INFO, format="--- %(levelname)s: %(message)s ---")
load_demo_env()


def cmd_retrieve(text: str) -> None:
    result = retrieve(text, dialogue_id="cli-retrieve")
    print("history:", result.dialog_history)
    print("corpus:", result.corpus_source)
    print("live_enabled:", result.live_enabled)
    print("live_sentences:", len(result.live_verbalized))
    print("passages:", len(result.passages))
    for i, passage in enumerate(result.passages[:5], 1):
        print(f"  [{i}] {passage[:240]}")
    print("sources:", result.sources)
    print("final_knowledge:")
    print(result.final_knowledge_text or "(empty)")
    if result.error:
        print("error:", result.error)
        sys.exit(1)


def cmd_rdpt(text: str) -> None:
    result = retrieve(text, dialogue_id="cli-rdpt")
    from demo.rdpt_generate import generate_draft

    draft = generate_draft(result.dialog_history, result.final_knowledge_text)
    print("draft:", draft or "(empty)")
    print("discard:", draft_should_discard(draft))


def cmd_refine(text: str) -> None:
    result = retrieve(text, dialogue_id="cli-refine")
    reply, status = refine(result.dialog_history, result.final_knowledge_text, "", user_message=text)
    print("status:", status)
    print(reply)


def cmd_chat(text: str, use_rdpt: bool) -> None:
    bot = RakshakOrchestrator(use_rdpt=use_rdpt, dialogue_id="cli-chat")
    record = bot.respond(text, [])
    print(json.dumps({k: record[k] for k in record if k != "verbalized"}, ensure_ascii=False, indent=2))
    print("--- reply ---")
    print(record["final_reply"])


def cmd_dialogues(use_rdpt: bool) -> None:
    failures = 0
    for spec in PANEL_DIALOGUES:
        bot = RakshakOrchestrator(use_rdpt=use_rdpt, dialogue_id=f"panel-{spec['id']}")
        history = []
        print("=" * 72)
        print(spec["title"], spec["id"])
        for turn in spec["turns"]:
            record = bot.respond(turn, history)
            history.append([turn, record["final_reply"]])
            print("USER:", turn)
            print("BOT:", record["final_reply"])
            print("draft_status:", record["draft_status"], "knowledge_chars:", len(record["retrieved_knowledge"]))
            if not (record["final_reply"] or "").strip():
                failures += 1
    if failures:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rakshak demo stage CLI")
    parser.add_argument(
        "stage",
        choices=["retrieve", "rdpt", "refine", "chat", "dialogues"],
    )
    parser.add_argument("--text", default="My husband beats me and I am scared.")
    parser.add_argument("--no-rdpt", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Disable live web search")
    args = parser.parse_args()
    if args.offline:
        os.environ["RAKSHAK_OFFLINE"] = "1"
    use_rdpt = not args.no_rdpt
    if args.stage == "retrieve":
        cmd_retrieve(args.text)
    elif args.stage == "rdpt":
        cmd_rdpt(args.text)
    elif args.stage == "refine":
        cmd_refine(args.text)
    elif args.stage == "chat":
        cmd_chat(args.text, use_rdpt)
    else:
        cmd_dialogues(use_rdpt)


if __name__ == "__main__":
    main()
