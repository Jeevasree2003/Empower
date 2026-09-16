"""Gradio chat UI. Shows only the Ollama (or fallback) reply."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from demo.retrieve import load_demo_env

logging.basicConfig(level=logging.INFO, format="--- %(levelname)s: %(message)s ---")


def _strip_demo_flags(argv: list[str]) -> list[str]:
    """Gradio re-parses sys.argv and crashes on --no-rdpt / --offline."""
    skip = {"--no-rdpt", "--share", "--offline"}
    kept = [argv[0]]
    for item in argv[1:]:
        if item in skip:
            continue
        kept.append(item)
    return kept


def build_interface(use_rdpt: bool = True):
    import gradio as gr

    from demo.orchestrator import RakshakOrchestrator

    session = {"bot": RakshakOrchestrator(use_rdpt=use_rdpt)}

    def reply(message, history):
        result = session["bot"].respond(message, history)
        return result["final_reply"]

    description = (
        "Women and child safety counselling assistant. "
        "Uses live web search when LIVE_SEARCH_API_KEY is set. "
        "Emergency: 112. Women in distress: 181. Childline: 1098. KIRAN: 1800-599-0019. "
        "This is not a substitute for police, medical care, or a lawyer."
    )

    with gr.Blocks(title="Rakshak") as demo:
        gr.Markdown(f"# Rakshak\n{description}")
        try:
            chatbot = gr.Chatbot(label="Conversation", type="messages")
            messages_format = True
        except TypeError:
            chatbot = gr.Chatbot(label="Conversation")
            messages_format = False
        msg = gr.Textbox(placeholder="Describe the situation…", label="Message", lines=3)
        send = gr.Button("Send")

        def user_submit(message, history):
            history = history or []
            text = (message or "").strip()
            if not text:
                return "", history
            answer = reply(text, history)
            if messages_format:
                history = history + [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": answer},
                ]
            else:
                history = history + [[text, answer]]
            return "", history

        msg.submit(user_submit, [msg, chatbot], [msg, chatbot])
        send.click(user_submit, [msg, chatbot], [msg, chatbot])
    return demo


def main() -> None:
    load_demo_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rdpt", action="store_true", help="Skip GPT-2 draft; KTC + Ollama only")
    parser.add_argument("--offline", action="store_true", help="Disable live web search")
    parser.add_argument("--share", action="store_true")
    args, _unknown = parser.parse_known_args()
    if args.offline:
        os.environ["RAKSHAK_OFFLINE"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    sys.argv = _strip_demo_flags(sys.argv)
    app = build_interface(use_rdpt=not args.no_rdpt)
    app.launch(share=args.share, inbrowser=False)


if __name__ == "__main__":
    main()
