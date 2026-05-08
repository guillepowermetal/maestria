#!/usr/bin/env python3
"""Small HuggingChat / hugchat experiment runner.

Usage:
  1. Install: python3 -m pip install -r requirements.txt
  2. Export credentials:
       export HF_EMAIL="you@example.com"
       export HF_PASSWORD="your-password"
     Preferably, reuse a cookie JSON file:
       export HUGCHAT_COOKIE_FILE="./cookies/you@example.com.json"
  3. Run:
       python3 hugchat_prompt_lab.py --mode all

This uses the unofficial `hugchat` package. Keep request volume low and never
commit credentials or generated cookie files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from hugchat import hugchat
from hugchat.login import Login


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs without adding another dependency."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()

COOKIE_DIR = Path(os.getenv("HUGCHAT_COOKIE_DIR", "./cookies/"))
COOKIE_PLACEHOLDERS = {
    "./cookies/your-email.json",
    "./cookies/tu-email.json",
    "./cookies/huggingface.json",
}


PROMPT_EXPERIMENTS = {
    "baseline": "Explica en 5 lineas que es la ingenieria de prompting.",
    "role": (
        "Actua como docente universitario de IA aplicada. Explica la ingenieria "
        "de prompting para estudiantes que empiezan, con un ejemplo practico."
    ),
    "structured": (
        "Necesito comparar tecnicas de prompting. Responde en una tabla Markdown "
        "con columnas: tecnica, cuando usarla, ejemplo breve. Incluye: rol, "
        "contexto, restricciones, ejemplos y refinamiento iterativo."
    ),
    "socratic": (
        "Quiero aprender a conversar mejor con un chatbot. Hazme 5 preguntas "
        "socraticas, una por una, para descubrir que informacion falta antes de "
        "pedirle una tarea compleja."
    ),
    "web_search": (
        "Busca informacion actual sobre HuggingChat y resume en 6 puntos: "
        "que es, que permite hacer, limitaciones y precauciones."
    ),
}


def cookie_dir_for_login() -> str:
    """Return a cookie directory string with the trailing slash hugchat expects."""
    cookie_dir = str(COOKIE_DIR)
    return cookie_dir if cookie_dir.endswith("/") else f"{cookie_dir}/"


def build_chatbot() -> hugchat.ChatBot:
    """Create a HuggingChat bot from environment credentials or a cookie file."""
    COOKIE_DIR.mkdir(parents=True, exist_ok=True)

    email = os.getenv("HF_EMAIL")
    password = os.getenv("HF_PASSWORD")
    cookie_file = os.getenv("HUGCHAT_COOKIE_FILE", "").strip()
    access_token = os.getenv("HF_ACCESS_TOKEN", "").strip()

    if access_token:
        raise RuntimeError(
            "HF_ACCESS_TOKEN was found, but hugchat does not authenticate with "
            "Hugging Face access tokens. It needs browser session cookies for "
            "HuggingChat, usually including keys like 'token' and 'hf-chat'.\n\n"
            "Fix .env by removing HF_ACCESS_TOKEN for this script and setting:\n"
            "HUGCHAT_COOKIE_FILE=./cookies/<your-cookie-file>.json"
        )

    if cookie_file in COOKIE_PLACEHOLDERS:
        raise RuntimeError(
            f"HUGCHAT_COOKIE_FILE is still set to the example path: {cookie_file}\n\n"
            "Export your Hugging Face browser cookies as JSON, save that file "
            "under ./cookies/, and update HUGCHAT_COOKIE_FILE to the real file "
            "name. Example:\n"
            "HUGCHAT_COOKIE_FILE=./cookies/hf-chat-cookies.json"
        )

    if cookie_file:
        if not Path(cookie_file).exists():
            raise RuntimeError(
                f"HUGCHAT_COOKIE_FILE points to a file that does not exist: "
                f"{cookie_file}\n\n"
                "This must be a JSON file containing browser cookies, not a "
                "Hugging Face access token."
            )
        return hugchat.ChatBot(cookie_path=cookie_file)

    if email and password:
        sign = Login(email, password)
        try:
            cookies = sign.login(cookie_dir_path=cookie_dir_for_login(), save_cookies=True)
        except Exception as exc:
            raise RuntimeError(
                "hugchat could not complete the Hugging Face email/password "
                "login flow. This usually happens when Hugging Face changes its "
                "web OAuth flow, requires an interactive browser step, or returns "
                "a challenge that the unofficial library cannot parse.\n\n"
                "Recommended fix:\n"
                "1. Open https://huggingface.co/chat in your browser and log in.\n"
                "2. Export your Hugging Face cookies as JSON.\n"
                "3. Save the JSON file under ./cookies/, for example "
                "./cookies/huggingface.json.\n"
                "4. Set this in .env:\n"
                "   HUGCHAT_COOKIE_FILE=./cookies/huggingface.json\n"
                "5. Leave HF_EMAIL and HF_PASSWORD empty, then run again."
            ) from exc
        return hugchat.ChatBot(cookies=cookies.get_dict())

    raise RuntimeError(
        "Set HF_EMAIL and HF_PASSWORD, or set HUGCHAT_COOKIE_FILE to a saved "
        "Hugging Face cookie JSON file."
    )


def run_prompt(chatbot: hugchat.ChatBot, name: str, prompt: str, web_search: bool) -> None:
    print(f"\n=== {name.upper()} ===")
    print(f"Prompt:\n{prompt}\n")
    result = chatbot.chat(prompt, web_search=web_search)
    response_text = result.wait_until_done()
    print(f"Respuesta:\n{response_text}\n")

    if web_search:
        print("Fuentes encontradas por HuggingChat:")
        for source in result.get_search_sources():
            print(f"- {source.title}: {source.link}")


def select_experiments(mode: str) -> Iterable[tuple[str, str]]:
    if mode == "all":
        return PROMPT_EXPERIMENTS.items()
    return [(mode, PROMPT_EXPERIMENTS[mode])]


def main() -> None:
    parser = argparse.ArgumentParser(description="HuggingChat prompt experiment lab")
    parser.add_argument(
        "--mode",
        choices=[*PROMPT_EXPERIMENTS.keys(), "all"],
        default="baseline",
        help="Which prompt experiment to run.",
    )
    parser.add_argument(
        "--model-index",
        type=int,
        help="Optional HuggingChat model index to switch to before testing.",
    )
    args = parser.parse_args()

    chatbot = build_chatbot()

    if args.model_index is not None:
        chatbot.switch_llm(args.model_index)

    chatbot.new_conversation(switch_to=True)

    for name, prompt in select_experiments(args.mode):
        run_prompt(
            chatbot=chatbot,
            name=name,
            prompt=prompt,
            web_search=name == "web_search",
        )


if __name__ == "__main__":
    main()
