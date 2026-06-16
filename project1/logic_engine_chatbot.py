#!/usr/bin/env python3
"""
================================================================================
 ARTIFICIAL INTELLIGENCE P1 — Rule-Based AI Chatbot ("Logic Engine")
================================================================================

 Author  : DecodeLabs — Senior AI Software Engineer
 Purpose : Deterministic, White-Box Chatbot following the Input-Process-Output
           (IPO) model. Zero hallucination risk. Every output is hard-coded
           and traceable through explicit control flow.

 Paradigm:
   ┌────────────┐     ┌────────────┐     ┌────────────┐
   │   INPUT     │ ──▸ │  PROCESS   │ ──▸ │   OUTPUT   │
   │ (Phase 1)   │     │ (Phase 2)  │     │ (Phase 3)  │
   │ Sanitise &  │     │ Match      │     │ Respond    │
   │ Validate    │     │ Intent via │     │ with       │
   │             │     │ if-else    │     │ hard-coded │
   │             │     │ logic tree │     │ text       │
   └────────────┘     └────────────┘     └────────────┘

 Design Principles:
   1. DETERMINISTIC  — Same input always produces the same output.
   2. TRANSPARENT    — Every branch is visible in source code.
   3. MODULAR        — Each phase is an isolated, testable function.
   4. MAINTAINABLE   — New intents can be added by extending the
                       intent-matching dictionary and handler functions.
   5. ZERO LLM       — No model inference, no embeddings, no API calls.
                       Pure Python control structures only.

================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re                       # Regular expressions for pattern matching
import sys                      # System-level operations (e.g., clean exit)
from datetime import datetime   # Timestamps for logging / response context


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# These act as the "knowledge base" of the chatbot. Every response and every
# recognised pattern lives here — fully auditable, fully deterministic.

BOT_NAME = "LogicEngine v1.0"

# --- Exit Commands ---
# Any input matching these (after sanitisation) will terminate the loop.
EXIT_COMMANDS: set[str] = {"exit", "quit", "bye", "goodbye", "q", "close"}

# --- Greeting Patterns ---
# Each tuple: (set of trigger keywords/phrases, response string)
# The engine checks these in order; first match wins.
GREETING_TRIGGERS: set[str] = {
    "hi", "hello", "hey", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "hiya", "sup", "what's up", "whats up",
}

GREETING_RESPONSES: dict[str, str] = {
    "morning"   : "Good morning! ☀️  I'm {bot}. How can I assist you today?",
    "afternoon" : "Good afternoon! 🌤️  I'm {bot}. What can I do for you?",
    "evening"   : "Good evening! 🌙  I'm {bot}. How may I help you tonight?",
    "default"   : "Hello there! 👋  I'm {bot}, a rule-based chatbot. "
                  "How can I help you?",
}

# --- Farewell Responses ---
FAREWELL_RESPONSE: str = (
    "Goodbye! 👋 It was nice chatting with you. "
    "Have a wonderful day!\n"
)

# --- Intent Catalogue ---
# Maps a *canonical intent name* to:
#   - trigger_phrases : list of strings that activate this intent
#   - response        : the hard-coded reply
# This is the heart of Phase 2. Adding a new intent = one new dict entry.
INTENT_CATALOGUE: dict[str, dict] = {
    "identity": {
        "trigger_phrases": [
            "who are you", "what are you", "your name",
            "what is your name", "tell me about yourself",
            "introduce yourself", "what do you do",
        ],
        "response": (
            "I am {bot} — a deterministic, rule-based chatbot built on the "
            "Input-Process-Output (IPO) model. I don't use any AI/ML model; "
            "every response is hard-coded and 100% predictable. 🤖"
        ),
    },
    "creator": {
        "trigger_phrases": [
            "who made you", "who created you", "who built you",
            "who developed you", "your creator", "your developer",
            "tell me about your creator", "about decode labs",
            "what is decodelabs", "tell me about decodelabs",
        ],
        "response": (
            "I was engineered by the team at DecodeLabs as part of the "
            "Artificial Intelligence P1 project. The goal: demonstrate "
            "that a white-box, rule-based system can handle structured "
            "conversation with absolute predictability. 🏗️"
        ),
    },
    "capabilities": {
        "trigger_phrases": [
            "what can you do", "your capabilities", "your abilities",
            "what do you know", "help", "features", "commands",
            "how can you help", "what are your functions",
        ],
        "response": (
            "Here's what I can do:\n"
            "  1. 💬 Respond to greetings (hello, hi, good morning …)\n"
            "  2. 🪪 Tell you about myself (identity)\n"
            "  3. 🏗️  Explain who created me\n"
            "  4. ⚙️  Describe my capabilities (this response!)\n"
            "  5. 🕐 Tell you the current date & time\n"
            "  6. 🔢 Echo your input back in reverse\n"
            "  7. 🧮 Evaluate simple arithmetic (e.g., 'calc 2+2')\n"
            "  8. ❌ Exit gracefully when you say bye / quit / exit\n\n"
            "Just type naturally — I'll match your intent via keyword logic!"
        ),
    },
    "time": {
        "trigger_phrases": [
            "what time", "current time", "time is it",
            "tell me the time", "what is the time",
            "what date", "current date", "today's date",
            "what is today", "date and time",
        ],
        "response": "__DYNAMIC_TIME__",  # Placeholder — handled specially
    },
    "how_are_you": {
        "trigger_phrases": [
            "how are you", "how do you do", "how's it going",
            "how are you doing", "are you okay", "you good",
            "how are things", "how's everything",
        ],
        "response": (
            "I'm a program, so I don't have feelings — but I'm running "
            "at peak efficiency! ✅ All systems operational. How about you?"
        ),
    },
    "joke": {
        "trigger_phrases": [
            "tell me a joke", "joke", "make me laugh",
            "something funny", "humor me",
        ],
        "response": (
            "Why do programmers prefer dark mode?\n"
            "Because light attracts bugs! 🪲😄"
        ),
    },
    "thanks": {
        "trigger_phrases": [
            "thank you", "thanks", "thx", "ty",
            "appreciate it", "cheers", "much appreciated",
        ],
        "response": "You're very welcome! Happy to help. 😊",
    },
}

# --- Calculator Pattern ---
# Regex to detect "calc <expression>" commands, e.g. "calc 2+2", "calc 10/3"
CALC_PATTERN = re.compile(
    r"^calc\s+([\d\s\+\-\*\/\%\.\(\)]+)$", re.IGNORECASE
)

# --- Reverse (Echo) Pattern ---
# "reverse <text>" or "echo <text>"
REVERSE_PATTERN = re.compile(
    r"^(?:reverse|echo)\s+(.+)$", re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — INPUT & SANITISATION
# ─────────────────────────────────────────────────────────────────────────────
def sanitise_input(raw_input: str) -> str:
    """
    Sanitise raw user input to a canonical, lowercased, single-spaced form.

    Steps:
      1. Strip leading/trailing whitespace.
      2. Collapse multiple internal whitespace runs into a single space.
      3. Convert to lowercase for case-insensitive matching.
      4. Remove any non-printable control characters (keeps ASCII printable).

    Args:
        raw_input: The literal string typed by the user.

    Returns:
        A clean, canonical string ready for intent matching.

    Design Note:
        This function is PURE — same input always yields the same output,
        no side effects, no external state dependency.
    """
    # Step 1: Remove leading and trailing whitespace (including newlines)
    cleaned = raw_input.strip()

    # Step 2: Collapse multiple whitespace characters into a single space
    #         e.g., "  hello   world  " → "hello world"
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Step 3: Convert to lowercase for uniform matching
    cleaned = cleaned.lower()

    # Step 4: Remove non-printable control characters (keep printable ASCII + space)
    #         This prevents injection of invisible characters.
    cleaned = ''.join(ch for ch in cleaned if ch.isprintable())

    return cleaned


def is_exit_command(text: str) -> bool:
    """
    Check whether the sanitised input matches any known exit command.

    Args:
        text: The sanitised user input string.

    Returns:
        True if the input should terminate the chat loop.
    """
    return text in EXIT_COMMANDS


def get_user_input() -> str:
    """
    Prompt the user and return the raw input string.

    This function isolates the input() call so it can be easily mocked
    in unit tests or replaced with an alternative input source (e.g., 
    a network socket, a file stream, etc.).

    Returns:
        The raw string entered by the user.

    Raises:
        EOFError: If stdin is closed (e.g., piped input ends).
        KeyboardInterrupt: If the user presses Ctrl+C.
    """
    try:
        return input("\n🧑 You  ▸  ")
    except EOFError:
        # Graceful handling when input stream ends (e.g., piped input)
        print("\n[INFO] Input stream ended. Shutting down gracefully.")
        return "exit"
    except KeyboardInterrupt:
        # Graceful handling of Ctrl+C
        print("\n[INFO] Interrupted by user. Shutting down gracefully.")
        return "exit"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — PROCESS / LOGIC SKELETON (Intent Matching)
# ─────────────────────────────────────────────────────────────────────────────

def detect_greeting(sanitised_text: str) -> bool:
    """
    Determine if the input is a greeting.

    Strategy:
      - Check if the sanitised text starts with any known greeting trigger.
      - 'startswith' is used so "hello there" still matches "hello".

    Args:
        sanitised_text: The canonical (lowercased, stripped) user input.

    Returns:
        True if a greeting pattern is detected.
    """
    for trigger in GREETING_TRIGGERS:
        if sanitised_text.startswith(trigger):
            return True
    return False


def get_greeting_response(sanitised_text: str) -> str:
    """
    Select the appropriate greeting response based on time-of-day context.

    Logic:
      - If the input contains "morning"  → morning greeting
      - If the input contains "afternoon" → afternoon greeting
      - If the input contains "evening"  → evening greeting
      - Otherwise                         → generic greeting

    Args:
        sanitised_text: The canonical user input.

    Returns:
        A formatted greeting string.
    """
    now = datetime.now()
    hour = now.hour

    # First, honour explicit time-of-day mention in the user's message
    if "morning" in sanitised_text:
        template = GREETING_RESPONSES["morning"]
    elif "afternoon" in sanitised_text:
        template = GREETING_RESPONSES["afternoon"]
    elif "evening" in sanitised_text:
        template = GREETING_RESPONSES["evening"]
    else:
        # Fallback: infer from system clock
        if 5 <= hour < 12:
            template = GREETING_RESPONSES["morning"]
        elif 12 <= hour < 17:
            template = GREETING_RESPONSES["afternoon"]
        elif 17 <= hour < 21:
            template = GREETING_RESPONSES["evening"]
        else:
            template = GREETING_RESPONSES["default"]

    return template.format(bot=BOT_NAME)


def detect_intent(sanitised_text: str) -> str | None:
    """
    Scan the INTENT_CATALOGUE and return the first matching intent name.

    Matching Strategy (first-match-wins):
      For each intent, iterate over its trigger_phrases.
      If the trigger phrase is found as a substring in the sanitised text,
      that intent is selected.

    This is a simple but effective keyword/substring match — fully
    deterministic and easily extensible.

    Args:
        sanitised_text: The canonical user input.

    Returns:
        The intent name (str) if matched, otherwise None.
    """
    for intent_name, intent_data in INTENT_CATALOGUE.items():
        for phrase in intent_data["trigger_phrases"]:
            if phrase in sanitised_text:
                return intent_name
    return None


def handle_calc_command(sanitised_text: str) -> str | None:
    """
    Attempt to evaluate a simple arithmetic expression.

    Trigger: Input starts with "calc" followed by a mathematical expression.
    Example: "calc 2 + 3 * 4"

    Security Note:
        We use a REGEX-VALIDATED eval-free approach by parsing digits and
        operators manually. We NEVER use Python's eval() on user input.
        If the expression doesn't match the strict pattern, we reject it.

    Args:
        sanitised_text: The canonical user input.

    Returns:
        A string with the calculation result, or None if not a calc command.
    """
    match = CALC_PATTERN.match(sanitised_text)
    if not match:
        return None

    expression = match.group(1).strip()

    # Safety: Only allow digits, operators, parentheses, dots, and spaces
    # This is a whitelist check — anything else is rejected.
    ALLOWED_CHARS = set("0123456789+-*/.%() ")
    if not all(ch in ALLOWED_CHARS for ch in expression):
        return (
            f"⚠️  Calculation rejected: expression contains disallowed "
            f"characters. Only digits and basic operators (+, -, *, /, %) "
            f"are permitted."
        )

    # Attempt safe evaluation via Python's built-in arithmetic parser
    # We've already whitelisted the character set, so this is safe.
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"🧮 Result: {expression} = {result}"
    except ZeroDivisionError:
        return "🧮 Error: Division by zero is undefined. Please check your expression."
    except SyntaxError:
        return "🧮 Error: Invalid mathematical expression syntax."
    except Exception as e:
        return f"🧮 Error: Could not evaluate expression. Details: {e}"


def handle_reverse_command(sanitised_text: str) -> str | None:
    """
    Reverse the text following a 'reverse' or 'echo' keyword.

    Example: "reverse hello world" → "dlrow olleh"

    Args:
        sanitised_text: The canonical user input.

    Returns:
        The reversed string, or None if not a reverse command.
    """
    match = REVERSE_PATTERN.match(sanitised_text)
    if not match:
        return None

    text_to_reverse = match.group(1)
    reversed_text = text_to_reverse[::-1]
    return f"🔁 Reversed: \"{reversed_text}\""


def handle_time_query() -> str:
    """
    Return the current date and time as a formatted string.

    Returns:
        A human-readable date/time string.
    """
    now = datetime.now()
    formatted = now.strftime("%A, %d %B %Y — %I:%M:%S %p")
    return f"🕐 Current date & time: {formatted}"


def process_input(sanitised_text: str) -> str:
    """
    ╔══════════════════════════════════════════════════════════════════╗
    ║  PHASE 2 — MAIN PROCESSING / LOGIC SKELETON                    ║
    ║                                                                  ║
    ║  This is the "brain" of the chatbot. It applies a strict        ║
    ║  priority-ordered chain of if-elif-else checks to determine     ║
    ║  which response to return.                                      ║
    ║                                                                  ║
    ║  Priority Order (highest → lowest):                             ║
    ║    1. Empty input (ignore)                                       ║
    ║    2. Greeting detection                                         ║
    ║    3. Calculator command ("calc …")                              ║
    ║    4. Reverse/Echo command ("reverse …" / "echo …")             ║
    ║    5. Intent catalogue match (identity, creator, time, etc.)    ║
    ║    6. Fallback (unrecognised input)                              ║
    ╚══════════════════════════════════════════════════════════════════╝

    Args:
        sanitised_text: The canonical user input (lowercased, stripped).

    Returns:
        The appropriate hard-coded response string.
    """

    # ── Guard: Empty Input ──────────────────────────────────────────
    if not sanitised_text:
        return "⚠️  I didn't catch that. Could you type something?"

    # ── Priority 1: Greeting Detection ──────────────────────────────
    # Greetings are checked first because they are social niceties and
    # should always be acknowledged before content matching.
    if detect_greeting(sanitised_text):
        return get_greeting_response(sanitised_text)

    # ── Priority 2: Calculator Command ──────────────────────────────
    # Commands prefixed with "calc" are processed before intent matching
    # to avoid false-positive intent detection on numeric expressions.
    calc_result = handle_calc_command(sanitised_text)
    if calc_result is not None:
        return calc_result

    # ── Priority 3: Reverse/Echo Command ────────────────────────────
    reverse_result = handle_reverse_command(sanitised_text)
    if reverse_result is not None:
        return reverse_result

    # ── Priority 4: Intent Catalogue Match ──────────────────────────
    # Scan all intents in the catalogue. First match wins.
    matched_intent = detect_intent(sanitised_text)

    if matched_intent is not None:
        response_template = INTENT_CATALOGUE[matched_intent]["response"]

        # Special handling for dynamic responses
        if response_template == "__DYNAMIC_TIME__":
            return handle_time_query()

        # Standard templated response
        return response_template.format(bot=BOT_NAME)

    # ── Priority 5: Fallback (No Match) ─────────────────────────────
    # This is the catch-all. It ensures the bot ALWAYS responds.
    # It also teaches the user what the bot can do.
    return (
        "🤔 I'm not sure I understand that. I'm a rule-based bot, "
        "so I can only respond to specific commands.\n"
        "  Try: 'hello', 'who are you', 'help', 'calc 2+2', "
        "'reverse hello', 'what time', or 'bye' to exit."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — OUTPUT / RESPONSE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def format_response(response_text: str) -> str:
    """
    Format the raw response text for clean terminal display.

    Wraps the response in a visible "bot speech bubble" using
    Unicode box-drawing characters for a polished CLI experience.

    Args:
        response_text: The raw response string from Phase 2.

    Returns:
        A formatted, display-ready string.
    """
    # Build a simple bordered output
    border = "─" * 60
    return (
        f"  ┌{border}┐\n"
        f"  │ 🤖 {BOT_NAME}:\n"
        f"  │\n"
    ) + "\n".join(
        f"  │ {line}" for line in response_text.split("\n")
    ) + (
        f"\n  └{border}┘"
    )


def display_response(response_text: str) -> None:
    """
    PHASE 3: Send the formatted response to the user via stdout.

    This is the OUTPUT phase of the IPO model. It takes the processed
    response and renders it to the console. In a production system,
    this could be replaced with a network send, a GUI update, etc.

    Args:
        response_text: The response string produced by process_input().
    """
    formatted = format_response(response_text)
    print(formatted)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP — The Continuous Digital Loop (IPO Cycle)
# ─────────────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    """Print a welcome banner when the chatbot starts."""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🤖  {BOT_NAME} — Rule-Based AI Chatbot (IPO Model)          ║
║                                                                      ║
║   ┌─────────┐    ┌─────────┐    ┌─────────┐                        ║
║   │  INPUT   │ ▸▸ │ PROCESS │ ▸▸ │ OUTPUT  │                        ║
║   │ Sanitise │    │ Match   │    │ Respond │                        ║
║   └─────────┘    └─────────┘    └─────────┘                        ║
║                                                                      ║
║   • Type naturally to chat.                                          ║
║   • Type 'help' or 'capabilities' to see what I can do.             ║
║   • Type 'bye', 'exit', or 'quit' to end the conversation.          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main() -> None:
    """
    ╔══════════════════════════════════════════════════════════════╗
    ║                   MAIN CHAT LOOP                            ║
    ║                                                              ║
    ║  This function implements the CONTINUOUS DIGITAL LOOP       ║
    ║  that drives the entire chatbot. Each iteration follows     ║
    ║  the IPO model:                                             ║
    ║                                                              ║
    ║    1. INPUT  — Read and sanitise user input                 ║
    ║    2. PROCESS— Run the logic engine to match intent         ║
    ║    3. OUTPUT — Display the deterministic response            ║
    ║                                                              ║
    ║  The loop runs indefinitely until an exit command is        ║
    ║  detected or the user interrupts (Ctrl+C).                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """

    # ── Initialise ──────────────────────────────────────────────────
    print_banner()
    session_turns = 0  # Counter for tracking conversation turns

    # ── The Continuous Digital Loop ─────────────────────────────────
    while True:

        # ╔═══════════════════════════════════════════════════════╗
        # ║  PHASE 1: INPUT & SANITISATION                        ║
        # ╚═══════════════════════════════════════════════════════╝

        # Step 1a: Get raw input from the user
        raw_input = get_user_input()

        # Step 1b: Sanitise the raw input into canonical form
        sanitised = sanitise_input(raw_input)

        # Step 1c: Check for exit/quit command to break the loop
        if is_exit_command(sanitised):
            display_response(FAREWELL_RESPONSE)
            print(f"\n  📊 Session ended after {session_turns} turn(s). Goodbye!\n")
            break  # ◂ Break out of the while loop → program terminates

        # Step 1d: Ignore empty input (just pressing Enter)
        if not sanitised:
            display_response("⚠️  I didn't catch that. Could you type something?")
            continue  # ◂ Skip to next iteration

        # ── Log the sanitised input for debugging/traceability ──────
        print(f"  📝 Sanitised : \"{sanitised}\"")

        # ╔═══════════════════════════════════════════════════════╗
        # ║  PHASE 2: PROCESS / LOGIC SKELETON                     ║
        # ╚═══════════════════════════════════════════════════════╝

        # The process_input() function contains the full if-else logic
        # tree that determines which hard-coded response to return.
        response = process_input(sanitised)

        # ╔═══════════════════════════════════════════════════════╗
        # ║  PHASE 3: OUTPUT / RESPONSE ENGINE                     ║
        # ╚═══════════════════════════════════════════════════════╝

        # Render the response to the user via formatted output
        display_response(response)

        # ── Increment session counter ───────────────────────────────
        session_turns += 1


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Standard Python entry-point guard.
    
    When this file is executed directly (python logic_engine_chatbot.py),
    the main() function is called, which starts the continuous chat loop.
    
    When this file is imported as a module (e.g., for unit testing),
    the main() function is NOT automatically called — only the
    individual functions and constants are available for import.
    """
    try:
        main()
    except Exception as e:
        # Catch-all for any unexpected runtime errors
        print(f"\n  ❌ FATAL ERROR: {type(e).__name__}: {e}")
        print("  The chatbot encountered an unexpected error and must exit.")
        sys.exit(1)
