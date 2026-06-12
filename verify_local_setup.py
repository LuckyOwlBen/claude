#!/usr/bin/env python3
"""
verify_local_setup.py

Verifies the full Claude Code + Ollama + Gemma 4 stack before use.
Runs three checks in sequence:
  1. Ollama health and model availability
  2. Basic Anthropic Messages API call
  3. Tool calling round-trip

Prerequisites:
  pip install httpx

How to run:
  python verify_local_setup.py

Expected output on a working setup:
  [PASS] Ollama is running on localhost:11434
  [PASS] Model 'gemma4-claude' is available
  [PASS] Anthropic Messages API call successful
  [PASS] Tool calling: model produced a valid tool_use block
  All checks passed -- Claude Code + Ollama + Gemma 4 is ready.
"""

import httpx
import json
import sys

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME      = "gemma4-claude"   # Must match your Modelfile variant name
TIMEOUT         = 120.0             # Seconds -- generation can be slow on first call


def check_ollama_health() -> bool:
    """
    Check 1: Verify Ollama is running and responding.
    Hits the root endpoint which returns 'Ollama is running' when healthy.
    """
    print("\nCheck 1: Ollama health")
    try:
        response = httpx.get(OLLAMA_BASE_URL, timeout=5.0)
        if "Ollama is running" in response.text:
            print(f"  [PASS] Ollama is running on {OLLAMA_BASE_URL}")
            return True
        else:
            print(f"  [FAIL] Unexpected response: {response.text[:100]}")
            return False
    except httpx.ConnectError:
        print(f"  [FAIL] Cannot connect to {OLLAMA_BASE_URL}")
        print("         Is Ollama running? Try: ollama serve")
        return False


def check_model_available() -> bool:
    """
    Check 2: Verify the specific model variant is available in Ollama.
    Uses the /api/tags endpoint which lists all pulled models.
    """
    print("\nCheck 2: Model availability")
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        data     = response.json()
        models   = [m["name"] for m in data.get("models", [])]

        # Normalize: Ollama may add ":latest" if not specified
        normalized = [m.split(":")[0] for m in models]

        if MODEL_NAME in models or MODEL_NAME in normalized:
            print(f"  [PASS] Model '{MODEL_NAME}' is available")
            return True
        else:
            print(f"  [FAIL] Model '{MODEL_NAME}' not found")
            print(f"         Available models: {', '.join(models) or 'none'}")
            print(f"         Run: ollama create {MODEL_NAME} -f ~/.ollama/Modelfiles/gemma4-claude")
            return False
    except Exception as e:
        print(f"  [FAIL] Error checking model list: {e}")
        return False


def check_messages_api() -> bool:
    """
    Check 3: Send a basic Anthropic Messages API call to the local endpoint.
    Verifies the request format, model routing, and basic generation work.
    Uses the same /v1/messages path and request schema that Claude Code uses.
    Note: Claude Code uses http://localhost:11434 (root), not /v1.
    The Anthropic-compatible API is at /api/chat or the root -- Ollama routes it.
    """
    print("\nCheck 3: Anthropic Messages API call")

    payload = {
        "model": MODEL_NAME,
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: VERIFICATION_OK"
            }
        ]
    }

    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         "ollama",            # Required by the API spec; value ignored locally
        "anthropic-version": "2023-06-01"         # Required version header
    }

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/v1/messages",
            json=payload,
            headers=headers,
            timeout=TIMEOUT
        )

        if response.status_code != 200:
            print(f"  [FAIL] HTTP {response.status_code}: {response.text[:200]}")
            return False

        data = response.json()

        # Anthropic Messages API response structure:
        # { "content": [{"type": "text", "text": "..."}], "stop_reason": "..." }
        content_blocks = data.get("content", [])
        text_blocks    = [b for b in content_blocks if b.get("type") == "text"]

        if not text_blocks:
            print(f"  [FAIL] No text content in response: {json.dumps(data, indent=2)}")
            return False

        response_text = text_blocks[0].get("text", "")
        print(f"  [PASS] Anthropic Messages API call successful")
        print(f"         Model response: {response_text[:80]}")
        return True

    except Exception as e:
        print(f"  [FAIL] Request failed: {e}")
        return False


def check_tool_calling() -> bool:
    """
    Check 4: Verify tool calling works end-to-end.
    This is the most important check for Claude Code agentic use.
    Claude Code relies on the model correctly producing tool_use blocks
    for every file operation, shell command, and code execution.

    Sends a simple tool definition and a prompt that should trigger it.
    Verifies the model returns a tool_use block (not just text describing the call).
    """
    print("\nCheck 4: Tool calling verification")

    # A minimal tool definition using the Anthropic function calling schema
    tools = [
        {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The absolute or relative file path to read"
                    }
                },
                "required": ["path"]
            }
        }
    ]

    payload = {
        "model": MODEL_NAME,
        "max_tokens": 256,
        "tools": tools,
        # Force the model to call a tool rather than respond in text.
        # tool_choice: {"type": "any"} requires any tool use.
        # Remove this if testing whether the model self-selects tools.
        "tool_choice": {"type": "any"},
        "messages": [
            {
                "role": "user",
                "content": "Read the file at /tmp/test.py and show me its contents."
            }
        ]
    }

    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         "ollama",
        "anthropic-version": "2023-06-01"
    }

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/v1/messages",
            json=payload,
            headers=headers,
            timeout=TIMEOUT
        )

        if response.status_code != 200:
            print(f"  [FAIL] HTTP {response.status_code}: {response.text[:200]}")
            return False

        data           = response.json()
        content_blocks = data.get("content", [])
        tool_blocks    = [b for b in content_blocks if b.get("type") == "tool_use"]

        if not tool_blocks:
            print("  [FAIL] Model did not produce a tool_use block")
            print("         This means tool calling is not working correctly.")
            print("         Agentic Claude Code sessions will fail on file operations.")
            print(f"         Full response: {json.dumps(data, indent=2)}")
            return False

        tool_call  = tool_blocks[0]
        tool_name  = tool_call.get("name", "")
        tool_input = tool_call.get("input", {})

        print(f"  [PASS] Tool calling: model produced a valid tool_use block")
        print(f"         Tool called: {tool_name}")
        print(f"         Parameters:  {json.dumps(tool_input)}")

        # Sanity check: did it call the right tool with the right parameter?
        if tool_name == "read_file" and "path" in tool_input:
            print(f"         Tool name and parameter are correct.")
        else:
            print(f"         WARNING: Unexpected tool name or missing 'path' parameter.")
            print(f"         The model called a tool but not the expected one.")

        return True

    except Exception as e:
        print(f"  [FAIL] Request failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Claude Code + Ollama + Gemma 4 Setup Verification")
    print("=" * 60)

    checks = [
        check_ollama_health,
        check_model_available,
        check_messages_api,
        check_tool_calling,
    ]

    results = [check() for check in checks]

    print("\n" + "=" * 60)
    passed = sum(results)
    total  = len(results)

    if all(results):
        print(f"All {total} checks passed.")
        print("Claude Code + Ollama + Gemma 4 is ready.")
        print(f"\nLaunch with: claude")
        sys.exit(0)
    else:
        failed_checks = [i + 1 for i, r in enumerate(results) if not r]
        print(f"{passed}/{total} checks passed. Failed: {failed_checks}")
        print("Resolve the failures above before using Claude Code locally.")
        sys.exit(1)


if __name__ == "__main__":
    main()