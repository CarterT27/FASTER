"""Test script for OpenRouter API."""

import os
import requests
import json

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

print(f"Using API key: {api_key[:5]}...{api_key[-5:]}")

try:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "deepseek/deepseek-chat:free",
            "messages": [{"role": "user", "content": "Say hello world"}],
            "temperature": 0.0,
        },
    )

    print(f"Status code: {response.status_code}")
    print(f"Response headers: {json.dumps(dict(response.headers), indent=2)}")
    print(f"Response body: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {str(e)}")
    if hasattr(e, "response"):
        print(f"Response status code: {e.response.status_code}")
        print(f"Response headers: {json.dumps(dict(e.response.headers), indent=2)}")
        print(f"Response body: {e.response.text}")
