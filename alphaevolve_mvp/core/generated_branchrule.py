#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from openai import OpenAI

API_KEY = os.getenv("COMPANY_API_KEY", "YOUR_API_KEY")
BASE_URL = os.getenv("COMPANY_BASE_URL", "YOUR_BASE_URL")
MODEL_NAME = os.getenv("COMPANY_MODEL_NAME", "gemini-2.5-flash")

OUTPUT_FILE = "generated_branchrule.py"


def extract_python_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def main():
    prompt = """
Write complete valid Python code that defines exactly one class:

class GeneratedBranchingRule(Branchrule)

Requirements:
- inherit from Branchrule
- implement __init__(self, model)
- implement branchexeclp(self, allowaddcons)
- use pseudocost-based branching
- return valid SCIP result dict
- output only one python code block

The rule should:
- get LP branching candidates
- score candidates using pseudocosts
- choose one variable to branch on
- keep the logic simple and computationally cheap
""".strip()

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2000,
        top_p=0.95,
    )

    text = resp.choices[0].message.content or ""
    text = text.strip()

    code = extract_python_code(text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(code)

    print("=== Raw Response ===")
    print(text)
    print("\n=== Saved Code File ===")
    print(os.path.abspath(OUTPUT_FILE))


if __name__ == "__main__":
    main()