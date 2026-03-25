import re
from typing import Type

from openai import OpenAI
from pyscipopt import Branchrule, SCIP_RESULT, SCIP_BRANCHDIR


class LLMBranchruleGenerator:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def build_prompt(self, baseline_code: str, feedback_text: str = "") -> str:
        return f"""
You are optimizing a SCIP branching rule for MILP set cover instances.

Task:
Write complete valid Python code that defines exactly one class:

    class GeneratedBranchingRule(Branchrule):

The class must implement:
- __init__(self, model)
- branchexeclp(self, allowaddcons)

Goal:
Reduce average solving time and node count on set cover MPS instances.

Available APIs:
- model.getLPBranchCands()
- model.getVarPseudocost(var, SCIP_BRANCHDIR.UPWARDS)
- model.getVarPseudocost(var, SCIP_BRANCHDIR.DOWNWARDS)
- model.branchVar(var)

Hard constraints:
- Must return a valid SCIP result dict
- Must branch on one variable if LP branching candidates exist
- Must produce complete valid Python code
- Output only one Python code block
- The generated class name must be GeneratedBranchingRule
- Keep the rule simple and computationally cheap
- Prefer modifying only the variable scoring logic
- Do not use file I/O, networking, subprocesses, or external packages

You may use:
- Branchrule
- SCIP_RESULT
- SCIP_BRANCHDIR

Feedback from previous evaluation:
{feedback_text}

Baseline code:
```python
{baseline_code}
"""

    def generate_code(self, baseline_code: str, feedback_text: str = "") -> str:
        prompt = self.build_prompt(baseline_code, feedback_text)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=3000,
            top_p=0.95,
        )

        text = resp.choices[0].message.content or ""
        text = text.strip()
        return self.extract_python_code(text)

    @staticmethod
    def extract_python_code(text: str) -> str:
        m = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    @staticmethod
    def validate_code(code: str) -> None:
        compile(code, "<generated_branchrule>", "exec")

    @staticmethod
    def load_branchrule_class(code: str) -> Type[Branchrule]:
        namespace = {}
        exec_globals = {
            "__builtins__": __builtins__,
            "Branchrule": Branchrule,
            "SCIP_RESULT": SCIP_RESULT,
            "SCIP_BRANCHDIR": SCIP_BRANCHDIR,
        }

        exec(code, exec_globals, namespace)

        cls = namespace.get("GeneratedBranchingRule")
        if cls is None:
            raise ValueError("Generated code does not define GeneratedBranchingRule")

        if not issubclass(cls, Branchrule):
            raise TypeError("GeneratedBranchingRule is not a subclass of Branchrule")

        if not hasattr(cls, "branchexeclp"):
            raise ValueError("GeneratedBranchingRule does not implement branchexeclp")

        return cls