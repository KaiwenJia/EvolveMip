import json

from baseline_branchrule import BaselinePseudoCostBranchrule
from branchrule_eval import benchmark_branchrule, summarize_results
from llm_codegen import LLMBranchruleGenerator


def main():
    # 1) 先确认 import 和符号都存在
    print("Baseline class:", BaselinePseudoCostBranchrule.__name__)
    print("benchmark_branchrule:", benchmark_branchrule.__name__)
    print("summarize_results:", summarize_results.__name__)
    print("LLM generator:", LLMBranchruleGenerator.__name__)

    # 2) 改成你真实可用的一个小 MPS 文件
    mps_files = [
        "/DATA/disk2/chenyitian/AutoConfig/jiakaiwen/TspEvolve/alphaevolve_mvp/data/setcover/test_200r_400c_0.01d_instance_49.mps",
    ]

    # 3) baseline 冒烟测试
    baseline_rows = benchmark_branchrule(
        mps_files,
        branchrule_cls=BaselinePseudoCostBranchrule,
        time_limit=10.0,
    )
    baseline_summary = summarize_results(baseline_rows)

    print("\n=== Baseline Rows ===")
    print(json.dumps(baseline_rows, indent=2, ensure_ascii=False))

    print("\n=== Baseline Summary ===")
    print(json.dumps(baseline_summary, indent=2, ensure_ascii=False))

    # 4) 跳过真实 LLM，直接测试动态加载 GeneratedBranchingRule
    generator = LLMBranchruleGenerator(
        api_key="DUMMY",
        base_url="https://example.com/v1",
        model_name="dummy-model",
    )

    generated_code = '''
from pyscipopt import Branchrule, SCIP_RESULT

class GeneratedBranchingRule(Branchrule):
    def __init__(self, model):
        self.model = model
        self.name = "GeneratedBranchingRule"
        self.priority = 1000000
        self.maxdepth = -1
        self.maxbounddist = 1.0

    def branchexeclp(self, allowaddcons):
        cands, candssol, candfrac, nlpcands, npriolpcands, nfracimplvars = self.model.getLPBranchCands()
        if npriolpcands == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}
        self.model.branchVar(cands[0])
        return {"result": SCIP_RESULT.BRANCHED}
'''.strip()

    generator.validate_code(generated_code)
    GeneratedCls = generator.load_branchrule_class(generated_code)
    print("\nGenerated class loaded:", GeneratedCls.__name__)

    candidate_rows = benchmark_branchrule(
        mps_files,
        branchrule_cls=GeneratedCls,
        time_limit=10.0,
    )
    candidate_summary = summarize_results(candidate_rows)

    print("\n=== Candidate Rows ===")
    print(json.dumps(candidate_rows, indent=2, ensure_ascii=False))

    print("\n=== Candidate Summary ===")
    print(json.dumps(candidate_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()