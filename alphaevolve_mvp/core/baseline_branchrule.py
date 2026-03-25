from pyscipopt import Branchrule, SCIP_RESULT


class BaselinePseudoCostBranchrule(Branchrule):
    """
    Baseline LP branching rule:
    - get LP branching candidates
    - score each priority candidate by pseudo cost
    - branch on the best-scoring candidate at its LP solution value

    This class is intentionally simple so that LLM-generated variants
    can modify only the variable scoring logic while keeping the overall
    branching flow valid.
    """

    def __init__(self, model):
        self.model = model

    def branchexeclp(self, allowaddcons):
        (
            branch_cands,
            branch_cand_sols,
            branch_cand_fracs,
            ncands,
            npriocands,
            nimplcands,
        ) = self.model.getLPBranchCands()

        if npriocands == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        best_idx = 0
        best_score = float("-inf")

        for i in range(npriocands):
            var = branch_cands[i]

            try:
                up = self.model.getVarPseudocost(var, +1.0)
                down = self.model.getVarPseudocost(var, -1.0)

                # Baseline scoring rule:
                # prefer variables with strong pseudo cost in both directions
                if up > 0.0 and down > 0.0:
                    score = up * down
                else:
                    score = up + down
            except Exception:
                score = float("-inf")

            if score > best_score:
                best_score = score
                best_idx = i

        self.model.branchVarVal(branch_cands[best_idx], branch_cand_sols[best_idx])
        return {"result": SCIP_RESULT.BRANCHED}