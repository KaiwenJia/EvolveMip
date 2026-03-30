from pyscipopt import Branchrule, SCIP_RESULT, SCIP_BRANCHDIR


class PseudoCostSumBranchrule(Branchrule):
    def __init__(self, model):
        self.model = model
        self.name = "PseudoCostSumBranchrule"
        self.priority = 1000000
        self.maxdepth = -1
        self.maxbounddist = 1.0

    def branchexeclp(self, allowaddcons):
        cands, candssol, candfrac, ncands, npriocands, nimplcands = self.model.getLPBranchCands()
        if npriocands == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        best_idx = 0
        best_score = float("-inf")

        for i in range(npriocands):
            var = cands[i]
            try:
                up = self.model.getVarPseudocost(var, SCIP_BRANCHDIR.UPWARDS)
                down = self.model.getVarPseudocost(var, SCIP_BRANCHDIR.DOWNWARDS)
                score = up + down
            except Exception:
                score = float("-inf")

            if score > best_score:
                best_score = score
                best_idx = i

        self.model.branchVar(cands[best_idx])
        return {"result": SCIP_RESULT.BRANCHED}


class PseudoCostProductBranchrule(Branchrule):
    def __init__(self, model):
        self.model = model
        self.name = "PseudoCostProductBranchrule"
        self.priority = 1000000
        self.maxdepth = -1
        self.maxbounddist = 1.0

    def branchexeclp(self, allowaddcons):
        cands, candssol, candfrac, ncands, npriocands, nimplcands = self.model.getLPBranchCands()
        if npriocands == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        best_idx = 0
        best_score = float("-inf")

        for i in range(npriocands):
            var = cands[i]
            try:
                up = self.model.getVarPseudocost(var, SCIP_BRANCHDIR.UPWARDS)
                down = self.model.getVarPseudocost(var, SCIP_BRANCHDIR.DOWNWARDS)
                score = (up + 1e-6) * (down + 1e-6)
            except Exception:
                score = float("-inf")

            if score > best_score:
                best_score = score
                best_idx = i

        self.model.branchVar(cands[best_idx])
        return {"result": SCIP_RESULT.BRANCHED}


class FractionalityWeightedBranchrule(Branchrule):
    def __init__(self, model):
        self.model = model
        self.name = "FractionalityWeightedBranchrule"
        self.priority = 1000000
        self.maxdepth = -1
        self.maxbounddist = 1.0

    def branchexeclp(self, allowaddcons):
        cands, candssol, candfrac, ncands, npriocands, nimplcands = self.model.getLPBranchCands()
        if npriocands == 0:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        best_idx = 0
        best_score = float("-inf")

        for i in range(npriocands):
            var = cands[i]
            try:
                up = self.model.getVarPseudocost(var, SCIP_BRANCHDIR.UPWARDS)
                down = self.model.getVarPseudocost(var, SCIP_BRANCHDIR.DOWNWARDS)
                frac = candfrac[i]
                frac_center = 0.5 - abs(frac - 0.5)   # 越接近 0.5 越大
                score = (up + down) * (1.0 + frac_center)
            except Exception:
                score = float("-inf")

            if score > best_score:
                best_score = score
                best_idx = i

        self.model.branchVar(cands[best_idx])
        return {"result": SCIP_RESULT.BRANCHED}