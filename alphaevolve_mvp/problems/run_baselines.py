from pyscipopt import Model

model = Model()
model.readProblem("xxx.mps")

branchrule = CustomBranchingRule(model)
model.includeBranchrule(
    branchrule,
    branchrule.name,
    "custom branching rule",
    priority=branchrule.priority,
    maxdepth=branchrule.maxdepth,
    maxbounddist=branchrule.maxbounddist,
)

model.optimize()