MODES = ("probe", "smoke", "evaluate", "benchmark", "accept", "report")


def make_execution_plan(mode):
    if mode == "all":
        return list(MODES)
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    return [mode]
