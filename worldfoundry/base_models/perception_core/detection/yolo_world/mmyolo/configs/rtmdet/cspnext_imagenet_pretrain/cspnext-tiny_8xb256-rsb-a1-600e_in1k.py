_base_ = "./cspnext-s_8xb256-rsb-a1-600e_in1k.py"

model = {"backbone": {"deepen_factor": 0.167, "widen_factor": 0.375}, "head": {"in_channels": 384}}
