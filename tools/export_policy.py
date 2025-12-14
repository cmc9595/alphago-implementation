# tools/export_policy.py
from pathlib import Path
import torch
from bots.nn.policy_net import PolicyNet

ckpt = Path("checkpoints/sl_policy/policy_sl_latest.pt")
out  = Path("checkpoints/sl_policy/policy_sl_infer.pt")

device = "cpu"
state = torch.load(ckpt, map_location=device)

model = PolicyNet()
model.load_state_dict(state["model"])
model.eval()

torch.save(model.state_dict(), out)
print("saved:", out)
