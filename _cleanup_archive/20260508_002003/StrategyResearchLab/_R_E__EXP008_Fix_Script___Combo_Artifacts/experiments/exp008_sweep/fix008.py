import json

path = "experiments/exp008_sweep/walk_forward_log.json"
with open(path) as f:
    log = json.load(f)

# Add window_id to holdout
log["final_holdout"]["window_id"] = "holdout"
log["final_holdout"]["status"] = "pending"

with open(path, "w") as f:
    json.dump(log, f, indent=2)

print("Fixed: added window_id to holdout")
print("Holdout:", log["final_holdout"])