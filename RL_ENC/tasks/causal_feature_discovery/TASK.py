import json
import secrets
import subprocess
import time
from pathlib import Path


seed_path = Path("/grader/episode_seed.json")
seed = secrets.randbits(64)
seed_path.write_text(json.dumps({"seed": seed}))
seed_path.chmod(0o600)

server_log = open("/grader/experiment_server.log", "w")
server = subprocess.Popen(
    ["python3", "/opt/causal-feature-lab/server.py"],
    stdin=subprocess.DEVNULL,
    stdout=server_log,
    stderr=server_log,
    start_new_session=True,
)
server_log.close()

socket_path = Path("/tmp/causal-feature-lab.sock")
deadline = time.monotonic() + 120
while not socket_path.exists():
    if server.poll() is not None:
        raise RuntimeError("Experiment service exited during startup")
    if time.monotonic() >= deadline:
        server.terminate()
        raise TimeoutError("Experiment service did not become ready")
    time.sleep(0.05)

Path("/TASK.txt").write_text(
    """# Causal feature discovery: causal shares, polysemantic features, and redundant pathways

You are given black-box access to a small attention model and its trained
sparse autoencoder (SAE) through the `experiment` CLI. The model's behavior is
caused by one or more hidden input concepts. Their effects may be represented
by multiple SAE units, and units may respond to concepts that do not affect the
output. SAE feature IDs are randomized for this episode. Discover which
learned features carry the behavior and estimate their signed shares using
experiments.

The model emits a logit difference: positive values favor output A; negative
values favor output B. The target behavior is associated with a trigger that
may appear in an input sequence. You can inspect sparse activations but cannot
access transformer weights, SAE weights, the hidden causal concept, or the
grader's held-out share vector.

## Experiment interface

- `experiment list-features`
- `experiment observe --distribution train --count 32`
- `experiment factorial --count 8 [--feature-ids ID [ID ...]]`
- `experiment ablate --feature-ids ID [ID ...] --distribution train --count 64`
- `experiment steer --feature-ids ID [ID ...] --scale 0.5 --count 64`
- `experiment patch --feature-ids ID [ID ...] --count 64`

The supported distributions are `train`, `decorrelated`, and `long`. Each
experiment call can use at most 64 inputs and at most 8 distinct feature IDs
for an intervention. Pass one or more IDs after `--feature-ids`; the listed set
is intervened on jointly. You have a budget of 22 experiment calls; feature
listing does not use the budget. Responses are JSON. If the budget is exhausted
or a request is invalid, the CLI reports an error.

`observe` returns the 48 SAE feature activations and model logit differences.
The feature vector positions correspond to the randomized feature IDs.
`factorial` returns observations for four matched conditions: neither concept,
target only, second concept only, and both. Other input concepts are held fixed
across each matched set. The second concept is a controlled comparison factor;
it can be noncausal or provide another causal route, depending on the episode.
The condition labels describe which inputs are present, not their causal role.
Use `--feature-ids` to also measure the selected set's joint ablation effect in
every condition. The count is per condition, up to 16.
`ablate` reports `clean logit_diff - ablated logit_diff`. A positive result
means zeroing the feature lowered the logit difference; a negative result means
it raised the logit difference.
`steer` returns the change after adding the requested activation to every
selected feature. `patch` returns the fraction of the clean/corrupt behavior
gap recovered by jointly patching the selected feature activations from
trigger-present sources into trigger-absent targets. Each measurement includes
the selected IDs and a standard error. Joint probes let you test the same
feature set for which you will submit held-out predictions; the held-out
outcomes themselves remain hidden.

Episode mechanisms vary. Multiple SAE units may respond to the same input
factor, and a unit's effect can depend on which other factors are present.
Activation strength and a single ablation result are not enough to establish
causal contribution. Use matched conditions and interventions to infer which
features carry behavior and how their contributions combine.

The `train` distribution may include noncausal proxy concepts correlated with
the target trigger. These proxies do not affect the model's output. In
`decorrelated`, their activations are sampled independently of the trigger,
while background concept rates are higher. `long` repeats the sequence to test
length generalization. The target concept's trigger rate is held fixed across
distributions. Use distribution shifts and interventions to identify which
features carry causal contribution. Features can share or oppose causal
contribution; use signed shares, not only activation correlation.

## Submission

Create `/workdir/submission.json` with up to 8 feature-share estimates and
predictions for the joint set of features you report. A signed share estimates
that SAE unit's contribution to the held-out target logit gap, normalized so
the true shares have total absolute mass 1. Omit unreported feature IDs (their
predicted share is zero).

Shares account for every causal route to the target behavior, not only the
route associated with the first trigger. If the second concept provides an
alternate route, include its causal SAE feature even when the first route is
also sufficient. When two routes are symmetric and either alone produces the
same full behavior gap, assign them equal signed shares.

1. Mean logit-difference change when ablating all reported features.
2. Mean logit-difference change when steering all reported features by 0.5.
3. Fraction of the clean/corrupt gap recovered when jointly patching all
   reported features.

Use this JSON shape (replace the example values):

```json
{
  "components": [
    {"feature_id": 3, "share": 0.65},
    {"feature_id": 9, "share": 0.25},
    {"feature_id": 14, "share": -0.10}
  ],
  "predictions": [
    {"test": "ablate", "lower": 0.0, "upper": 1.0},
    {"test": "steer", "lower": 0.0, "upper": 1.0},
    {"test": "patch", "lower": 0.0, "upper": 1.0}
  ],
  "rationale": "Briefly explain the evidence and interventions you used."
}
```

Provide honest 80% uncertainty intervals. The rationale is recorded but not
scored. The grader rewards accurate recovery of the signed share distribution
and held-out predictions for your selected feature set, with equal 50% weights
for share recovery and prediction quality; prediction credit is weighted by
share recovery. The held-out ablation prediction is evaluated on
fresh inputs from the `decorrelated` distribution. Use measurements from that
distribution to estimate its center, and let your interval reflect the
variation you observe. The grader's held-out outcomes are not returned by the
CLI.
"""
)
