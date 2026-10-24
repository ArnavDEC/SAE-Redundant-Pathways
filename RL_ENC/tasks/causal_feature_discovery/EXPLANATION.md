# EXPLANATION

Research journal for `causal_feature_discovery`. Last updated 2026-10-18.

## Where things stand

I built an environment where an agent (claude 4.8 or 5.5) gets a small blackbox model and has to work out which learned SAE units carries the causal effect behind it output.

It runs multiple experiments through a CLI (ablate, steer, patch, and also a factorial probe) and predicts the effect of some joint interventions on heldout parts. the grader checks those predictions against numbers the agent never sees.

The model is a toy. One attention head, 12 concept directions, a 48-unit SAE I trained on top. Epsidoes will either bea base case with correlated decoys, a polysemantic case, or a redundant pathway case, the last two built by hand. Among the rollouts, agent scores range from about 0.39 to 0.87, which is a spread I can work with. That being said said, I have roughly 50 graded rollouts spread over several versions of the task, and some  can't be compared with each other. 

Two things kinda irk me.  A correct share hedger with useless intervals tops out around 0.49. Second, nearly everything I say about why agents lost points comes from reading transcripts. The grader only gives me aggregates.

## The task as it is now

The model reads a sequence and outputs a logit difference. Twelve binary concepts are put in a 24D residual stream, and a trained top-k SAE with 48 units reads it. Unit IDs are swapped around every episode so nothing carries over.

In a normal episode one part drives the output with a gain drawn from 1.5 to 2.5. Three other concepts are train only proxies with each active 80% of the time when the trigger is on and 1% of the time when it's off, and they have no effect on the output. In the 'decorrelated' and 'long' distributions they're independent of the trigger. I went to three proxies after the first three agents beat a single one without trouble.

The polysemantic level overwrites SAE unit 47 with a direction that fires for both the target concept and a nuisance concept ('(target + 2) % 12'). The redundant level uses two concepts that each produce the full output on their own through a readout that max's out   so ablating one route alone shows almost nothing. About  1/4 of episodes are redundant. Each route also gets twelve higher activation decoy units that point along directions the readout ignores so there are 26 plausible candidates for two real ones.

The agent has 22 experiment calls. Ablate steer and patch take up to eight feature IDs at once. The 'factorial' command gives matched 'neither', 'target_only', 'second_only' and 'both' inputs. I explain the name below.

Submission: signed shares for up to eight units, plus 80% intervals for the ablation, steering and patch effects of the jointly intervened set. Held-out ablation is evaluated on the 'decorrelated' distribution.


## How it's scored

The final score is 0.5 × share recovery + 0.5 × (held-out prediction × share recovery). Prediction is '1 − mean interval score', and it is multiplied by share recovery so an agent cannot pick up prediction credit by naming the wrong units.

I moved to shares because identification stopped being a yes or no type of thing. At L0 naming one feature had only two choices and came out at 1.0 in almost every run, so it was cheap. Share recovery is a signed split across several units and ranged from 0.50 to 0.97, so it carries real information, which is why it gets half the weight. The other half goes to prediction, so a perfect-share hedger with no prediction credit can score at most 0.5.

## Journal

### Oct 16: first prototype and an exploit

The first version had 12 directly exposed features and one causal one. The first rollout ('1d2f281a0252') got perfect identifcation and a decent prediction score. The agent found feature 7 by looking at which feature separated high and low outputs, checked it with ablate, steer and patch, and ablated a control feature (F2) that did nothing. Ten more rollouts landed in a similar range, so that first score was typical and L0 was too easy.

Before building anything harder I wrote baselines over 1000 seeds, and they caught something. A random agent that picked a random feature and predicted '[0, 0.1]' for everything scored far higher than it should have, because wrong features often do have near-zero effects. I fixed it by making prediction credit depend on identification (grader v1.1). After the fix the oracle sat near the ceiling, observation-only greedy still identified the feature every time from correlations alone, a correct-feature hedger with huge intervals scored low, and random scored near zero. End-to-end rollouts through Docker matched the baseline results, so the fix works through the real grading path and not only in my baseline script.

That rollouts write-up said ablation was -0.99, and for a while I thought I had a sign bug. The tool output was +0.994. Typo in the agents prose, nothing wrong with the grader.

### Oct 17: moving to a real SAE

Evan pushed me to do the SAE work before tuning difficulty so I replaced the feature-space model with the attention model and a trained SAE and switched to share attribution. A unit's share euation is defined as below.

(activation change) × (decoder direction projected onto readout) = unit share

I'm not even completely certain this is the right definition. Shares from unit-wise ablation effects would be the obvious alternative, and I haven't compared them.

The first three agent rollouts averaged about 0.675 (mean share recovery 0.883, mean prediction 0.517). In one of them the agent found two correlated carriers (units 35 and 44) and wrote "context-gated synergy: 35 amplified 2.6x in context" and then said it was modeling the joint effect because it could only measure units one at a time. That pushed me to let agents probe joint interventions directly up to eight IDs, since the grader asks about joint effects.

While adding that I found the grader's patch target was wrong. It summed L1-normalized shares but the CLI reports the fraction of the full model gap. I changed the target to each unit's signed contribution divided by the measured effect. The mean mismatch on five seeds went from 0.088 to 0.010.

The joint-interface batch averaged about 0.742 (mean share recovery 0.870, mean prediction 0.703). Prediction was definitely higher than in the first batch (0.703 vs 0.517).

### Oct 17: decoys

I added a correlated decoy to the train distribution. It was active 45% of the time when the trigger is on and 2% when it's off. With 20 seeds greedy dropped welllll below the oracle with random and the hedger far below it as well. I also hardened the grader with a 64 kib cap on submissions, finite and bounded intervals, rejection of duplicate or out-of-range ids, and tests for perfect predictions with zero share, very wide intervals, and NaN or infinite values.

The first three agents on the decoy version all noticed the train only correlation, checked the decorrelated distribution and ruled the decoy out with an intervention. That's good, but the prompt pointed them at distribution shift, so I don't know if they'd have gone there on their own. I later replaced the single decoy with the three strong proxies described above. On the first polysemantic run on that version the agent found a unit with +0.93 correlation in train and -0.28 in decorrelated and dropped it then patched seven suspicious units together and saw only 1.3% gap recovery.

### Oct 17: polysemantic units

Before constructing a shared unit I checked whether the trained SAE already had one. It mostly did not. For 10 of the 12 target/nuisance pairs, no unit rose above baseline on both concepts, and the other two were weak. So I overwrote unit 47 with a shared target+nuisance direction. For one pair, its activation was 0, 1.5, 1.5 and 3.0 for neither, target-only, nuisance-only and both. Ablating it gives some sort of guaranteed effect on nuisance-only inputs because the shared decoder partly points along the readout. The right comparison is the target-on/off gap at a fixed nuisance state which is about 1.57 either way.

The first batch on this image was two polysemantic runs and four base runs. The two lower base runs missed on ablation. I recomputed the hidden targets offline and the misses were all on 'decorrelated'. One agent measured 0.488 there, close to the true 0.5125, but submitted '[0.58, 0.90]', which looks like it was centered on the train estimate. The prompt never said held-out ablation is scored on 'decorrelated'. Now it does.

I then had two runs on the clarified prompt, one polysemantic and one base. The polysemantic agent got shares right (0.954) but its ablation interval '[1.50, 2.10]` fell just under the true 2.24, so prediction was 0.178 and the score came out at about 0.562. In the base run, the agent claimed a shared unit (feature 30) was the constructed prototype. It wasn't, but feature 30 really does have shared activation and context-dependent effects in the unmodified SAE. A one-seed-per-target sweep found shared units in the base SAE for 1 of 12 pairs, against 12 of 12 for the prototype. The agent's mistake was assuming it knew how the unit was built, which it couldn't tell from the observations.

After I made that distinction explicit in the prompt a base run described clean separation, and a polysemantic run scored about 0.668 (share recovery about 0.955, prediction 0.400). That run had all three intervals covering their targets and still lost most of its prediction score to width, because the agent widened ablation after seeing batches between 1.95 and 3.18.

### Oct 17-18: redundant pathways

I added the redundant level because scores stayed high on the multi proxy decoy and I wanted something distribution shift can't fix. An agent has to see that two routes each suffice and that ablating one tells you nothing.

The first forced- edundant rollout did very well, but that was before the extra decoys. After I added them and cut the budget, the next rollout ('656ec0fd705e') scored about 0.388 with share recovery 0.500. The agent saw both single concept conditions produce the full output then treated the second concept as a distractor. My factorial output called it 'nuisance_only', while the prompt said nuisance concepts don't affect the output, so the wording contradicted the mechanism. I renamed it 'second_only' and said the label only describes which input is present. With that and a 28-call budget, 'eec27aadc7bd' found both routes.

I then set the budget to 22 and went back to random episode selection. The four-run batch (`20261017T231500`, ran on the 18th) gave: 

| Run | Level | Score | Share recovery | Prediction |
| --- | --- | ---: | ---: | ---: |
| `9855aaa916ee` | redundant | 0.4358 | 0.500 | 0.743 |
| `a1fbc488bceb` | polysemantic | 0.6346 | 0.760 | 0.670 |
| `19d842f7af54` | polysemantic | 0.7643 | 0.953 | 0.604 |
| `8a5342c2dac1` | base | 0.8667 | 0.939 | 0.846 |


The transcripts show three different failures. In the redundant run the agent noticed the second route and left it out because it was "not the target". In `a1fbc488bceb` the agent found that patching its two units recovered about 164% of the gap, concluded there was opposing mass somewhere, and submitted only two units with total absolute mass 0.76. The grader penalizes that, and share recovery came out 0.760. In `19d842f7af54` the shares were right but the agent's own ablation batches ranged from 1.58 to 1.95, and the interval it submitted didn't earn much.

The first failure wasn't about reasoning so much as dit was about the prompt. It didn't say that every causal route counts, or how to split credit between symmetric routes. Now it includes every causal route and gives symmetric sufficient routes equal shares. A fresh redundant rollout on the new wording (`1f581ed83eb6`) scored 0.817, with share recovery 1.000 and prediction 0.633. The agent used the factorial probe, bisected within activation-similar clusters, found units 15 and 37, and submitted 0.5 / 0.5. It used all 22 calls, never measured patch for the final set, and guessed about 0.95 for it. I can't tell which interval cost it the most points.

### Oct 18: larger batch on the final image

After that I ran about 25 more episodes on the 22-call image with the equal-share wording, mixed across levels. Mean scores by level:

| Level | Runs | Mean score | Share recovery | Prediction |
| --- | ---: | ---: | ---: | ---: |
| Base | 8 | 0.8797 | 0.94 | 0.84 |
| Polysemantic | 8 | 0.6695 | 0.86 | 0.58 |
| Redundant | 9 | 0.4553 | 0.54 | 0.39 |


## What the agents did

Mostly good causal practice. They ran control patches (one wrote "Control patches (features 5,8,42,37) were  about 0, setting the noise floor"), used the decorrelated distribution to rule out proxies, compared single and joint interventions, and separated activation from causal contribution. I read the transcripts for allmost all the batchruns. I didn't see anyone try to read grader files or touch the scoring. One agent tried to 'pip install numpy', got blocked by network isolation and carried on. 

## Checking the grader

I wrote 'baselines/evaluate_docker.py' to run the four baseline policies inside the task image and send every submission through the real 'grader.grade()' entry point, so it tests the production path and not just my baseline script.  I ran seeds 0-99 against 'rl-task-causal_feature_discovery' (ID 'sha256:3576a2dd2a164cf964cc80ef5074aaf4549ecbaf90eb292e8872a149934f708f', built 2026-10-18 at 00:25:08 -05:00). That gave 35 base, 47 polysemantic and 18 redundant episodes.

| Baseline | Mean score | Share recovery | Prediction |
| --- | ---: | ---: | ---: |
| Observation-only greedy | 0.132 | 0.264 | 0.000 |
| Random feature, zero predictions | 0.019 | 0.037 | 0.592 |
| Oracle (hidden-truth ceiling) | 0.972 | 0.982 | 0.980 |
| Correct-share hedger | 0.490 | 0.982 | 0.000 |

Random's prediction subscore looks high (0.592), but its total stays tiny because prediction gets multiplied by share recovery. The oracle reads the hidden truth, so it is only a ceiling check and not a real agent baseline. These are means over a mixed sample of levels so I can't say anything per level from them. I still haven't built a greedy baseline that uses interventions instead of just observing, which would be the fairer comparison for the agents.

Sixteen tests pass locally and in the image. They cover repeatability, the readout and decoy noncausality, distribution shift, share attribution, joint intervention limits, patch-target agreement, malformed submissions, wide-interval hedging and prediction gating.

### Reward Hacking
The grader is hard to game by guessing. Signed share recovery costs you for leaving out causal mass and for adding shares that aren't real. Prediction credit is multiplied by share recovery, and the interval score punishes both wide intervals and ones that miss the held-out target.

Very wide intervals drive prediction credit to zero, but an agent with recovered shares still earns the share-weighted portion of the score. The correct share hedger averages 0.491 (share recovery 0.982), so hedging remains a possible strategy when an agent is uncertain about its predictions, but it requires solving the share-recovery task first and still would be capped. The random baseline scores 0.019.

### Difficulty
Some tasks can look too easy. I settled on 22 for max calls because it gave me the best mix of successes and failures across the agent runs. If your definition of "too easy" differs from mine, or you think the model is doing too well at 22, or vice versa please feel free to decrease or increase it.

## What's wrong with it

The polysemantic and redundant levels are constructed by hand, not found by SAE training. The base SAE has a shared unit in about 1 of 12 pairs, so the constructed one is a guaranteed challenge and not a natural one.
Prediction credit is still weighted by share recovery, and a zero-prediction hedger with perfect shares can still score about 0.49. Lowering the share weight further, or giving prediction credit independently of shares, is something I could try.
The grader returns aggregates, so "this interval lost the points" is always inferred. A developer only per-prediction breakdown would settle several open questions and I still haven't added it.
The prompt has been rewritten several times, so maybe some of the agents' good behavior is probably the prompt's doing.


### Failure mOdes (tldr of what I discussed earlier)

These all come from reading transcripts, so the causes are my inference and not something the grader told me.

Dropping a real causal route. This was the big one on the redundant level. In '9855aaa916ee' the agent saw the second route and left it out because it was "not the target". In '656ec0fd705e' it saw both single-concept conditions produce the full output and still called the second concept a distractor. Some of that was on me tbf. The factorial label said 'nuisance_only' while the prompt said nuisance concepts don't affect the output, and the prompt never said every causal route counts. Both are fixed, but redundant is still the lowest level (0.4553), so wording wasn't the whole story.

Explaining away a weird number. In ;'1fbc488bceb' patching two units recovered about 164% of the gap. The agent decided there was opposing mass somewhere, but only submitted the two units it had. Total absolute mass was 0.76 and share recovery came out 0.760.

Centering on the wrong distribution. One early agent measured 0.488 for ablation on 'decorrelated', close to the true 0.5125, then submitted '[0.58, 0.90]', which looks centered on the train estimate. The prompt never said held-out ablation is scored on 'decorrelated'. Now it does.

Intervals that miss or run wide. One polysemantic agent got shares right (0.954) but submitted '[1.50, 2.10]' for ablation when the truth was 2.24, so prediction was 0.178. Another covered all three targets and still lost most of its prediction score (0.400) because it widened ablation after seeing batches from 1.95 to 3.18. In '19d842f7af54' the ablation batches ranged from 1.58 to 1.95 and the interval it submitted didn't earn much.

Running out of calls. In '1f581ed83eb6' the agent used all 22 calls finding units 15 and 37 and never measured patch for the final set. It guessed about 0.95. I can't tell if that guess was what cost it points.

Assuming how the environment was built. One base agent said feature 30 was the constructed polysemantic prototype. It wasn't, but feature 30 really does have shared activation and context-dependent effects in the normal SAE, and the agent couldn't tell the difference from what it saw.

Measuring one unit at a time for a joint effect. In a run the agent found two correlated carriers (units 35 and 44), said it was modeling the joint effect because it could only measure units one at a time, and ended up with a "context-gated synergy" story. That's why the CLI takes up to eight IDs now.

Mostly the agents are fine at the causal testing. They lose points on what to submit once the evidence gets messy.

## What I'd do next

1. Add a developer facing per-prediction breakdown to the grader. It currently returns only aggregate prediction scores, so claims about which interval lost points are inference. The breakdown would settle several open questions and is a small improvement I would definitely implement w/ more time
2. Build an intervention-using greedy baseline.

## Reproducing

```bash
# TODO: environment setup (WSL, requirements)

# tests
cd tasks/causal_feature_discovery/grader
python -m unittest test_private_model test_transformer_sae -v

# grade a rollout
./tools/view-grade.sh 1f581ed83eb6
# expected: score 0.817, share recovery 1.000, held-out prediction 0.633
```
