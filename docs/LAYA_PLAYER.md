# Laya player benchmark

Measured 2026-09-28 with `python -m src.routing.laya_player_benchmark --repeats 3` (convaiinnovations/laya, device auto).
Machine: Windows-10-10.0.26200-SP0. Answers are checked against `truth` (the rule player) at lead 14.

| Style | Correct | Median | p95 |
|---|---|---|---|
| choice, raw numbers | 7/16 | 37ms | 55ms |
| yes/no, raw numbers | 5/16 | 38ms | 51ms |
| yes/no, sentence | 13/16 | 36ms | 43ms |

## Answers

| Scenario | Expected | choice, raw numbers | yes/no, raw numbers | yes/no, sentence |
|---|---|---|---|---|
| nothing ahead | hold | jump (wrong) | duck (wrong) | duck (wrong) |
| cactus far | hold | jump (wrong) | hold | hold |
| cactus inside | jump | jump | hold (wrong) | jump |
| cactus just inside | jump | jump | hold (wrong) | jump |
| cactus just outside | hold | jump (wrong) | duck (wrong) | hold |
| cactus inside, fast | jump | jump | hold (wrong) | hold (wrong) |
| cactus outside, slow | hold | jump (wrong) | hold | hold |
| cactus group inside | jump | jump | hold (wrong) | jump |
| cactus group outside | hold | jump (wrong) | hold | hold |
| low bird inside | jump | jump | duck (wrong) | jump |
| low bird far | hold | jump (wrong) | hold | hold |
| head-height bird inside | duck | jump (wrong) | hold (wrong) | hold (wrong) |
| head-height bird far | hold | jump (wrong) | hold | hold |
| high bird, cactus far | hold | jump (wrong) | duck (wrong) | hold |
| in the air, cactus inside | jump | jump | hold (wrong) | jump |
| passing over, in the air | jump | jump | hold (wrong) | jump |

Sentence for the first cactus inside:

> The dinosaur is on the ground, running at speed 8.0. Nearest obstacle: a cactus. Distance to the cactus: 40 pixels, which is less than the 104-pixel window.
## Live play

`dino-ai play --player laya --episodes 5` on the vendored game (RTX 5070, 2026-09-28). The game keeps running while Laya decides, and scores are the game's own score. Chrome's `chrome://dino` reads the same state (`--game chrome`), but these runs used the vendored copy.

| Player | Cut | Game scores | Mean | Per decision |
|---|---|---|---|---|
| Laya | 0.5 | 300, 46, 322, 138, 637 | 289 | ~40ms |
| Laya | **0.7** | 4536, 2498, 4361, 624, 7341 | **3872** | ~39ms |
| Rule, 45ms delay, lead 12 | – | 23289, 6614, 3822, 4323, 8321 | 9274 | 45ms |
| Rule, 45ms delay, lead 14 | – | 6546, 1033, 6945 (sweep still running) | – | 45ms |

For scale, the best RL agent (round 1, `dqn-a2-eps`) survives about 230 steps of 50ms, roughly 15 seconds. Laya at the 0.7 cut survived 41–386 seconds.

**Why the cut is 0.7:** at 0.5, every crash was an early jump that landed on a cactus. Over those 5 games, Laya's jump detector reached 0.5–0.69 on 8% of sentences that said "more than the window" (39 of 128 jumps started too early). It reached 0.7 or more on 96% of "less than" sentences, and never on "more than" or on an empty road. At 0.7, duck false alarms on an empty road also drop from 76% to 8%. The single-decision benchmark above falls from 14/16 to 13/16, because a borderline "inside" can now answer hold. In live play that costs little: each obstacle gets several decisions before the take-off point, while one early jump is fatal.

**What ends Laya's games now:** in all 5 games at 0.7, a bird at head height at top speed. Laya answered jump (0.70–0.72) instead of duck (0.62–0.68), the same confusion as the benchmark's "head-height bird inside". Its detectors mostly follow "less than" vs "more than" and barely separate the kinds of obstacle: rewording the sentence and the questions (about 30 variants) did not fix it.

**Next:**
- Ask only the question that fits the obstacle, so Laya judges the distance and the code names the move.
- Try lead 12 with Laya.
- A fast drop (speed drop) when an obstacle follows a landing closely: the rule player's crashes are birds right after a jump at top speed.
