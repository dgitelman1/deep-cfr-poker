# Deep CFR Poker Bot

> A PyTorch **Deep Counterfactual Regret Minimization** agent that learns heads-up poker from scratch through Ray-distributed self-play — no human data.

**Topics:** `deep-cfr` · `counterfactual-regret-minimization` · `poker` · `reinforcement-learning` · `self-play` · `game-theory` · `nash-equilibrium` · `imperfect-information` · `pytorch` · `ray`

Built for the [CMU Data Science Club](https://github.com/cmu-dsc) poker hackathon (March 2026), on top of the club's [`poker-engine-2026`](https://github.com/cmu-dsc/poker-engine-2026).

## The game

A custom heads-up (1v1) poker variant:

- **27-card deck** — 9 ranks (`2–9, A`) × 3 suits.
- Each player is dealt **5 hole cards** and, on the flop, must **discard down to 2** — a large, information-rich decision most poker bots never face.
- Streets: pre-flop → flop (+ discard) → turn → river, then showdown.

## Approach

Classic CFR solves poker by tabulating regret for every information set — infeasible once the state space explodes (and the 5-card discard makes it explode fast). **Deep CFR** replaces those tables with neural networks:

1. **Self-play traversal** — External-Sampling Monte Carlo CFR walks the game tree, sampling the opponent's actions and enumerating the traverser's, computing counterfactual regret at each decision.
2. **Advantage (regret) network** learns to predict the regret of each action, generalizing to states never explicitly visited.
3. **Strategy network** distills the time-averaged strategy that CFR converges toward — this is the network used for play.

Implementation Details:

- **Ray-distributed self-play** — traversals are fanned out to worker processes; network weights are shared through Ray's zero-copy store and results are merged as each worker finishes.
- **Opponent-pool curriculum** — instead of always training against its current self, the bot faces a rotating pool of past checkpoints (mostly the latest, occasionally a random older one) to avoid overfitting to a single opponent.
- **Dual-stream network** — card features and betting/state features are encoded separately, then merged through residual blocks.
- **Suit-isomorphism abstraction** — each state is folded to a canonical form under all 6 suit permutations, so the network only ever learns one representative of each strategically identical situation.

An earlier prototype (`experiments/`) used a from-scratch bitboard engine and hand evaluator for fast single-threaded tree search. It worked but was fiddly to keep correct, so the project moved to the gym-based engine plus Ray parallelism; the bitboard code is kept for reference.

## Repository layout

```
├── core/                  # Deep CFR agent, dual-stream network, encoding & canonicalization
├── train/                 # Training entry points (train_batched.py = Ray trainer)
├── parallelism_training/  # Ray remote traversal worker
├── submission/            # Tournament agent (inference only)
└── experiments/           # Early bitboard prototype (engine, evaluator, CFR trainer)
```

This agent runs on top of the [`cmu-dsc/poker-engine-2026`](https://github.com/cmu-dsc/poker-engine-2026) engine, which provides `gym_env`, the match harness, the base `Agent` class, and the baseline bots. Clone it alongside this repo, then drop these modules in.

## Quickstart

```bash
pip install -r requirements.txt

# Train with distributed self-play (Ray) against a pool of past checkpoints
python -m train.train_batched --mixed --iterations 10000 --traversals 200
```

Checkpoints are written to `models_final/`, and a new opponent snapshot is banked into the pool periodically. Trained weights are not included; run the trainer to produce one, then point `submission/player.py` at it.

## Acknowledgements

The game engine, gym environment, and match harness are from the **CMU Data Science Club** competition: [`cmu-dsc/poker-engine-2026`](https://github.com/cmu-dsc/poker-engine-2026). All Deep CFR agent, training, parallelism, and (experimental) bitboard code here is my own.
