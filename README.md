# Restricted Boltzmann Machines — user preference segmentation

A from-scratch **Restricted Boltzmann Machine** (TypeScript + TensorFlow.js) that learns, **without labels**, the latent tastes of customers of a food-delivery company — and rediscovers a preference that is invisible in the raw data: **loving spicy food** 🌶️.

This repository is the companion code for the article (in Spanish):

> **[Segmentación de preferencias de usuarios mediante Restricted Boltzmann Machines](https://arrowfunction.blog/articulos-y-noticias/segmentaci%C3%B3n-de-preferencias-de-usuarios-mediante-restricted-boltzmann-machines-b5946e)** — arrowfunction.blog

---

## Table of contents

- [How an RBM works](#how-an-rbm-works)
  - [The parts](#the-parts)
  - [The phases](#the-phases)
  - [Training: Contrastive Divergence (CD-1)](#training-contrastive-divergence-cd-1)
  - [Regularization](#regularization)
- [The experiment](#the-experiment)
  - [Dataset design](#dataset-design)
  - [What the network discovers on its own](#what-the-network-discovers-on-its-own)
  - [Results](#results)
  - [The technical lesson: saturated hidden biases](#the-technical-lesson-saturated-hidden-biases)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Honest caveats](#honest-caveats)
- [References](#references)

---

## How an RBM works

An RBM is one of the simplest generative neural networks: two layers of binary units connected by a single weight matrix, with **no connections inside a layer** (that is the "restricted" part). The full model lives in [src/model/rbm.ts](src/model/rbm.ts) and is self-contained — ~130 lines including comments.

### The parts

| Component | Shape | Role |
|---|---|---|
| **Visible units** `v` | `[nVisible]` | The observed data — here, one binary unit per dish (1 = the user orders it) |
| **Hidden units** `h` | `[nHidden]` | The latent factors the network invents to explain the data |
| **Weights** `W` | `[nVisible, nHidden]` | How strongly each dish and each latent factor excite each other |
| **Visible biases** `bv` | `[nVisible]` | Base popularity of each dish |
| **Hidden biases** `bh` | `[nHidden]` | Base tendency of each factor to switch on |

The model assigns every joint configuration `(v, h)` an **energy** — low energy means "plausible combination":

```
E(v, h) = −bv·v − bh·h − v·W·h
```

Because there are no intra-layer connections, all hidden units are **conditionally independent given the visible layer** (and vice versa). That yields two closed-form conditional distributions, each a single matrix multiply plus a sigmoid:

```
P(h = 1 | v) = σ(v·W  + bh)     // "reading": data → latent factors
P(v = 1 | h) = σ(h·Wᵀ + bv)     // "generating": latent factors → data
```

### The phases

Every interaction with an RBM is a back-and-forth between the two layers:

1. **Upward pass (inference).** Clamp a user's preference vector on the visible layer and compute `P(h | v)` — the probability that each latent factor is active for that user. This is how the trained model is used: the hidden activations *are* the segmentation.
2. **Downward pass (generation / reconstruction).** From a hidden state, compute `P(v | h)` — what the model believes the user would order. Comparing reconstruction to the original input measures how well the model has captured the data.
3. **Sampling.** Probabilities can be turned into binary states by Bernoulli sampling ([`sample`](src/model/rbm.ts#L73-L77)). During training the hidden layer is sampled so it acts as a **binary bottleneck** that forces compression.

### Training: Contrastive Divergence (CD-1)

Exact maximum-likelihood training is intractable (it requires the model's full equilibrium distribution), so RBMs use Hinton's **Contrastive Divergence** shortcut: run the up–down chain for just **one step** and compare where it starts with where it lands ([`cd1`](src/model/rbm.ts#L98-L120)):

```
v⁰ ── P(h|v) ──► ph0 ── sample ──► h0 ── P(v|h) ──► pv1 ── P(h|v) ──► ph1
     positive phase                        negative phase
```

- **Positive phase** `v⁰ᵀ·ph0`: correlations the *data* exhibits — reinforce them.
- **Negative phase** `pv1ᵀ·ph1`: correlations the *model dreams up* on its own — suppress them.

The updates are the difference between the two, averaged over the batch:

```
ΔW  = (v⁰ᵀ·ph0 − pv1ᵀ·ph1) / batch
Δbv = mean(v⁰  − pv1)
Δbh = mean(ph0 − ph1)
```

Intuitively: *make the data more probable than the model's fantasies*. When the fantasies match the data, the gradient vanishes and training converges.

Two practical choices follow Hinton's guide (2010): only the hidden layer is sampled (`h0`), while the visible reconstruction uses probabilities (`pv1`) — this reduces gradient noise and produces sharper filters.

### Regularization

The training loop ([src/model/train.ts](src/model/train.ts)) supports mini-batch SGD plus three optional regularizers:

- **L2 weight decay** — keeps filters small and clean.
- **Sparsity penalty** — pushes each hidden unit's mean activation toward a low target, encouraging units to specialize.
- **Hidden bias decay** — the one that made this experiment work; see [the technical lesson](#the-technical-lesson-saturated-hidden-biases).

---

## The experiment

Can an RBM segment users by taste **without ever seeing a label**? We give it only binary order histories and check whether the hidden units recover the ground-truth latent traits used to generate the data.

### Dataset design

12 dishes across 4 cuisines, plus one **cross-cutting** trait ([src/data/dataset.ts](src/data/dataset.ts)):

| Cuisine | Dishes | Spicy 🌶️ |
|---|---|---|
| mexicano | Tacos al pastor, Quesadillas, **Totopos con jalapeños** | Totopos |
| italiano | Rigatoni Carbonara, Lasaña, Risotto | *(none — control cuisine)* |
| cuchara | Fabada, **Lentejas con chorizo picante**, Sopas de ajo | Lentejas |
| asiático | **Kimchi Chigae**, Sushi, **Pad Thai** | Kimchi, Pad Thai |

Each synthetic user has **two independent latent traits**: a `cuisine` (one-hot) and a `spicy` flag (independent coin flip). A spicy lover raises their probability on **all** spicy dishes, regardless of cuisine — that cross-cutting overlap is exactly what the network must rediscover. Italian cuisine has no spicy dish of its own: it is an orthogonal control proving that "spicy" is a genuine second axis and not just another cuisine.

The training set has 240 users (4 cuisines × 60) with 3% bit-flip noise. A separate demo set contains 10 named users arranged as **matched pairs**: same cuisine, differing only in the spicy trait.

### What the network discovers on its own

With 6 hidden units, 5 come out interpretable — a clean **bijection between cuisines and units**, plus a dedicated spicy detector:

```
Hidden 1 = cuchara    Hidden 3 = mexicano    Hidden 5 = italiano
Hidden 4 = asiatico   Hidden 6 = SPICY 🌶️     (Hidden 2 redundant)
```

The spicy unit is identified automatically (largest mean-weight gap between spicy and non-spicy dishes, **+2.83**). Its learned weights are a textbook filter — positive on all 4 spicy dishes from 3 different cuisines, negative on the rest:

| Dish | Weight on H6 |
|---|---|
| Kimchi Chigae 🌶️ | **+2.46** |
| Totopos con jalapeños 🌶️ | **+2.39** |
| Pad Thai 🌶️ | **+1.40** |
| Lentejas con chorizo picante 🌶️ | **+1.32** |
| Sopas de ajo | −2.08 |
| Fabada | −2.81 |

**Nobody labeled "spicy" anywhere in the data** — the network infers it purely from co-occurrence patterns.

### Results

On the 10 demo users:

- **Cuisine: 10/10** assigned to their true cuisine via the dominant hidden unit.
- **Spicy: 10/10** classified correctly (non-spicy < 0.5, spicy > 0.5 on H6).
- Matched pairs from the same cuisine are separated **only** by the spicy unit — e.g. mexicano: Arancha 0.47 vs Maria **1.00**.
- The hardest case works: **Marta** orders Italian food (no spicy dish of its own), yet the network flags her as spicy (H6 = 0.72) because she crosses over to Lentejas and Kimchi. Spicy is truly transversal.
- Users express **combinations** of factors, not a single cluster: a spicy asiático lights up his cuisine unit *and* H6 at once — enabling non-obvious recommendations like suggesting spicy dishes from *other* cuisines.

### The technical lesson: saturated hidden biases

At first the cuisines did **not** separate. The culprit was not the model but **saturated hidden biases**: some units grew a huge bias (bh ≈ +5.4), stayed "on by default" for every user (σ(5.4) ≈ 0.996), and their activation stopped meaning anything. The fix was decaying the hidden biases (`hiddenBiasDecay`): max|bh| drops from ~5.8 to ~1.5, activations are again governed by evidence, and each cuisine falls into its own unit. The moral: in unsupervised models, **making the units interpretable is half the work**. (Full write-up in the comments of [src/train.ts](src/train.ts#L12-L31).)

---

## Quick start

Requires Node **v22.15.0** (see [.nvmrc](.nvmrc)).

```bash
npm install

# 1. Generate the synthetic data (deterministic — seeds fixed in code)
npm run generate
#    → data/train.json      240 users (4 cuisines × 60), seed 42, noise 0.03
#    → data/new-users.json  10 demo users,               seed 22, noise 0

# 2. Train the RBM (defaults shown — identical to `npm run train`)
npx tsx src/train.ts 6 500 0.1 32 42 0.02
#                    │ │   │   │  │  └─ hiddenBiasDecay
#                    │ │   │   │  └──── seed (init + shuffle + CD-1 sampling)
#                    │ │   │   └─────── batchSize
#                    │ │   └─────────── learningRate
#                    │ └─────────────── epochs
#                    └───────────────── nHidden

# 3. Inference + report (weight matrix, preferences, hidden activations)
npx tsx src/index.ts 6      # equivalent to `npm run dev`
#                    └─ nHidden: must match the trained model (loads data/model-6.json)
```

### `train` parameters (positional)

| Pos | Parameter | Default | What it does |
|-----|-----------|---------|--------------|
| 1 | `nHidden` | `6` | Hidden units. ≥5 for 4 cuisines + spicy. |
| 2 | `epochs` | `500` | Passes over the dataset. |
| 3 | `learningRate` | `0.1` | CD-1 learning rate. |
| 4 | `batchSize` | `32` | Mini-batch size (SGD). |
| 5 | `seed` | `42` | Global seed → fully deterministic training. |
| 6 | `hiddenBiasDecay` | `0.02` | Decays hidden biases to disentangle the cuisines. |

---

## Project structure

```
src/
├── model/
│   ├── rbm.ts          # The RBM itself: energy, P(h|v), P(v|h), sampling, CD-1
│   ├── train.ts        # Training loop: mini-batch SGD + regularizers
│   └── persistence.ts  # Save/load model as JSON
├── data/
│   ├── dataset.ts      # Generative model: dishes, cuisines, spicy trait, seeded RNG
│   ├── loader.ts       # Read the JSON datasets
│   └── generate.ts     # Produce data/train.json and data/new-users.json
├── train.ts            # CLI: train and save a model
└── index.ts            # CLI: inference report with colored terminal tables
data/
├── train.json          # 240 synthetic users
├── new-users.json      # 10 demo users (matched spicy/non-spicy pairs)
└── model-6.json        # Trained weights (committed for reproducibility)
```

## Reproducibility

Training is **fully deterministic**: weight initialization, epoch shuffling, and CD-1 Bernoulli sampling are all driven by explicit seeds (a seeded Mulberry32 PRNG plus per-call TensorFlow seeds). Two runs with the same arguments produce byte-identical weights.

## Honest caveats

- Cuisine disentanglement is 100% on the 10 demo users; over the full 240-user training set the purity is ~0.82. RBMs do not guarantee one-to-one interpretable units — the hidden bias decay is what made it work here.
- Bias decay smooths the probabilities, so the spicy unit separates with less margin than without it (0.47 / 0.72 instead of 0.01 / 0.99) — but still classifies 10/10.
- The data is synthetic and designed so the experiment is clean. This is a pedagogical demonstration, not a validation on real data.

## References

- G. Hinton — [*A Practical Guide to Training Restricted Boltzmann Machines*](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) (2010). Source of the CD-1 recipe, the "probabilities for reconstruction" trick (§3), and the sparsity penalty (§10).
- X. Glorot & Y. Bengio — [*Understanding the difficulty of training deep feedforward neural networks*](https://proceedings.mlr.press/v9/glorot10a.html) (2010). Weight initialization.
- The companion article: [Segmentación de preferencias de usuarios mediante Restricted Boltzmann Machines](https://arrowfunction.blog/articulos-y-noticias/segmentaci%C3%B3n-de-preferencias-de-usuarios-mediante-restricted-boltzmann-machines-b5946e) (Spanish).
