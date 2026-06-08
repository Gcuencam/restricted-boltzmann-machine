import { RBM } from "./model/rbm.js";
import { train } from "./model/train.js";
import { saveModel } from "./model/persistence.js";
import { loadTrainDataset } from "./data/loader.js";
import { N_DISHES } from "./data/dataset.js";

const nHidden        = parseInt(process.argv[2] ?? "6",    10);
const epochs         = parseInt(process.argv[3] ?? "500",   10);
const learningRate   = parseFloat(process.argv[4] ?? "0.1");
const batchSize      = process.argv[5] ? parseInt(process.argv[5], 10) : 32;
const seed           = process.argv[6] ? parseInt(process.argv[6], 10) : 42;
// ─────────────────────────────────────────────────────────────────────────────
// HIDDEN BIAS DECAY — the piece that disentangles the cuisines.
//
// Symptom: without this, during training some hidden units develop a very large
// bias (e.g. bh ≈ +5.4). That unit stays "on by default": σ(5.4) ≈ 0.996 for
// ALMOST any user. Its activation no longer reflects evidence (which dishes the
// diner ordered) and instead reflects an arbitrary baseline. Result: several
// units saturate to ~1 at the same time and the argmax that assigns the
// "dominant factor" becomes a lottery → cuisines do not separate.
//
// Fix: subtract a fraction of bh itself at each step (W -= ... is not enough;
// the problem is in bh, not W). This way no unit can grow until saturation and
// its activation is again governed by weights = the evidence. With hbDecay≈0.02,
// max|bh| drops from ~5.8 to ~1.5 and each cuisine falls into its own clean unit
// (bijection cuisine↔unit). See model/train.ts for the formula.
//
// Side effect: by not saturating, probabilities are smoother, so the spicy unit
// separates with less margin (0.47 vs 0.72 instead of 0.01/0.99), but still
// classifies correctly. It is a good trade-off.
// ─────────────────────────────────────────────────────────────────────────────
const hiddenBiasDecay = process.argv[7] ? parseFloat(process.argv[7]) : 0.02;

const dataset = loadTrainDataset();
const rbm = new RBM({ nVisible: N_DISHES, nHidden, seed });

const batchLabel = batchSize ? `batch=${batchSize}` : "batch=full";
console.log(`Entrenando — hidden: ${nHidden}  epochs: ${epochs}  lr: ${learningRate}  ${batchLabel}  seed: ${seed}  hbDecay: ${hiddenBiasDecay}\n`);
train(rbm, dataset.data, { epochs, learningRate, batchSize, seed, hiddenBiasDecay });

await saveModel(rbm, `data/model-${nHidden}.json`);
console.log(`\n✓ Modelo guardado en data/model-${nHidden}.json`);

rbm.dispose();
