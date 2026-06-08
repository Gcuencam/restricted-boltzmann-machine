import * as tf from "@tensorflow/tfjs-node";
import type { RBM } from "./rbm.js";
import { mulberry32 } from "../data/dataset.js";

export interface TrainConfig {
  epochs: number;
  learningRate: number;
  /**
   * Seed for mini-batch shuffling. Fixing it makes training reproducible epoch
   * to epoch (important for a publishable experiment).
   */
  seed?: number;
  /**
   * Mini-batch size for stochastic gradient descent.
   *
   * - undefined (default): uses the full dataset each epoch (batch GD).
   *   Stable gradients but the RBM may get stuck in local minima.
   *
   * - number (e.g. 32): splits the dataset into random mini-batches each epoch.
   *   Introduces gradient noise that helps escape local minima and usually speeds
   *   up convergence. In return, training is more irregular.
   *
   * Typical values: 16–64. With small datasets (<500 users) try 16 or 32.
   */
  batchSize?: number;
  /**
   * L2 weight decay: penalises large weights (W += lr·(ΔW − weightDecay·W)).
   * Keeps filters small and clean and prevents a few connections from dominating.
   * Typical values: 1e-4 – 1e-3.
   */
  weightDecay?: number;
  /**
   * Weight decay applied to the HIDDEN BIASES bh. Unlike normal weight decay
   * (which only touches W), this slows the growth of biases that create "always-on"
   * units (high bias → saturation). Useful so that each unit reflects its weight
   * evidence rather than an arbitrary baseline.
   */
  hiddenBiasDecay?: number;
  /**
   * Sparsity penalty. Pushes the mean activation of each hidden unit towards
   * `target` with strength `cost`. By forcing each unit to fire for FEW users,
   * there are no "always-on" units (high bias) and each one tends to specialise
   * in a factor — here, a cuisine. Typical values: target 0.05–0.2, cost 0.1–1.0.
   * (Hinton 2010, §10.)
   */
  sparsity?: { target: number; cost: number };
}

/**
 * Train an RBM on binary data using CD-1.
 * @param data  [nUsers, nVisible] — full dataset as nested array
 * @returns     reconstruction error per epoch (computed on full dataset)
 */
export function train(rbm: RBM, data: number[][], config: TrainConfig): number[] {
  const errors: number[] = [];
  const nSamples = data.length;
  const effectiveBatchSize = config.batchSize ?? nSamples;
  const vAll = tf.tensor2d(data) as tf.Tensor2D;

  // Seeded RNG so the shuffle is identical on every run.
  const rng = config.seed !== undefined ? mulberry32(config.seed) : Math.random;

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    // Shuffle indices every epoch so mini-batches see different combinations
    // and the model does not memorise the data order.
    const indices = Array.from({ length: nSamples }, (_, i) => i);
    for (let i = nSamples - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [indices[i], indices[j]] = [indices[j]!, indices[i]!];
    }

    // One pass over all mini-batches in the epoch
    for (let start = 0; start < nSamples; start += effectiveBatchSize) {
      const batchData = indices
        .slice(start, start + effectiveBatchSize)
        .map(i => data[i]!);

      tf.tidy(() => {
        const vBatch = tf.tensor2d(batchData) as tf.Tensor2D;
        const { dW, dbv, dbh } = rbm.cd1(vBatch);

        // L2 weight decay: subtract a fraction of W itself from the gradient.
        const dWreg = config.weightDecay
          ? dW.sub(rbm.W.mul(config.weightDecay))
          : dW;

        // Sparsity: push the mean activation of each unit towards the target.
        // q = per-unit mean of P(h=1 | vBatch); the gradient (target − q)
        // lowers the bias of units that fire too often.
        let dbhReg = dbh;
        if (config.sparsity) {
          const q = rbm.probHgivenV(vBatch).mean(0) as tf.Tensor1D;
          const grad = tf.scalar(config.sparsity.target).sub(q).mul(config.sparsity.cost);
          dbhReg = dbhReg.add(grad) as tf.Tensor1D;
        }
        if (config.hiddenBiasDecay) {
          dbhReg = dbhReg.sub(rbm.bh.mul(config.hiddenBiasDecay)) as tf.Tensor1D;
        }

        rbm.W.assign(rbm.W.add(dWreg.mul(config.learningRate)));
        rbm.bv.assign(rbm.bv.add(dbv.mul(config.learningRate)));
        rbm.bh.assign(rbm.bh.add(dbhReg.mul(config.learningRate)));
      });
    }

    // Error is always computed over the FULL dataset so the metric is comparable
    // across epochs regardless of batch size. We use probabilities (mean-field,
    // no sampling) so the curve is smooth and reproducible rather than noisy.
    const error = tf.tidy(() => {
      const ph0 = rbm.probHgivenV(vAll);
      const pv1 = rbm.probVgivenH(ph0);
      return vAll.sub(pv1).square().mean().arraySync() as number;
    });

    errors.push(error);

    if (epoch % 10 === 0 || epoch === config.epochs - 1) {
      console.log(`Epoch ${String(epoch).padStart(3)}: reconstruction error = ${error.toFixed(4)}`);
    }
  }

  vAll.dispose();
  return errors;
}
