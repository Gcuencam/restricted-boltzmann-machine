import * as tf from "@tensorflow/tfjs-node";

export interface RBMConfig {
  nVisible: number;
  nHidden: number;
  seed?: number;
}

/**
 * Restricted Boltzmann Machine with binary visible and hidden units.
 *
 * Energy:  E(v,h) = -bv·v - bh·h - v·W·h
 * Params:  W [nVisible, nHidden], bv [nVisible], bh [nHidden]
 */
export class RBM {
  readonly nVisible: number;
  readonly nHidden: number;

  W: tf.Variable<tf.Rank.R2>;
  bv: tf.Variable<tf.Rank.R1>;
  bh: tf.Variable<tf.Rank.R1>;

  // Incremental seed for Bernoulli sampling. Makes training fully deterministic
  // (same weights on every run) instead of depending on TensorFlow's global PRNG.
  private sampleSeed: number;
  private sampleCount = 0;

  constructor(config: RBMConfig) {
    this.nVisible = config.nVisible;
    this.nHidden = config.nHidden;
    this.sampleSeed = config.seed ?? Math.floor(Math.random() * 1e9);

    // Xavier/Glorot uniform init scaled by 0.1 — keeps gradients well-conditioned at the start of training (Glorot & Bengio, 2010).
    const limit = 0.1 * Math.sqrt(6 / (config.nVisible + config.nHidden));
    const W = tf.randomUniform(
      [config.nVisible, config.nHidden],
      -limit,
      limit,
      "float32",
      config.seed
    );

    this.W = tf.variable(W, true, "W") as tf.Variable<tf.Rank.R2>;
    this.bv = tf.variable(tf.zeros([config.nVisible]), true, "bv") as tf.Variable<tf.Rank.R1>;
    this.bh = tf.variable(tf.zeros([config.nHidden]), true, "bh") as tf.Variable<tf.Rank.R1>;

    W.dispose();
  }

  /**
   * P(h=1 | v) = σ(v W + bh)
   * @param v  [batch, nVisible] — binary visible states
   * @returns  [batch, nHidden]  — hidden activation probabilities
   */
  probHgivenV(v: tf.Tensor2D): tf.Tensor2D {
    return tf.sigmoid(v.matMul(this.W).add(this.bh)) as tf.Tensor2D;
  }

  /**
   * P(v=1 | h) = σ(h Wᵀ + bv)
   * @param h  [batch, nHidden]  — binary hidden states
   * @returns  [batch, nVisible] — visible activation probabilities
   */
  probVgivenH(h: tf.Tensor2D): tf.Tensor2D {
    return tf.sigmoid(h.matMul(this.W.transpose()).add(this.bv)) as tf.Tensor2D;
  }

  /**
   * Bernoulli sample from a probability matrix.
   * @param probs  [batch, n] — values in (0, 1)
   * @returns      [batch, n] — binary 0/1 tensor, same shape
   */
  sample(probs: tf.Tensor2D): tf.Tensor2D {
    const seed = this.sampleSeed + this.sampleCount++;
    return tf.randomUniform(probs.shape as [number, number], 0, 1, "float32", seed)
      .less(probs).cast('float32') as tf.Tensor2D;
  }

  /** Learnable parameters — used by the optimizer */
  get params(): tf.Variable[] {
    return [this.W, this.bv, this.bh];
  }

  /**
   * Contrastive Divergence 1 — computes parameter gradients for one batch.
   *
   * Chain: v⁰ → ph0 → h0 → pv1 → ph1
   * ΔW   = (v⁰ᵀ ph0  −  pv1ᵀ ph1) / batch
   * Δbv  = mean(v⁰ − pv1,  axis=0)
   * Δbh  = mean(ph0 − ph1, axis=0)
   *
   * Only the hidden layer is sampled (h0): it acts as a binary bottleneck that
   * forces the network to compress. Visible reconstruction uses probabilities
   * (pv1) instead of sampling, which reduces gradient noise and produces sharper
   * filters — recommended practice from Hinton (A Practical Guide to Training
   * RBMs, 2010, §3).
   */
  cd1(v0: tf.Tensor2D): { dW: tf.Tensor2D; dbv: tf.Tensor1D; dbh: tf.Tensor1D } {
    return tf.tidy(() => {
      const batchSize = v0.shape[0];

      const ph0 = this.probHgivenV(v0);
      const h0  = this.sample(ph0);
      const pv1 = this.probVgivenH(h0);
      const ph1 = this.probHgivenV(pv1);

      // (v0ᵀ @ ph0 − pv1ᵀ @ ph1) / batch  →  [nVisible, nHidden]
      const dW  = v0.transpose().matMul(ph0)
                    .sub(pv1.transpose().matMul(ph1))
                    .div(batchSize) as tf.Tensor2D;

      // mean(v0 − pv1, axis=0)  →  [nVisible]
      const dbv = v0.sub(pv1).mean(0) as tf.Tensor1D;

      // mean(ph0 − ph1, axis=0)  →  [nHidden]
      const dbh = ph0.sub(ph1).mean(0) as tf.Tensor1D;

      return { dW, dbv, dbh };
    }) as { dW: tf.Tensor2D; dbv: tf.Tensor1D; dbh: tf.Tensor1D };
  }

  dispose(): void {
    this.W.dispose();
    this.bv.dispose();
    this.bh.dispose();
  }
}
