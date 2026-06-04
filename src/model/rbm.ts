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

  constructor(config: RBMConfig) {
    this.nVisible = config.nVisible;
    this.nHidden = config.nHidden;

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

  /** Learnable parameters — used by the optimizer */
  get params(): tf.Variable[] {
    return [this.W, this.bv, this.bh];
  }

  dispose(): void {
    this.W.dispose();
    this.bv.dispose();
    this.bh.dispose();
  }
}
