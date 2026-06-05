import * as tf from "@tensorflow/tfjs-node";
import type { RBM } from "./rbm.js";

export interface TrainConfig {
  epochs: number;
  learningRate: number;
}

/**
 * Train an RBM on binary data using CD-1.
 * @param data  [nUsers, nVisible] — full dataset as nested array
 * @returns     reconstruction error per epoch
 */
export function train(rbm: RBM, data: number[][], config: TrainConfig): number[] {
  const errors: number[] = [];
  const vAll = tf.tensor2d(data) as tf.Tensor2D;

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    const error = tf.tidy(() => {
      const { dW, dbv, dbh } = rbm.cd1(vAll);

      rbm.W.assign(rbm.W.add(dW.mul(config.learningRate)));
      rbm.bv.assign(rbm.bv.add(dbv.mul(config.learningRate)));
      rbm.bh.assign(rbm.bh.add(dbh.mul(config.learningRate)));

      // Error de reconstrucción: MSE entre v⁰ y P(v|h⁰)
      const ph0 = rbm.probHgivenV(vAll);
      const pv1 = rbm.probVgivenH(rbm.sample(ph0));
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
