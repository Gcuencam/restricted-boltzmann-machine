import * as tf from "@tensorflow/tfjs-node";
import type { RBM } from "./rbm.js";

export interface TrainConfig {
  epochs: number;
  learningRate: number;
  /**
   * Mini-batch size for stochastic gradient descent.
   *
   * - undefined (default): usa el dataset completo cada epoch (batch GD).
   *   Gradientes estables pero la RBM puede quedarse atascada en mínimos locales.
   *
   * - número (ej. 32): divide el dataset en mini-batches aleatorios cada epoch.
   *   Introduce ruido en el gradiente que ayuda a escapar mínimos locales y
   *   suele acelerar la convergencia. A cambio, el entrenamiento es más irregular.
   *
   * Valores típicos: 16–64. Con datasets pequeños (<500 usuarios) probar con 16 o 32.
   */
  batchSize?: number;
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

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    // Mezclamos los índices cada epoch para que los mini-batches vean
    // combinaciones distintas y el modelo no memorice el orden de los datos.
    const indices = Array.from({ length: nSamples }, (_, i) => i);
    for (let i = nSamples - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j]!, indices[i]!];
    }

    // Una pasada por todos los mini-batches del epoch
    for (let start = 0; start < nSamples; start += effectiveBatchSize) {
      const batchData = indices
        .slice(start, start + effectiveBatchSize)
        .map(i => data[i]!);

      tf.tidy(() => {
        const vBatch = tf.tensor2d(batchData) as tf.Tensor2D;
        const { dW, dbv, dbh } = rbm.cd1(vBatch);
        rbm.W.assign(rbm.W.add(dW.mul(config.learningRate)));
        rbm.bv.assign(rbm.bv.add(dbv.mul(config.learningRate)));
        rbm.bh.assign(rbm.bh.add(dbh.mul(config.learningRate)));
      });
    }

    // El error se calcula siempre sobre el dataset COMPLETO para que la métrica
    // sea comparable entre epochs independientemente del tamaño de batch.
    const error = tf.tidy(() => {
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
