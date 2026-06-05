import * as tf from "@tensorflow/tfjs-node";
import type { RBM } from "./rbm.js";
import { mulberry32 } from "../data/dataset.js";

export interface TrainConfig {
  epochs: number;
  learningRate: number;
  /**
   * Semilla para el barajado de los mini-batches. Fijarla hace el entrenamiento
   * reproducible epoch a epoch (importante para un experimento publicable).
   */
  seed?: number;
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
  /**
   * Weight decay L2: penaliza pesos grandes (W += lr·(ΔW − weightDecay·W)).
   * Mantiene los filtros pequeños y limpios y evita que unas pocas conexiones
   * dominen. Valores típicos: 1e-4 – 1e-3.
   */
  weightDecay?: number;
  /**
   * Weight decay aplicado a los BIAS ocultos bh. A diferencia del weight decay
   * normal (que solo toca W), este frena el crecimiento de los bias que crean
   * unidades "encendidas por defecto" (bias alto → saturación). Útil para que
   * cada unidad refleje su evidencia de pesos y no un baseline arbitrario.
   */
  hiddenBiasDecay?: number;
  /**
   * Penalización de dispersión (sparsity). Empuja la activación media de cada
   * unidad oculta hacia `target` con fuerza `cost`. Al forzar que cada unidad se
   * encienda para POCOS usuarios, deja de haber unidades "encendidas por defecto"
   * (bias alto) y cada una tiende a especializarse en un factor — aquí, una cocina.
   * Valores típicos: target 0.05–0.2, cost 0.1–1.0. (Hinton 2010, §10.)
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

  // RNG sembrado para que el barajado sea idéntico en cada corrida.
  const rng = config.seed !== undefined ? mulberry32(config.seed) : Math.random;

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    // Mezclamos los índices cada epoch para que los mini-batches vean
    // combinaciones distintas y el modelo no memorice el orden de los datos.
    const indices = Array.from({ length: nSamples }, (_, i) => i);
    for (let i = nSamples - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
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

        // Weight decay L2: resta una fracción del propio W al gradiente.
        const dWreg = config.weightDecay
          ? dW.sub(rbm.W.mul(config.weightDecay))
          : dW;

        // Sparsity: empuja la activación media de cada unidad hacia el target.
        // q = media por unidad de P(h=1 | vBatch); el gradiente (target − q)
        // baja el bias de las unidades que se activan de más.
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

    // El error se calcula siempre sobre el dataset COMPLETO para que la métrica
    // sea comparable entre epochs independientemente del tamaño de batch.
    // Usamos las probabilidades (mean-field, sin muestreo) para que la curva sea
    // suave y reproducible en lugar de ruidosa.
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
