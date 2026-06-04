import * as tf from "@tensorflow/tfjs-node";
import { RBM } from "./model/rbm.js";
import { N_DISHES } from "./data/dataset.js";

const rbm = new RBM({ nVisible: N_DISHES, nHidden: 3, seed: 42 });

// Batch de 2 usuarios ficticios: [batch=2, nVisible=12]
const v = tf.tensor2d([
  [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0], // perfil mexicana
  [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], // perfil pescado
]);

const pH = rbm.probHgivenV(v);
console.log("P(h|v) shape:", pH.shape);       // [2, 3]
console.log("P(h|v) values:");
pH.print();

const hSample = rbm.sample(pH);
console.log("h sample:");
hSample.print();

const pV = rbm.probVgivenH(hSample);
console.log("P(v|h) shape:", pV.shape);       // [2, 12]
console.log("P(v|h) values (reconstrucción):");
pV.print();

tf.dispose([v, pH, hSample, pV]);
rbm.dispose();
