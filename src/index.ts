import { RBM } from "./model/rbm.js";
import { N_DISHES } from "./data/dataset.js";

const rbm = new RBM({ nVisible: N_DISHES, nHidden: 3, seed: 42 });

console.log("W shape:", rbm.W.shape);   // [12, 3]
console.log("bv shape:", rbm.bv.shape); // [12]
console.log("bh shape:", rbm.bh.shape); // [3]

const wVals = await rbm.W.data();
const min = Math.min(...wVals).toFixed(4);
const max = Math.max(...wVals).toFixed(4);
const limit = (0.1 * Math.sqrt(6 / (N_DISHES + 3))).toFixed(4);
console.log(`W range: [${min}, ${max}]  (expected ≈ ±${limit})`);

rbm.dispose();
