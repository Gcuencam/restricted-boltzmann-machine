import { RBM } from "./model/rbm.js";
import { train } from "./model/train.js";
import { loadTrainDataset } from "./data/loader.js";
import { N_DISHES } from "./data/dataset.js";

const dataset = loadTrainDataset();
const rbm = new RBM({ nVisible: N_DISHES, nHidden: 3, seed: 42 });

console.log(`Entrenando con ${dataset.data.length} usuarios, ${N_DISHES} platos, 3 unidades ocultas\n`);

train(rbm, dataset.data, { epochs: 500, learningRate: 0.1 });

rbm.dispose();
