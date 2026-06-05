import { RBM } from "./model/rbm.js";
import { train } from "./model/train.js";
import { saveModel } from "./model/persistence.js";
import { loadTrainDataset } from "./data/loader.js";
import { N_DISHES } from "./data/dataset.js";

const nHidden      = parseInt(process.argv[2] ?? "3",    10);
const epochs       = parseInt(process.argv[3] ?? "500",   10);
const learningRate = parseFloat(process.argv[4] ?? "0.1");
const batchSize    = process.argv[5] ? parseInt(process.argv[5], 10) : undefined;

const dataset = loadTrainDataset();
const rbm = new RBM({ nVisible: N_DISHES, nHidden, seed: 42 });

const batchLabel = batchSize ? `batch=${batchSize}` : "batch=full";
console.log(`Entrenando — hidden: ${nHidden}  epochs: ${epochs}  lr: ${learningRate}  ${batchLabel}\n`);
train(rbm, dataset.data, { epochs, learningRate, batchSize });

await saveModel(rbm, `data/model-${nHidden}.json`);
console.log(`\n✓ Modelo guardado en data/model-${nHidden}.json`);

rbm.dispose();
