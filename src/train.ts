import { RBM } from "./model/rbm.js";
import { train } from "./model/train.js";
import { saveModel } from "./model/persistence.js";
import { loadTrainDataset } from "./data/loader.js";
import { N_DISHES } from "./data/dataset.js";

const nHidden        = parseInt(process.argv[2] ?? "6",    10);
const epochs         = parseInt(process.argv[3] ?? "500",   10);
const learningRate   = parseFloat(process.argv[4] ?? "0.1");
const batchSize      = process.argv[5] ? parseInt(process.argv[5], 10) : 32;
const seed           = process.argv[6] ? parseInt(process.argv[6], 10) : 42;
// ─────────────────────────────────────────────────────────────────────────────
// DECAIMIENTO DE LOS BIAS OCULTOS — la pieza que desenreda las cocinas.
//
// Síntoma: sin esto, durante el entrenamiento algunas unidades ocultas
// desarrollan un bias muy grande (p.ej. bh ≈ +5.4). Esa unidad queda "encendida
// por defecto": σ(5.4) ≈ 0.996 para CASI cualquier usuario. Su activación deja de
// reflejar evidencia (qué platos pidió el comensal) y pasa a reflejar un baseline
// arbitrario. Resultado: varias unidades saturan a ~1 a la vez y el argmax que
// asigna "factor dominante" se vuelve una lotería → las cocinas no se separan.
//
// Arreglo: restar una fracción del propio bh en cada paso (W -= ... no basta;
// el problema está en bh, no en W). Así ninguna unidad puede crecer hasta
// saturar y su activación vuelve a estar gobernada por los pesos = la evidencia.
// Con hbDecay≈0.02, max|bh| baja de ~5.8 a ~1.5 y cada cocina cae en su propia
// unidad limpia (bijección cocina↔unidad). Ver model/train.ts para la fórmula.
//
// Efecto secundario: al no saturar, las probabilidades son más suaves, así que la
// unidad del picante separa con menos margen (0.47 vs 0.72 en vez de 0.01/0.99),
// pero sigue clasificando correctamente. Es un buen intercambio.
// ─────────────────────────────────────────────────────────────────────────────
const hiddenBiasDecay = process.argv[7] ? parseFloat(process.argv[7]) : 0.02;

const dataset = loadTrainDataset();
const rbm = new RBM({ nVisible: N_DISHES, nHidden, seed });

const batchLabel = batchSize ? `batch=${batchSize}` : "batch=full";
console.log(`Entrenando — hidden: ${nHidden}  epochs: ${epochs}  lr: ${learningRate}  ${batchLabel}  seed: ${seed}  hbDecay: ${hiddenBiasDecay}\n`);
train(rbm, dataset.data, { epochs, learningRate, batchSize, seed, hiddenBiasDecay });

await saveModel(rbm, `data/model-${nHidden}.json`);
console.log(`\n✓ Modelo guardado en data/model-${nHidden}.json`);

rbm.dispose();
