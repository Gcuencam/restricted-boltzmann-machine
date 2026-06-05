import * as tf from "@tensorflow/tfjs-node";
import { loadModel } from "./model/persistence.js";
import { loadNewUsers } from "./data/loader.js";
import { DISHES, N_DISHES } from "./data/dataset.js";

const nHidden = parseInt(process.argv[2] ?? "3", 10);
/** "argmax": mayor activación absoluta. "deviation": mayor desviación respecto a la media poblacional. */
const dominantMethod = (process.argv[3] ?? "argmax") === "deviation" ? "deviation" : "argmax";
const rbm = loadModel(`data/model-${nHidden}.json`);
const users = loadNewUsers();

// ── 1. Matriz de pesos W ──────────────────────────────────────────────────────

const W = await rbm.W.array() as number[][];

const fmt = (n: number) => (n >= 0 ? " " : "") + n.toFixed(8);
const COL = 14;
const hiddenHeaders = Array.from({ length: rbm.nHidden }, (_, h) => `Hidden ${h + 1}`.padStart(COL)).join("  ");

console.log("\nThe network learned the following weights:\n");
console.log(" ".repeat(28) + hiddenHeaders);
for (let d = 0; d < N_DISHES; d++) {
  const row = W[d]!.map(v => fmt(v).padStart(COL)).join("  ");
  console.log(DISHES[d]!.padEnd(28) + row);
}

// ── 2. Preferencias de usuarios (platos × usuarios) ───────────────────────────

const NAME_COL = 8;
const nameHeaders = users.map(u => u.name.padStart(NAME_COL)).join(" ");
console.log("\n\nPreferencias de usuarios nuevos (1 = quiere pedir):\n");
console.log(" ".repeat(30) + nameHeaders);
for (let d = 0; d < N_DISHES; d++) {
  const row = users.map(u => String(u.preferences[d] ?? 0).padStart(NAME_COL)).join(" ");
  console.log(DISHES[d]!.padEnd(30) + row);
}

// ── 3. Activación oculta ──────────────────────────────────────────────────────

const ACT_COL = 10;
const hHeaders = Array.from({ length: nHidden }, (_, h) => `Hidden ${h + 1}`.padStart(ACT_COL)).join("  ");

console.log("\n\nActivación oculta de usuarios nuevos  P(h=1 | usuario):\n");
console.log(`  ${"Usuario".padEnd(12)}  ${"Arquetipo".padEnd(12)}  ${hHeaders}   → Factor`);

const vUsers = tf.tensor2d(users.map(u => u.preferences)) as tf.Tensor2D;
const pH = await rbm.probHgivenV(vUsers).array() as number[][];

// Medias por unidad — necesarias para el método "deviation".
// "deviation": el factor dominante es el que más se desvía de su media poblacional.
//   Útil cuando una unidad activa alto para casi todos (ej. "anti-cuchara"):
//   el valor absoluto alto no es informativo, pero la desviación sí lo es.
//   Es el mismo principio que un z-score.
// "argmax" (por defecto): simplemente el índice de mayor activación absoluta.
const means = Array.from({ length: nHidden }, (_, h) =>
  pH.reduce((sum, acts) => sum + acts[h]!, 0) / pH.length
);

for (let i = 0; i < users.length; i++) {
  const user = users[i]!;
  const acts = pH[i]!;
  const scores = dominantMethod === "deviation"
    ? acts.map((a, h) => a - means[h]!)
    : acts;
  const dominant = scores.indexOf(Math.max(...scores));
  const vals = acts.map(a => a.toFixed(3).padStart(ACT_COL)).join("  ");
  console.log(`  ${user.name.padEnd(12)}  ${user.archetype.padEnd(12)}  ${vals}   → Hidden ${dominant + 1}`);
}

tf.dispose([vUsers]);
rbm.dispose();
