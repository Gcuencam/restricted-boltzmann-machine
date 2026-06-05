import * as tf from "@tensorflow/tfjs-node";
import chalk from "chalk";
import { loadModel } from "./model/persistence.js";
import { loadNewUsers, loadTrainDataset } from "./data/loader.js";
import { DISHES, N_DISHES, DISH_SPICY, CUISINES } from "./data/dataset.js";

/** Color por intensidad de activación: verde ≥0.9, amarillo ≥0.75, naranja ≥0.5. */
const colorByActivation = (a: number, text: string): string =>
  a >= 0.9 ? chalk.green(text)
  : a >= 0.75 ? chalk.yellow(text)
  : chalk.hex("#FFA500")(text); // naranja (0.5 ≤ a < 0.75)

const nHidden = parseInt(process.argv[2] ?? "6", 10);
const rbm = loadModel(`data/model-${nHidden}.json`);
const users = loadNewUsers();

// ── 1. Matriz de pesos W ──────────────────────────────────────────────────────

const W = await rbm.W.array() as number[][];

const fmt = (n: number) => (n >= 0 ? " " : "") + n.toFixed(8);
const COL = 14;
const hiddenHeaders = Array.from({ length: rbm.nHidden }, (_, h) => `Hidden ${h + 1}`.padStart(COL)).join("  ");

console.log("\nThe network learned the following weights:\n");
console.log(" ".repeat(28) + hiddenHeaders + "   picante?");
for (let d = 0; d < N_DISHES; d++) {
  const row = W[d]!.map(v => fmt(v).padStart(COL)).join("  ");
  console.log(DISHES[d]!.padEnd(28) + row + (DISH_SPICY[d] ? "      🌶️" : ""));
}

// ── 1b. ¿Qué unidad oculta codifica el "picante"? ─────────────────────────────
// Para cada unidad medimos cuánto más pesa sobre los platos picantes que sobre
// los no picantes. La que maximiza esa diferencia es la candidata a factor picante.

const spicyScore = Array.from({ length: rbm.nHidden }, (_, h) => {
  let sumSpicy = 0, nSpicy = 0, sumMild = 0, nMild = 0;
  for (let d = 0; d < N_DISHES; d++) {
    if (DISH_SPICY[d]) { sumSpicy += W[d]![h]!; nSpicy++; }
    else { sumMild += W[d]![h]!; nMild++; }
  }
  return sumSpicy / nSpicy - sumMild / nMild;
});
const spicyUnit = spicyScore.indexOf(Math.max(...spicyScore));
console.log(
  `\n→ Unidad latente del PICANTE: Hidden ${spicyUnit + 1} ` +
  `(peso picante − no picante = ${spicyScore[spicyUnit]!.toFixed(3)})`
);

// ── 1c. Mapa unidad → cocina (descubierto sobre el train set) ──────────────────
// El picante es UN eje; la cocina es OTRO eje ortogonal. Para que la columna
// "Factor" signifique la cocina del comensal, buscamos su unidad dominante entre
// las NO picantes. La etiqueta de cada unidad (qué cocina representa) se descubre
// por mayoría: a qué unidad envía cada cocina la mayor parte de sus usuarios.

const cuisineUnits = Array.from({ length: nHidden }, (_, h) => h).filter(h => h !== spicyUnit);

const train = loadTrainDataset();
const pHtrain = tf.tidy(() =>
  rbm.probHgivenV(tf.tensor2d(train.data) as tf.Tensor2D).arraySync() as number[][]
);

/** Unidad de cocina dominante: la unidad NO picante más activa. */
const dominantCuisineUnit = (acts: number[]): number =>
  cuisineUnits.reduce((a, b) => (acts[b]! > acts[a]! ? b : a), cuisineUnits[0]!);

const unitToCuisine = new Map<number, string>();
for (let c = 0; c < CUISINES.length; c++) {
  const counts = new Map<number, number>();
  train.cuisineLabels.forEach((lab, i) => {
    if (lab !== c) return;
    const u = dominantCuisineUnit(pHtrain[i]!);
    counts.set(u, (counts.get(u) ?? 0) + 1);
  });
  const major = [...counts.entries()].reduce((a, b) => (b[1] > a[1] ? b : a))[0];
  unitToCuisine.set(major, CUISINES[c]!);
}
console.log(
  "→ Mapa unidad → cocina (descubierto): " +
  [...unitToCuisine].sort((a, b) => a[0] - b[0]).map(([u, c]) => `Hidden ${u + 1}=${c}`).join("  ")
);


// ── 2. Preferencias de usuarios (platos × usuarios) ───────────────────────────

const NAME_COL = 8;
const nameHeaders = users.map(u => u.name.padStart(NAME_COL)).join(" ");
console.log("\n\nPreferencias de usuarios nuevos (1 = quiere pedir):\n");
console.log(" ".repeat(30) + nameHeaders);
for (let d = 0; d < N_DISHES; d++) {
  const row = users.map(u => String(u.preferences[d] ?? 0).padStart(NAME_COL)).join(" ");
  console.log(DISHES[d]!.padEnd(30) + row + (DISH_SPICY[d] ? "  🌶️" : ""));
}

// ── 3. Activación oculta ──────────────────────────────────────────────────────

const ACT_COL = 10;
const hHeaders = Array.from({ length: nHidden }, (_, h) =>
  (h === spicyUnit ? `*Hidden ${h + 1}` : `Hidden ${h + 1}`).padStart(ACT_COL)
).join("  ");

console.log("\n\nActivación oculta de usuarios nuevos  P(h=1 | usuario):");
console.log(`(* = unidad del picante, Hidden ${spicyUnit + 1})\n`);
console.log(`  ${"Usuario".padEnd(10)}  ${"Cocina".padEnd(10)}  ${"picante".padEnd(8)}  ${hHeaders}   → Factores más activos`);

const vUsers = tf.tensor2d(users.map(u => u.preferences)) as tf.Tensor2D;
const pH = await rbm.probHgivenV(vUsers).array() as number[][];

// Última columna: TODAS las unidades activas (P > 0.5) de cada comensal, ordenadas
// de mayor a menor y coloreadas por intensidad (verde ≥0.9, amarillo ≥0.75, naranja
// ≥0.5). Revela que un comensal expresa varios factores a la vez — p.ej. los
// picantes encienden su cocina Y la unidad del picante (H6).
for (let i = 0; i < users.length; i++) {
  const user = users[i]!;
  const acts = pH[i]!;
  const factores = acts
    .map((a, h) => ({ a, h }))
    .filter(({ a }) => a >= 0.5)
    .sort((x, y) => y.a - x.a)
    .map(({ a, h }) => colorByActivation(a, `H${h + 1}`))
    .join(" ");
  const vals = acts.map((a, h) =>
    (h === spicyUnit ? "*" + a.toFixed(3) : a.toFixed(3)).padStart(ACT_COL)
  ).join("  ");
  const spicyTag = user.spicy ? "Sí" : "no";
  console.log(`  ${user.name.padEnd(10)}  ${user.cuisine.padEnd(10)}  ${spicyTag.padEnd(8)}  ${vals}   → ${factores || "—"}`);
}

tf.dispose([vUsers]);
rbm.dispose();
