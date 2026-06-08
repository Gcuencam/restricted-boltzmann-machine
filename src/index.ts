import * as tf from "@tensorflow/tfjs-node";
import chalk from "chalk";
import { loadModel } from "./model/persistence.js";
import { loadNewUsers, loadTrainDataset } from "./data/loader.js";
import { DISHES, N_DISHES, DISH_SPICY, CUISINES } from "./data/dataset.js";

/** Color by activation intensity: green ≥0.9, yellow ≥0.75, orange ≥0.5. */
const colorByActivation = (a: number, text: string): string =>
  a >= 0.9 ? chalk.green(text)
  : a >= 0.75 ? chalk.yellow(text)
  : chalk.hex("#FFA500")(text); // orange (0.5 ≤ a < 0.75)

const nHidden = parseInt(process.argv[2] ?? "6", 10);
const rbm = loadModel(`data/model-${nHidden}.json`);
const users = loadNewUsers();

// ── 1. Weight matrix W ───────────────────────────────────────────────────────

const W = await rbm.W.array() as number[][];

const fmt = (n: number) => (n >= 0 ? " " : "") + n.toFixed(8);
const COL = 14;
const hiddenHeaders = Array.from({ length: rbm.nHidden }, (_, h) => `Hidden ${h + 1}`.padStart(COL)).join("  ");

console.log("\nThe network learned the following weights:\n");
console.log(" ".repeat(28) + hiddenHeaders + "   picante?");
for (let d = 0; d < N_DISHES; d++) {
  // Color strong associations (weight ≥ 1) in green. Padding is applied
  // BEFORE the color so ANSI codes do not misalign the columns.
  const row = W[d]!.map(v => {
    const cell = fmt(v).padStart(COL);
    return v >= 1 ? chalk.green(cell) : cell;
  }).join("  ");
  console.log(DISHES[d]!.padEnd(28) + row + (DISH_SPICY[d] ? "      🌶️" : ""));
}

// ── 1b. Which hidden unit encodes "spicy"? ────────────────────────────────────
// For each unit we measure how much more it weights spicy dishes than non-spicy
// ones. The one that maximises that difference is the spicy factor candidate.

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

// ── 1c. Unit → cuisine map (discovered on the train set) ──────────────────────
// Spicy is ONE axis; cuisine is ANOTHER orthogonal axis. For the "Factor" column
// to mean the diner's cuisine, we look for their dominant unit among the NON-spicy
// ones. The label of each unit (which cuisine it represents) is discovered by
// majority vote: which unit receives the most users from each cuisine.

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


// ── 2. User preferences (dishes × users) ─────────────────────────────────────

const NAME_COL = 8;
const nameHeaders = users.map(u => u.name.padStart(NAME_COL)).join(" ");
console.log("\n\nPreferencias de usuarios nuevos (1 = quiere pedir):\n");
console.log(" ".repeat(30) + nameHeaders);
for (let d = 0; d < N_DISHES; d++) {
  const row = users.map(u => String(u.preferences[d] ?? 0).padStart(NAME_COL)).join(" ");
  console.log(DISHES[d]!.padEnd(30) + row + (DISH_SPICY[d] ? "  🌶️" : ""));
}

// ── 3. Hidden activation ──────────────────────────────────────────────────────

const ACT_COL = 10;
const hHeaders = Array.from({ length: nHidden }, (_, h) =>
  (h === spicyUnit ? `*Hidden ${h + 1}` : `Hidden ${h + 1}`).padStart(ACT_COL)
).join("  ");

console.log("\n\nActivación oculta de usuarios nuevos  P(h=1 | usuario):");
console.log(`(* = unidad del picante, Hidden ${spicyUnit + 1})\n`);
console.log(`  ${"Usuario".padEnd(10)}  ${"Cocina".padEnd(10)}  ${"picante".padEnd(8)}  ${hHeaders}   → Factores más activos`);

const vUsers = tf.tensor2d(users.map(u => u.preferences)) as tf.Tensor2D;
const pH = await rbm.probHgivenV(vUsers).array() as number[][];

// Last column: ALL active units (P > 0.5) for each diner, sorted from highest to
// lowest and colored by intensity (green ≥0.9, yellow ≥0.75, orange ≥0.5).
// Reveals that a diner expresses several factors at once — e.g. spicy diners
// activate their cuisine unit AND the spicy unit (H6).
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
