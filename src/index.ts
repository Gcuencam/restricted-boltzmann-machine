import * as tf from "@tensorflow/tfjs-node";
import { loadModel } from "./model/persistence.js";
import { loadNewUsers } from "./data/loader.js";
import { DISHES, N_DISHES, DISH_SPICY } from "./data/dataset.js";

// ── Helpers de ancho de visualización (terminal) ──────────────────────────────
// padStart/padEnd cuentan unidades de código UTF-16, no columnas del terminal.
// El emoji 🌶️ (U+1F336 + U+FE0F) mide .length 3 pero ocupa ~2 columnas.
const displayWidth = (s: string): number => {
  let w = 0;
  for (const ch of s) {
    const cp = ch.codePointAt(0)!;
    if (cp === 0xfe0f || (cp >= 0x0300 && cp <= 0x036f)) continue; // variación / combinantes
    if (cp >= 0x1f000) w += 2; // emoji (incl. 🌶️) ocupan 2 columnas
    else w += 1; // ✓/✗ (U+2713/2717), → (U+2192), acentos latinos, dígitos, *, etc.
  }
  return w;
};
const padEndV = (s: string, width: number): string => s + " ".repeat(Math.max(0, width - displayWidth(s)));
const padStartV = (s: string, width: number): string => " ".repeat(Math.max(0, width - displayWidth(s))) + s;

const nHidden = parseInt(process.argv[2] ?? "6", 10);
/** "argmax": mayor activación absoluta. "deviation": mayor desviación respecto a la media poblacional. */
const dominantMethod = (process.argv[3] ?? "argmax") === "deviation" ? "deviation" : "argmax";
const rbm = loadModel(`data/model-${nHidden}.json`);
const users = loadNewUsers();

// ── 1. Matriz de pesos W ──────────────────────────────────────────────────────

const W = await rbm.W.array() as number[][];

const fmt = (n: number) => (n >= 0 ? " " : "") + n.toFixed(8);
const COL = 14;
const hiddenHeaders = Array.from({ length: rbm.nHidden }, (_, h) => padStartV(`Hidden ${h + 1}`, COL)).join("  ");

console.log("\nThe network learned the following weights:\n");
console.log(" ".repeat(28) + hiddenHeaders + "   picante?");
for (let d = 0; d < N_DISHES; d++) {
  const row = W[d]!.map(v => padStartV(fmt(v), COL)).join("  ");
  console.log(padEndV(DISHES[d]!, 28) + row + (DISH_SPICY[d] ? "      🌶️" : ""));
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

// ── 2. Preferencias de usuarios (platos × usuarios) ───────────────────────────

const NAME_COL = 8;
const nameHeaders = users.map(u => padStartV(u.name, NAME_COL)).join(" ");
console.log("\n\nPreferencias de usuarios nuevos (1 = quiere pedir):\n");
console.log(" ".repeat(30) + nameHeaders);
for (let d = 0; d < N_DISHES; d++) {
  const row = users.map(u => padStartV(String(u.preferences[d] ?? 0), NAME_COL)).join(" ");
  console.log(padEndV(DISHES[d]!, 30) + row + (DISH_SPICY[d] ? "  🌶️" : ""));
}

// ── 3. Activación oculta ──────────────────────────────────────────────────────

const ACT_COL = 10;
const hHeaders = Array.from({ length: nHidden }, (_, h) =>
  padStartV(h === spicyUnit ? `*Hidden ${h + 1}` : `Hidden ${h + 1}`, ACT_COL)
).join("  ");

console.log("\n\nActivación oculta de usuarios nuevos  P(h=1 | usuario):");
console.log(`(* = unidad del picante)\n`);
console.log(`  ${padEndV("Usuario", 10)}  ${padEndV("Cocina", 10)}  ${padEndV("picante", 8)}  ${hHeaders}   → Factor`);

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
  const vals = acts.map((a, h) =>
    padStartV(h === spicyUnit ? "*" + a.toFixed(3) : a.toFixed(3), ACT_COL)
  ).join("  ");
  const spicyTag = user.spicy ? "Sí 🌶️" : "no";
  console.log(`  ${padEndV(user.name, 10)}  ${padEndV(user.cuisine, 10)}  ${padEndV(spicyTag, 8)}  ${vals}   → Hidden ${dominant + 1}`);
}

tf.dispose([vUsers]);
rbm.dispose();
