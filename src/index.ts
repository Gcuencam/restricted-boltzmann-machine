import { generateDataset, ARCHETYPES, DISHES, N_DISHES } from "./data/dataset.js";

const dataset = generateDataset({
  archetypes: ARCHETYPES,
  usersPerArchetype: 50,
  noise: 0.05,
  seed: 42,
});

console.log(`Usuarios generados: ${dataset.data.length}`);
console.log(`Platos por usuario: ${dataset.data[0]?.length}`);
console.log(`Etiquetas únicas: ${[...new Set(dataset.labels)].sort().join(", ")}`);

// Media de activación por plato (debería ser ~0.4-0.6, no 0 ni 1)
const means = Array.from({ length: N_DISHES }, (_, d) =>
  dataset.data.reduce((sum, row) => sum + (row[d] ?? 0), 0) / dataset.data.length
);
console.log("\nActivación media por plato:");
DISHES.forEach((dish, i) => {
  const bar = "█".repeat(Math.round((means[i] ?? 0) * 20));
  console.log(`  ${dish.padEnd(28)} ${bar} ${(means[i] ?? 0).toFixed(2)}`);
});
