import { writeFileSync } from "fs";
import { resolve } from "path";
import {
  type Cuisine,
  CUISINES,
  generateDataset,
  generateUser,
  mulberry32,
} from "./dataset.js";

// ── Train dataset: 4 cuisines × 60 users ─────────────────────────────────────

const train = generateDataset({
  usersPerCuisine: 60,
  spicyProb: 0.45,
  noise: 0.03,
  seed: 42,
});

const trainRecords = train.data.map((preferences, i) => ({
  cuisine: CUISINES[train.cuisineLabels[i]!]!,
  spicy: train.spicyLabels[i]!,
  preferences,
}));

writeFileSync(resolve("data/train.json"), JSON.stringify(trainRecords, null, 2));
const nSpicy = trainRecords.filter((r) => r.spicy).length;
console.log(
  `✓ data/train.json — ${trainRecords.length} usuarios (${nSpicy} picantes, ${trainRecords.length - nSpicy} no)`
);

// ── New users: matched pairs (same cuisine, different spicy preference) ────────
// The key moment of the article: two diners from the SAME cuisine separated only
// by the spicy latent unit.

const NEW_USERS: { name: string; cuisine: Cuisine; spicy: boolean }[] = [
  { name: "Arancha", cuisine: "mexicano", spicy: false },
  { name: "Maria",   cuisine: "mexicano", spicy: true  },
  { name: "Antoni",  cuisine: "italiano", spicy: false },
  { name: "Marta",   cuisine: "italiano", spicy: true  },
  { name: "Juan",    cuisine: "cuchara",  spicy: false },
  { name: "Ramón",   cuisine: "cuchara",  spicy: true  },
  { name: "Elena",   cuisine: "asiatico", spicy: false },
  { name: "Laura",   cuisine: "asiatico", spicy: true  },
  { name: "Jose",    cuisine: "mexicano", spicy: true  },
  { name: "Jesús",   cuisine: "asiatico", spicy: true  },
];

const rng = mulberry32(22);

const newUserRecords = NEW_USERS.map(({ name, cuisine, spicy }) => ({
  name,
  cuisine,
  spicy,
  preferences: generateUser({ cuisine, spicy }, 0, rng),
}));

writeFileSync(
  resolve("data/new-users.json"),
  JSON.stringify(newUserRecords, null, 2)
);
console.log(`✓ data/new-users.json — ${newUserRecords.length} usuarios`);
