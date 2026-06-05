import { writeFileSync } from "fs";
import { resolve } from "path";
import {
  ARCHETYPES,
  generateDataset,
  generateUser,
  mulberry32,
} from "./dataset.js";

// ── Train dataset: 200 usuarios fijos ────────────────────────────────────────

const train = generateDataset({
  archetypes: ARCHETYPES,
  usersPerArchetype: 50,
  noise: 0.05,
  seed: 42,
});

const trainRecords = train.data.map((preferences, i) => ({
  archetype: ARCHETYPES[train.labels[i]!]!.name,
  preferences,
}));

writeFileSync(
  resolve("data/train.json"),
  JSON.stringify(trainRecords, null, 2)
);
console.log(`✓ data/train.json — ${trainRecords.length} usuarios`);

// ── New users: 10 usuarios nombrados ─────────────────────────────────────────

const NEW_USERS: { name: string; archetype: string }[] = [
  { name: "Arancha", archetype: "mexicano" },
  { name: "Jose",    archetype: "picante"  },
  { name: "Juan",    archetype: "cuchara"  },
  { name: "Jesús",   archetype: "picante"  },
  { name: "Antoni",  archetype: "italiano" },
  { name: "Elena",   archetype: "asiatico" },
  { name: "Maria",   archetype: "mexicano" },
  { name: "Ramón",   archetype: "cuchara"  },
  { name: "Laura",   archetype: "asiatico" },
  { name: "Marta",   archetype: "italiano" },
];

const rng = mulberry32(99);

const newUserRecords = NEW_USERS.map(({ name, archetype: archetypeName }) => {
  const archetype = ARCHETYPES.find((a) => a.name === archetypeName)!;
  return {
    name,
    archetype: archetypeName,
    preferences: generateUser(archetype, 0.1, rng),
  };
});

writeFileSync(
  resolve("data/new-users.json"),
  JSON.stringify(newUserRecords, null, 2)
);
console.log(`✓ data/new-users.json — ${newUserRecords.length} usuarios`);
