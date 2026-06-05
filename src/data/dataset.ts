export const DISHES = [
  "Tacos al pastor",
  "Quesadillas",
  "Totopos con jalapeños",
  "Rigatoni Carbonara",
  "Lasaña",
  "Risotto",
  "Fabada Asturiana",
  "Lentejas con chorizo picante",
  "Sopas de ajo",
  "Kimchi Chigae",
  "Sushi",
  "Pad Thai",
] as const;

export const N_DISHES = DISHES.length; // 12

export interface Archetype {
  name: string;
  /** Probability that a user of this archetype wants each dish. Length = N_DISHES. */
  dishProbs: number[];
}

export interface DatasetConfig {
  archetypes: Archetype[];
  usersPerArchetype: number;
  /** Probability of randomly flipping each bit after sampling */
  noise: number;
  seed?: number;
}

export interface Dataset {
  /** Binary matrix [nUsers, N_DISHES] */
  data: number[][];
  /** Archetype index that generated each user */
  labels: number[];
}

// Orden de platos: Tacos, Quesadillas, Totopos, Rigatoni, Lasaña, Risotto,
//                  Fabada, Lentejas, Sopas, Kimchi, Sushi, Pad Thai
//
// Ejes transversales deliberados:
//   Totopos con jalapeños  → mexicano (0.8) + picante (0.9)
//   Lentejas con chorizo   → cuchara  (0.8) + picante (0.8)
//   Kimchi Chigae          → asiático (0.8) + picante (0.8)
//   Sopas de ajo           → cuchara  (0.9) + picante (0.6, lleva guindilla)
export const ARCHETYPES: Archetype[] = [
  {
    name: "mexicano",
    dishProbs: [0.9, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
  },
  {
    name: "italiano",
    dishProbs: [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
  },
  {
    name: "cuchara",
    dishProbs: [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.9, 0.8, 0.9, 0.3, 0.1, 0.1],
  },
  {
    name: "asiatico",
    dishProbs: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.9, 0.9],
  },
  {
    name: "picante",
    dishProbs: [0.5, 0.3, 0.9, 0.1, 0.1, 0.1, 0.1, 0.8, 0.6, 0.8, 0.2, 0.3],
  },
];

/** Mulberry32 — fast seeded PRNG. Returns a () => number in [0, 1). */
function mulberry32(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleBernoulli(prob: number, rng: () => number): 0 | 1 {
  return rng() < prob ? 1 : 0;
}

/**
 * Generate a synthetic binary dataset from archetype profiles.
 *
 * For each user:
 *   1. Sample each dish from the archetype's dishProbs   → vector binario
 *   2. Apply noise: flip each bit independently with probability config.noise
 *   3. Shuffle all users before returning
 */
export function generateDataset(config: DatasetConfig): Dataset {
  const rng = config.seed !== undefined ? mulberry32(config.seed) : Math.random;
  const rows: number[][] = [];
  const labels: number[] = [];

  for (let a = 0; a < config.archetypes.length; a++) {
    const archetype = config.archetypes[a];
    if (!archetype) continue;

    for (let u = 0; u < config.usersPerArchetype; u++) {
      const user: number[] = [];

      for (let d = 0; d < N_DISHES; d++) {
        const prob = archetype.dishProbs[d];
        if (prob === undefined) continue;

        const sampled = sampleBernoulli(prob, rng);
        const noisy = sampleBernoulli(config.noise, rng) ? 1 - sampled : sampled;
        user.push(noisy);
      }

      rows.push(user);
      labels.push(a);
    }
  }

  // Fisher-Yates shuffle usando el mismo rng
  for (let i = rows.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [rows[i], rows[j]] = [rows[j]!, rows[i]!];
    [labels[i], labels[j]] = [labels[j]!, labels[i]!];
  }

  return { data: rows, labels };
}

/** Generate preferences for a single user given an archetype. */
export function generateUser(
  archetype: Archetype,
  noise: number,
  rng: () => number
): number[] {
  return archetype.dishProbs.map((prob) => {
    const sampled = sampleBernoulli(prob, rng);
    return sampleBernoulli(noise, rng) ? 1 - sampled : sampled;
  });
}

export { mulberry32 };
