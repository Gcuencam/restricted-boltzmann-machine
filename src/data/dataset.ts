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

// ─────────────────────────────────────────────────────────────────────────────
// Generative model: each user has TWO independent latent traits
//
//   1. cuisine  ∈ {mexicano, italiano, cuchara, asiatico}   (one-hot)
//   2. spicy    ∈ {yes, no}                                  (independent coin flip)
//
// Spicy is DELIBERATELY cross-cutting: a spicy lover raises their probability on
// ALL spicy dishes, regardless of which cuisine they belong to. That is why it
// does not "show up" in cuisine clusters and the RBM must rediscover it as its
// own latent factor. Italian cuisine has no spicy dish: it acts as an orthogonal
// control that demonstrates that spicy is not just another cuisine axis.
// ─────────────────────────────────────────────────────────────────────────────

export type Cuisine = "mexicano" | "italiano" | "cuchara" | "asiatico";
export const CUISINES: Cuisine[] = ["mexicano", "italiano", "cuchara", "asiatico"];

/** Cuisine each dish belongs to (index-aligned with DISHES). */
export const DISH_CUISINE: Cuisine[] = [
  "mexicano", "mexicano", "mexicano", // Tacos, Quesadillas, Totopos
  "italiano", "italiano", "italiano", // Rigatoni, Lasaña, Risotto
  "cuchara",  "cuchara",  "cuchara",  // Fabada, Lentejas, Sopas
  "asiatico", "asiatico", "asiatico", // Kimchi, Sushi, Pad Thai
];

/** Spicy dish: the cross-cutting trait the RBM must discover. */
export const DISH_SPICY: boolean[] = [
  false, false, true,  // Tacos no, Quesadillas no, Totopos con jalapeños YES
  false, false, false, // Italian: none (control cuisine)
  false, true,  false, // Fabada no, Lentejas con chorizo picante YES, Sopas no
  true,  false, true,  // Kimchi YES, Sushi no, Pad Thai YES
];

/** Índices de los platos picantes — útil para inspección/visualización. */
export const SPICY_DISHES = DISH_SPICY.flatMap((s, d) => (s ? [d] : []));

/**
 * Generator parameters. Separating these probabilities makes the experiment's
 * assumptions explicit and allows fine-tuning the signal clarity.
 */
export interface GenProbs {
  /** Non-spicy dish from your own cuisine. */
  ownMild: number;
  /** Spicy dish from your cuisine and you like spicy food. */
  ownSpicyLover: number;
  /** Spicy dish from your cuisine but you do NOT like spicy food. */
  ownSpicyAvoid: number;
  /** Spicy dish from ANOTHER cuisine and you like spicy food (the cross-cutting overlap). */
  crossSpicyLover: number;
  /** Any other dish (another cuisine, non-spicy / spicy without interest). */
  base: number;
}

export const DEFAULT_PROBS: GenProbs = {
  ownMild: 0.85,
  ownSpicyLover: 0.92,
  ownSpicyAvoid: 0.35,
  crossSpicyLover: 0.65,
  base: 0.04,
};

export interface UserProfile {
  cuisine: Cuisine;
  spicy: boolean;
}

export interface DatasetConfig {
  usersPerCuisine: number;
  /** Probability that a user is a spicy lover. */
  spicyProb: number;
  /** Probability of flipping each bit after sampling. */
  noise: number;
  seed?: number;
  probs?: GenProbs;
}

export interface DatasetRecord {
  cuisine: Cuisine;
  spicy: boolean;
  /** Binary vector [N_DISHES]. */
  preferences: number[];
}

export interface Dataset {
  /** Binary matrix [nUsers, N_DISHES]. */
  data: number[][];
  /** Cuisine index that generated each user. */
  cuisineLabels: number[];
  /** Spicy trait of each user (hidden ground truth). */
  spicyLabels: boolean[];
}

/** Probability that a user wants a dish given their latent profile. */
export function dishProb(
  dish: number,
  cuisine: Cuisine,
  spicy: boolean,
  p: GenProbs = DEFAULT_PROBS
): number {
  const own = DISH_CUISINE[dish] === cuisine;
  const hot = DISH_SPICY[dish];

  if (own && !hot) return p.ownMild;
  if (own && hot) return spicy ? p.ownSpicyLover : p.ownSpicyAvoid;
  if (!own && hot) return spicy ? p.crossSpicyLover : p.base;
  return p.base;
}

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

/** Generates the preferences of a user given their latent profile (cuisine × spicy). */
export function generateUser(
  profile: UserProfile,
  noise: number,
  rng: () => number,
  probs: GenProbs = DEFAULT_PROBS
): number[] {
  const out: number[] = [];
  for (let d = 0; d < N_DISHES; d++) {
    const prob = dishProb(d, profile.cuisine, profile.spicy, probs);
    const sampled = sampleBernoulli(prob, rng);
    out.push(sampleBernoulli(noise, rng) ? 1 - sampled : sampled);
  }
  return out;
}

/**
 * Generates a synthetic compositional binary dataset (cuisine × spicy).
 *
 * For each cuisine, `usersPerCuisine` users are generated; each one is a spicy
 * lover with probability `spicyProb`, INDEPENDENTLY of their cuisine.
 * This makes spicy a genuinely cross-cutting latent factor.
 */
export function generateDataset(config: DatasetConfig): Dataset {
  const rng = config.seed !== undefined ? mulberry32(config.seed) : Math.random;
  const probs = config.probs ?? DEFAULT_PROBS;

  const data: number[][] = [];
  const cuisineLabels: number[] = [];
  const spicyLabels: boolean[] = [];

  for (let c = 0; c < CUISINES.length; c++) {
    const cuisine = CUISINES[c]!;
    for (let u = 0; u < config.usersPerCuisine; u++) {
      const spicy = rng() < config.spicyProb;
      data.push(generateUser({ cuisine, spicy }, config.noise, rng, probs));
      cuisineLabels.push(c);
      spicyLabels.push(spicy);
    }
  }

  // Fisher-Yates shuffle using the same rng.
  for (let i = data.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [data[i], data[j]] = [data[j]!, data[i]!];
    [cuisineLabels[i], cuisineLabels[j]] = [cuisineLabels[j]!, cuisineLabels[i]!];
    [spicyLabels[i], spicyLabels[j]] = [spicyLabels[j]!, spicyLabels[i]!];
  }

  return { data, cuisineLabels, spicyLabels };
}

export { mulberry32 };
