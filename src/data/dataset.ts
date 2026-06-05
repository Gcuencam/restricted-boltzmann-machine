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
// Modelo generador: cada usuario tiene DOS rasgos latentes independientes
//
//   1. cocina  ∈ {mexicano, italiano, cuchara, asiatico}   (one-hot)
//   2. picante ∈ {sí, no}                                  (moneda independiente)
//
// El picante es DELIBERADAMENTE transversal: un amante del picante eleva su
// probabilidad en TODOS los platos picantes, sin importar de qué cocina sean.
// Por eso no "se ve" en los clústeres de cocina y la RBM tiene que redescubrirlo
// como un factor latente propio. La cocina italiana no tiene plato picante: actúa
// como control ortogonal que demuestra que el picante no es un eje culinario más.
// ─────────────────────────────────────────────────────────────────────────────

export type Cuisine = "mexicano" | "italiano" | "cuchara" | "asiatico";
export const CUISINES: Cuisine[] = ["mexicano", "italiano", "cuchara", "asiatico"];

/** Cocina a la que pertenece cada plato (alineado por índice con DISHES). */
export const DISH_CUISINE: Cuisine[] = [
  "mexicano", "mexicano", "mexicano", // Tacos, Quesadillas, Totopos
  "italiano", "italiano", "italiano", // Rigatoni, Lasaña, Risotto
  "cuchara",  "cuchara",  "cuchara",  // Fabada, Lentejas, Sopas
  "asiatico", "asiatico", "asiatico", // Kimchi, Sushi, Pad Thai
];

/** Plato picante: el rasgo transversal que la RBM debe descubrir. */
export const DISH_SPICY: boolean[] = [
  false, false, true,  // Tacos no, Quesadillas no, Totopos con jalapeños SÍ
  false, false, false, // italianos: ninguno (cocina de control)
  false, true,  false, // Fabada no, Lentejas con chorizo picante SÍ, Sopas no
  true,  false, true,  // Kimchi SÍ, Sushi no, Pad Thai SÍ
];

/** Índices de los platos picantes — útil para inspección/visualización. */
export const SPICY_DISHES = DISH_SPICY.flatMap((s, d) => (s ? [d] : []));

/**
 * Parámetros del generador. Separar estas probabilidades hace explícito el
 * supuesto del experimento y permite afinar la nitidez de la señal.
 */
export interface GenProbs {
  /** Plato (no picante) de tu propia cocina. */
  ownMild: number;
  /** Plato picante de tu cocina y te gusta el picante. */
  ownSpicyLover: number;
  /** Plato picante de tu cocina pero NO te gusta el picante. */
  ownSpicyAvoid: number;
  /** Plato picante de OTRA cocina y te gusta el picante (el cruce transversal). */
  crossSpicyLover: number;
  /** Cualquier otro plato (otra cocina, no picante / picante sin afición). */
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
  /** Probabilidad de que un usuario sea amante del picante. */
  spicyProb: number;
  /** Probabilidad de voltear cada bit tras el muestreo. */
  noise: number;
  seed?: number;
  probs?: GenProbs;
}

export interface DatasetRecord {
  cuisine: Cuisine;
  spicy: boolean;
  /** Vector binario [N_DISHES]. */
  preferences: number[];
}

export interface Dataset {
  /** Matriz binaria [nUsers, N_DISHES]. */
  data: number[][];
  /** Índice de cocina que generó cada usuario. */
  cuisineLabels: number[];
  /** Rasgo picante de cada usuario (ground truth oculto). */
  spicyLabels: boolean[];
}

/** Probabilidad de que un usuario quiera un plato dado su perfil latente. */
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

/** Genera las preferencias de un usuario dado su perfil latente (cocina × picante). */
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
 * Genera un dataset binario sintético compositivo (cocina × picante).
 *
 * Para cada cocina se generan `usersPerCuisine` usuarios; cada uno es amante del
 * picante con probabilidad `spicyProb`, de forma INDEPENDIENTE de su cocina.
 * Esto hace del picante un factor latente transversal genuino.
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

  // Fisher-Yates shuffle con el mismo rng.
  for (let i = data.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [data[i], data[j]] = [data[j]!, data[i]!];
    [cuisineLabels[i], cuisineLabels[j]] = [cuisineLabels[j]!, cuisineLabels[i]!];
    [spicyLabels[i], spicyLabels[j]] = [spicyLabels[j]!, spicyLabels[i]!];
  }

  return { data, cuisineLabels, spicyLabels };
}

export { mulberry32 };
