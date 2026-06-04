import { readFileSync } from "fs";
import { resolve } from "path";
import type { Dataset } from "./dataset.js";
import { ARCHETYPES } from "./dataset.js";

export interface NamedUser {
  name: string;
  archetype: string;
  preferences: number[];
}

export function loadTrainDataset(): Dataset {
  const raw = JSON.parse(readFileSync(resolve("data/train.json"), "utf-8")) as {
    archetype: string;
    preferences: number[];
  }[];

  return {
    data: raw.map((r) => r.preferences),
    labels: raw.map((r) => ARCHETYPES.findIndex((a) => a.name === r.archetype)),
  };
}

export function loadNewUsers(): NamedUser[] {
  return JSON.parse(
    readFileSync(resolve("data/new-users.json"), "utf-8")
  ) as NamedUser[];
}
