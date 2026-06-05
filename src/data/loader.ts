import { readFileSync } from "fs";
import { resolve } from "path";
import { type Cuisine, CUISINES, type Dataset } from "./dataset.js";

export interface NamedUser {
  name: string;
  cuisine: Cuisine;
  spicy: boolean;
  preferences: number[];
}

interface TrainRecord {
  cuisine: Cuisine;
  spicy: boolean;
  preferences: number[];
}

export function loadTrainDataset(): Dataset {
  const raw = JSON.parse(
    readFileSync(resolve("data/train.json"), "utf-8")
  ) as TrainRecord[];

  return {
    data: raw.map((r) => r.preferences),
    cuisineLabels: raw.map((r) => CUISINES.indexOf(r.cuisine)),
    spicyLabels: raw.map((r) => r.spicy),
  };
}

export function loadNewUsers(): NamedUser[] {
  return JSON.parse(
    readFileSync(resolve("data/new-users.json"), "utf-8")
  ) as NamedUser[];
}
