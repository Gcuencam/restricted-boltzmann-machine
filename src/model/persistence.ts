import * as tf from "@tensorflow/tfjs-node";
import { readFileSync, writeFileSync } from "fs";
import { resolve } from "path";
import { RBM } from "./rbm.js";

interface SavedModel {
  nVisible: number;
  nHidden: number;
  W: number[][];
  bv: number[];
  bh: number[];
}

export async function saveModel(rbm: RBM, path: string): Promise<void> {
  const [W, bv, bh] = await Promise.all([
    rbm.W.array(),
    rbm.bv.array(),
    rbm.bh.array(),
  ]);
  const saved: SavedModel = { nVisible: rbm.nVisible, nHidden: rbm.nHidden, W, bv, bh };
  writeFileSync(resolve(path), JSON.stringify(saved, null, 2));
}

export function loadModel(path: string): RBM {
  const saved = JSON.parse(readFileSync(resolve(path), "utf-8")) as SavedModel;
  const rbm = new RBM({ nVisible: saved.nVisible, nHidden: saved.nHidden });
  rbm.W.assign(tf.tensor2d(saved.W));
  rbm.bv.assign(tf.tensor1d(saved.bv));
  rbm.bh.assign(tf.tensor1d(saved.bh));
  return rbm;
}
