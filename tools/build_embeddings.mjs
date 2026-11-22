// Precomputes CLIP text embeddings for every label in tools/labels.json,
// so the web app only has to process the image at runtime.
//
// Usage (from the repo root):
//   cd tools && npm install && node build_embeddings.mjs
//
// Output (repo root):
//   labels.json     – the label names, in order
//   embeddings.bin  – Float32 array, labels × 512, L2-normalised
import { AutoTokenizer, CLIPTextModelWithProjection } from "@huggingface/transformers";
import { readFileSync, writeFileSync } from "node:fs";

const MODEL = "Xenova/clip-vit-base-patch32";
// Averaging a few phrasings ("prompt ensembling") makes CLIP's guesses more accurate.
const TEMPLATES = ["a photo of a {}.", "a picture of a {}.", "a close-up photo of a {}.", "a drawing of a {}."];
const BATCH = 64;

const { labels } = JSON.parse(readFileSync(new URL("./labels.json", import.meta.url)));
const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
const model = await CLIPTextModelWithProjection.from_pretrained(MODEL, { dtype: "fp32" });

async function embed(texts) {
  const inputs = tokenizer(texts, { padding: true, truncation: true });
  const { text_embeds } = await model(inputs);
  return text_embeds; // [n, 512]
}

const DIM = 512;
const out = new Float32Array(labels.length * DIM);
for (let start = 0; start < labels.length; start += BATCH) {
  const chunk = labels.slice(start, start + BATCH);
  for (const t of TEMPLATES) {
    const emb = await embed(chunk.map((l) => t.replace("{}", l)));
    const data = emb.data;
    for (let i = 0; i < chunk.length; i++) {
      // normalise each template's embedding, then accumulate
      let n = 0;
      for (let j = 0; j < DIM; j++) n += data[i * DIM + j] ** 2;
      n = Math.sqrt(n);
      for (let j = 0; j < DIM; j++) out[(start + i) * DIM + j] += data[i * DIM + j] / n;
    }
  }
  process.stdout.write(`\r${Math.min(start + BATCH, labels.length)}/${labels.length}`);
}
// final normalisation of the averaged vector
for (let i = 0; i < labels.length; i++) {
  let n = 0;
  for (let j = 0; j < DIM; j++) n += out[i * DIM + j] ** 2;
  n = Math.sqrt(n);
  for (let j = 0; j < DIM; j++) out[i * DIM + j] /= n;
}

writeFileSync(new URL("../embeddings.bin", import.meta.url), Buffer.from(out.buffer));
writeFileSync(new URL("../labels.json", import.meta.url), JSON.stringify(labels));
console.log(`\nWrote ${labels.length} labels × ${DIM} dims.`);
