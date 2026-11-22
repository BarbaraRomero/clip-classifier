import { AutoProcessor, CLIPVisionModelWithProjection, RawImage } from "@huggingface/transformers";
import { readFileSync } from "node:fs";
const MODEL = "Xenova/clip-vit-base-patch32";
const labels = JSON.parse(readFileSync("../labels.json"));
const buf = readFileSync("../embeddings.bin");
const E = new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
const processor = await AutoProcessor.from_pretrained(MODEL);
const vision = await CLIPVisionModelWithProjection.from_pretrained(MODEL, { dtype: "q8" });
for (const url of process.argv.slice(2)) {
  const img = await RawImage.read(url);
  const { image_embeds } = await vision(await processor(img));
  const v = image_embeds.data; let n = Math.hypot(...v);
  const logits = labels.map((_, i) => { let s = 0; for (let j = 0; j < 512; j++) s += E[i*512+j]*v[j]/n; return 100*s; });
  const m = Math.max(...logits); const ex = logits.map(l => Math.exp(l - m)); const Z = ex.reduce((a,b)=>a+b);
  const top = ex.map((e,i)=>[e/Z, labels[i]]).sort((a,b)=>b[0]-a[0]).slice(0,5);
  console.log(url.split("/").pop(), "→", top.map(([p,l]) => `${l} ${(p*100).toFixed(1)}%`).join(" | "));
}
