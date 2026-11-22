# CLIP Image Classifier

A small web app where students upload an image and the [CLIP](https://openai.com/research/clip) model estimates what it shows, giving a percentage for each description.

- Runs entirely in the browser using [Transformers.js](https://huggingface.co/docs/transformers.js) (model: `Xenova/clip-vit-base-patch32`)
- Images are never uploaded to a server
- The descriptions CLIP chooses from can be edited, and there are presets for animals, food, places and emotions
- Includes a "How does it work?" section for students

## Run locally

Open `index.html` through a local server (ES modules don't load from `file://`):

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

## Publish on GitHub Pages

1. Push this folder to a GitHub repository.
2. On GitHub, go to **Settings → Pages**.
3. Under **Source**, choose **Deploy from a branch**, then select `main` and `/ (root)`.
4. After a minute the app will be live at `https://<your-username>.github.io/<repo-name>/`.

The first visit downloads the model (~150 MB). After that the browser caches it.

## Game modes

- **Big Brain** compares the image with 1,437 things: ImageNet-1k (the cleaned class names CLIP's researchers used, from [open_clip](https://github.com/mlfoundations/open_clip)), the [Open Images](https://storage.googleapis.com/openimages/web/index.html) object classes, and a few extras such as "baby" and "selfie". The text side of these is precomputed into `embeddings.bin` (labels in `labels.json`), so the browser only has to process the image.
- **My Options** lets students write their own descriptions, which CLIP compares the image with live.

### Changing the Big Brain list

Edit `tools/labels.json`, then rebuild the embeddings:

```bash
cd tools && npm install && node build_embeddings.mjs
```
