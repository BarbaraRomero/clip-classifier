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
