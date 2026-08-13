# Deploying the demo

Three ways to run it. Local needs nothing but the repo; Docker reproduces the
Spaces environment exactly; Spaces gives the public link.

## 1. Local

```bash
uv sync
python src/edd/fetch_mvtec.py bottle
uv run python src/edd/export.py bottle      # writes models/bottle.pt
uv run streamlit run app.py
```

## 2. Docker (same environment Spaces uses: CPU, port 7860)

```bash
docker build -t defect-detector .
docker run --rm -p 7860:7860 defect-detector
```

The build bakes the WideResNet50-2 weights into the image, so the first request
does not stall on a 100 MB download. `models/` must be populated before
building — the artefacts are what make the image self-contained.

## 3. Hugging Face Spaces (the public link)

Spaces needs a Hugging Face account and a write token, which only you can
create — I can't create accounts or handle tokens on your behalf.

1. Create a token at https://huggingface.co/settings/tokens (role: **write**).
2. Create the Space (Docker SDK) at https://huggingface.co/new-space —
   name it `explainable-defect-detector`.
3. Push:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
git remote add space https://huggingface.co/spaces/<your-hf-username>/explainable-defect-detector
git push space main
```

Spaces reads the `Dockerfile` and serves on 7860 automatically.

### One thing to check before pushing

`models/*.pt` are ~9–14 MB each. Three categories is ~35 MB, which is fine for
a normal git push. If you export all 15, use Git LFS:

```bash
git lfs install
git lfs track "models/*.pt"
git add .gitattributes && git commit -m "Track model artefacts with LFS"
```

### Space metadata

Spaces expects YAML frontmatter in the Space's own `README.md`. Keep the repo
README as-is for GitHub and add this at the top of the copy you push to Spaces:

```yaml
---
title: Explainable Visual Defect Detector
emoji: 🔍
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---
```
