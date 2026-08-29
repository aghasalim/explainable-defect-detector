# Deploying the demo

## Hosting reality check (August 2026)

Hugging Face Spaces **no longer runs Docker or Gradio Spaces on the free tier**: both
now return `402 Payment Required` and require a PRO subscription. Only `static` Spaces
are free, and a static Space cannot run PyTorch server-side.

So the recommended host is **Streamlit Community Cloud**: free, permanent, and it runs
this app from the GitHub repo with no code changes.

---

## 1. Streamlit Community Cloud (recommended, free)

Everything in the repo is already prepared. You only need to do the OAuth step, which
requires your GitHub login.

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `aghasalim/explainable-defect-detector`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings** and set **Python version: 3.12**.
5. Click **Deploy**.

First build takes ~5 minutes (it installs CPU-only PyTorch and downloads the
WideResNet50-2 weights once). You get a permanent URL like
`https://explainable-defect-detector.streamlit.app`.

**Then tell me the URL** and I will add the live badge to both READMEs.

### Why the requirements pin `+cpu`

On linux-amd64 the default PyPI `torch` is the CUDA build, roughly 800 MB of `nvidia-*`
wheels that would exhaust the free tier for a demo that never touches a GPU.
`requirements.txt` pins `torch==2.13.0+cpu` against the PyTorch CPU index for that reason.

---

## 2. Docker (verified working)

The image is built and tested, it serves on port 7860 and scores every bundled sample
correctly on CPU.

```bash
docker build -t defect-detector .
docker run --rm -p 7860:7860 defect-detector
```

Image is ~2.5 GB, mostly PyTorch plus the backbone weights baked in at build time so the
first request does not stall on a 100 MB download. This runs unchanged on Render, Fly.io,
Railway, or any container host, and on HF Spaces if you ever take a PRO subscription.

---

## 3. Hugging Face Spaces (requires PRO, ~$9/month)

Kept ready in case you subscribe. `spaces/README.md` already carries the correct YAML
frontmatter (Docker SDK, port 7860).

```bash
hf auth login                                   # write token
./scripts/deploy_space.sh aghasalim
```

The script stages only what the Space needs and uploads via `hf upload`, so no git-lfs
setup is required. Without PRO, `hf repos create --type space --sdk docker` fails with
`402 Payment Required`.

---

## Updating the model artefacts

`models/*.pt` hold the memory bank and calibrated threshold, they are what make the
image self-contained. Regenerate and refresh the demo samples with:

```bash
uv run python src/edd/export.py bottle          # -> models/bottle.pt
uv run python src/edd/verify_threshold.py       # audit the shipped threshold
uv run python src/edd/samples.py                # re-pick demo samples from real scores
```

`samples.py` chooses each sample by scoring the real test split with the exported
artefact, so a `_MISSED` sample is a genuine false negative rather than a broken demo.
If you export all 15 categories, use Git LFS:

```bash
git lfs install && git lfs track "models/*.pt"
```
