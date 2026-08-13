# Cats vs Dogs — MLOps Pipeline

Binary image classification (cats vs dogs) for a pet adoption platform, built as an
end-to-end MLOps pipeline: data versioning (DVC), experiment tracking (MLflow), a
FastAPI inference service, Docker containerization, GitHub Actions CI/CD, and
Prometheus-style monitoring.

The baseline CNN (`src/inference/model.py`) reaches **~81.7% validation accuracy**
after 8 epochs of training, and scored **~86% accuracy** on a live post-deployment
evaluation sample (50 images) drawn from the held-out test split and sent through the
deployed API's `/predict` endpoint.

## Project layout

- `src/data/preprocess.py` — preprocessing (resize to 224x224 RGB, stratified 80/10/10
  split); wired up as the `preprocess` stage in `dvc.yaml`
- `src/train/train.py` — training loop with MLflow logging (params, per-epoch metrics,
  confusion matrix artifact, model checkpoint)
- `src/inference/` — model definition (`model.py`, a small CNN) and predict logic
  (`predict.py`), shared by the API and the scripts
- `src/api/main.py` — FastAPI service (`/health`, `/predict`, `/metrics`)
- `tests/` — pytest unit tests (preprocessing, model, inference, API, metrics,
  evaluate_deployed)
- `scripts/smoke_test.py` — post-deploy health check + single-image prediction check
- `scripts/evaluate_deployed.py` — samples the test split against the live API and
  writes an accuracy report
- `deploy/docker-compose.yml` — deployment manifest (pulls `axlebits/catsdogs-api:latest`
  from Docker Hub)
- `Dockerfile` — inference service image (CPU-only torch build, so the published image
  stays small and runs anywhere)
- `.github/workflows/ci.yml` — CI: test, build, and publish to Docker Hub
- `.github/workflows/cd.yml` — CD: deploy + smoke test on the self-hosted runner
- `dvc.yaml` / `dvc.lock` — DVC pipeline definition (`data/raw/PetImages` ->
  `data/processed`)
- `data/raw/PetImages.dvc`, `models/model.pt.dvc` — DVC pointer files; the actual raw
  images and trained weights live in the DVC remote, not in git
- `.dvc/config` — DVC remote configuration
- `models/` — trained weights (`model.pt`), DVC-tracked and gitignored
- `mlruns/` — local MLflow tracking store, gitignored
- `results/` — output of `scripts/evaluate_deployed.py`, gitignored

## One-time setup

Local development uses a project-local virtual environment rather than the global
Python install.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\dvc pull
```

`requirements.txt` pins `torch==2.12.0` / `torchvision==0.27.0` — check that file for
the current pins if they've since moved. `requirements-dev.txt` additionally installs
MLflow, DVC, scikit-learn, and matplotlib on top of the runtime requirements.

The DVC remote (`.dvc/config`) is a **local folder** at `../dvc-storage/assignment2`
(a directory sibling to this repo, not inside it) — `dvc pull` / `dvc push` are plain
file copies with no credentials, OAuth, or Google Drive setup required. (The `dvc[gdrive]`
extra in `requirements-dev.txt` is installed but unused by this project's remote.)

### GPU training (optional)

`requirements.txt` installs the CPU build of torch. To train on a local GPU, reinstall
the matching CUDA build after the setup above (adjust the version tag to your CUDA
toolkit and to whatever torch/torchvision versions are currently pinned in
`requirements.txt`):

```powershell
.venv\Scripts\python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130 --force-reinstall --no-deps
```

## Train the model

```powershell
# 1. Preprocess raw images into the 224x224 train/val/test split (skip if
#    data/processed was already restored by `dvc pull`, or re-run to regenerate it)
.venv\Scripts\dvc repro preprocess

# 2. Train (defaults to 8 epochs, batch size 32, lr 1e-3; auto-picks cuda if available)
.venv\Scripts\python.exe -m src.train.train --epochs 8 --device cuda

# 3. Version the new weights with DVC
.venv\Scripts\dvc add models/model.pt
.venv\Scripts\dvc push
```

Training logs params/metrics/artifacts to MLflow under the `catsdogs-baseline-cnn`
experiment (local store at `./mlruns`); inspect a run with:

```powershell
.venv\Scripts\mlflow ui
```

then open `http://localhost:5000`. A confusion matrix is also written to
`confusion_matrix.png` and logged as an MLflow artifact each run.

## Run the API locally

Directly, using the venv:

```powershell
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Or built into a container, same as the CI/CD image:

```bash
docker build -t catsdogs-api:local .
docker run -d -p 8000:8000 -v "$(pwd)/models:/app/models:ro" catsdogs-api:local
curl http://localhost:8000/health
```

The API reads its weights from `MODEL_PATH` (default `models/model.pt`) and exposes:

- `GET /health` — liveness check
- `POST /predict` — multipart image upload, returns `{"label": "cat"|"dog", "probabilities": {...}}`
- `GET /metrics` — Prometheus text-format metrics (request counts and latency histograms)

## Deploy via Docker Compose

```bash
export DOCKERHUB_USERNAME=axlebits
docker compose -f deploy/docker-compose.yml pull
docker compose -f deploy/docker-compose.yml up -d
python scripts/smoke_test.py scripts/sample_pet.jpg
```

`deploy/docker-compose.yml` pulls `${DOCKERHUB_USERNAME}/catsdogs-api:latest` (the
published image is `axlebits/catsdogs-api:latest`) and mounts `../models` (i.e. this
repo's `models/` directory) into the container read-only at `/app/models`, so the
running container always serves whatever weights are on the host without needing a
rebuild.

## CI/CD

**CI** (`.github/workflows/ci.yml`) runs on every push to any branch and on pull
requests into `main`: sets up Python 3.11, installs the CPU build of torch plus
`requirements.txt`, and runs `pytest -v`. On a push to `main` it additionally builds
and pushes the Docker image to Docker Hub as `axlebits/catsdogs-api:latest` and
`axlebits/catsdogs-api:<git-sha>` (using the `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`
repo secrets). On any other branch or PR it only does a build-verification pass —
image built, not pushed.

**CD** (`.github/workflows/cd.yml`) is triggered by a `workflow_run` event once the CI
workflow completes successfully on `main`. It runs on `self-hosted`, **not** a
GitHub-hosted runner — this is required because the job deploys via Docker Compose
directly onto the host machine, and a GitHub-hosted runner's VM is torn down after the
job and can't host a persistent deployment. The job pulls the new image, runs
`docker compose up -d`, and then runs `scripts/smoke_test.py` against the freshly
deployed container.

For CD to pick up jobs, a self-hosted runner must be registered and running on the
deployment machine:

1. In the GitHub repo, go to **Settings > Actions > Runners > New self-hosted
   runner** and follow the generated download/config commands.
2. Start it with `run.cmd` (Windows) — it needs to stay running (or be installed as a
   service) for the CD workflow to have a runner to dispatch to.

## Post-deployment evaluation

```powershell
.venv\Scripts\python.exe scripts\evaluate_deployed.py --per-class 25
```

Sends `--per-class` images per class (default 50, i.e. 100 total if the flag is
omitted; the example above passes `--per-class 25` for a 50-image sample) from
`data/processed/test` to the running API's `/predict` endpoint, writes per-image
results to `results/evaluation.csv`, and prints an accuracy summary as JSON. A live run
against the deployed service with `--per-class 25` scored **~86% accuracy** on that
50-image sample (25 cats + 25 dogs).
