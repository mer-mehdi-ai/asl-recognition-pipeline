#  ASL Recognition Pipeline

[![CI](https://github.com/mer-mehdi-ai/asl-recognition-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/mer-mehdi-ai/asl-recognition-pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Recognising American Sign Language fingerspelling from images and from a webcam, with
**two interchangeable backends behind one pipeline**: MediaPipe hand landmarks + Random
Forest, and an end-to-end CNN. Same catalog, same split policy, same metrics, same
inference contract - so the two approaches can actually be compared, and either one can
be served over HTTP from the exact artifact that produced the published numbers.

The modelling here is deliberately ordinary. The point of the repository is everything
around it: reproducible runs, an evaluation protocol that does not lie to itself, and a
deployable service.

```
images ──▶ catalog ──▶ split ──┬─▶ landmarks ─▶ Random Forest ─┐
(29 classes)  (DataFrame)      │   (42 features)               ├─▶ immutable run ──▶ FastAPI
                               └─▶ tensors ───▶ CNN ───────────┘   metrics + model      webcam
```

---

## Why this exists

The starting point was a student project: a Colab notebook that trained a CNN, and three
scripts that pickled MediaPipe features, fitted a classifier and drove a Tkinter window.
It worked, and none of it could be re-run, compared or deployed - the label mapping was
hard-coded twice (and differently), the test set was drawn at random from near-duplicate
frames, and the only record of a result was a number printed in a cell.

This repository is the same two ideas rebuilt as a system. The engineering decisions and
what each one fixes are in [docs/architecture.md](docs/architecture.md).

## What is in the box

| | |
|---|---|
| **Typed config** | One YAML file describes a whole experiment; unknown keys are rejected at load time (`configs/`). |
| **Explicit catalog** | The dataset becomes a DataFrame (`path, label, order, split`) stored with every run - no implicit `ImageFolder` scan, no invisible class order. |
| **Honest splits** | `blocked` (default) and `random`, so the leakage on this dataset can be measured instead of argued about - see [docs/evaluation.md](docs/evaluation.md). |
| **Immutable runs** | `runs/<timestamp>-<name>-<config hash>/` with manifest (git SHA, platform, command), config, `metrics.json`, model and figures. Never overwritten. |
| **Two backends** | `landmark_rf` (CPU, real-time) and `cnn` (GPU, no hand detector) behind one `Recognizer` interface. |
| **Evaluation** | Accuracy, macro F1, per-class report, confusion matrix and the top confused pairs - for both backends. |
| **Explainability** | Grad-CAM and feature maps for CNN runs (`asl explain`). |
| **Serving** | FastAPI service loading a run directory, in a slim Docker image (no CUDA, no PyTorch). |
| **Real-time** | A tested text composer (dwell time + confidence gate) and an OpenCV webcam loop. |
| **CI** | Lint, 66 tests on Python 3.10 and 3.11, an end-to-end smoke run on the committed sample, and a Docker build that must answer `/health`. |

## Quickstart

```bash
git clone https://github.com/mer-mehdi-ai/asl-recognition-pipeline
cd asl-recognition-pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install            # or: make install-cpu for the CPU-only PyTorch build

# 1. end-to-end run on the 56 images committed in this repo - no Kaggle account needed
make smoke
asl runs list

# 2. the real thing
asl data pull                              # ~87k images from Kaggle
asl catalog --config configs/landmark_rf.yaml   # inspect the split before training
asl train   --config configs/landmark_rf.yaml   # landmarks + Random Forest
asl train   --config configs/cnn.yaml           # CNN (GPU recommended)

# 3. use it
asl predict image data/sample/L/L1.jpg --run latest
asl predict webcam --run latest            # 'q' quits, 'c' clears the buffer
asl serve --run latest                     # http://localhost:8000/docs
```

Serving the same run from a container:

```bash
make docker-build
make docker-run RUN=<run id>
curl -F "file=@data/sample/L/L1.jpg" localhost:8000/predict
# {"label":"L","confidence":0.6,"latency_ms":20.7,"backend":"landmark_rf","run_id":"..."}
```

## Results

The dataset is [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
(87 000 images, 29 classes: A-Z, `space`, `del`, `nothing`).

| Backend | Split | Test accuracy | Notes |
|---|---|---|---|
| CNN, 4 epochs | random (stratified) | **0.9874** | Full dataset, one T4. Reproduced from the original exploration ([notebook](notebooks/01_cnn_exploration.ipynb)). |
| CNN, 4 epochs | blocked | `make train-cnn` | |
| Landmarks + Random Forest | blocked | `make train` | |

The empty cells are deliberate: this repository does not publish numbers it has not
measured. `asl runs list` prints them for every run that exists on your machine, and each
one is traceable to a config, a git revision and a machine.

**The 0.9874 should not be read as "98.7% accurate at recognising sign language."** The
images come in long bursts of near-identical frames, so a random split puts neighbouring
frames of the same recording in both train and test. The `blocked` policy (hold out the
tail of each class's frame sequence) is the default here for that reason.
[docs/evaluation.md](docs/evaluation.md) has the argument in full.

### What the pipeline surfaced

Running the landmark backend over the committed 56-image sample (2 images per class):

```
hand detected in 66.1% of images
warning: no hand detected at all for 8 classes: C, D, E, N, U, V, X, del
test accuracy 0.9444 | macro F1 0.9259
```

Those eight classes are simply *absent* from the landmark model - MediaPipe never returns
a hand for them at these settings, so the Random Forest is never shown them and can never
predict them. This is the sharpest difference between the two backends, and it is invisible
in an accuracy number: the CNN sees pixels and always predicts something, the landmark
model depends entirely on an upstream detector. Both the detection rate and the list of
lost classes are written into every run's `metrics.json`. The sample is far too small to
generalise from - run `asl train --config configs/landmark_rf.yaml` to measure it on all
87 000 images.

## Repository layout

```
src/asl/
  config.py          typed experiment configuration (pydantic)
  runs.py            immutable run directories, manifests, metrics
  pipeline.py        catalog -> split -> features -> model -> metrics
  evaluate.py        one evaluation protocol for both backends
  explain.py         Grad-CAM, feature maps
  realtime.py        webcam loop + tested text composer
  registry.py        load the right backend out of a run
  cli.py             the `asl` command
  data/              catalog, split policies, MediaPipe features, Kaggle download
  models/            Recognizer interface, landmark_rf, cnn
  api/               FastAPI inference service
configs/             landmark_rf.yaml, cnn.yaml, smoke.yaml
data/sample/         56 images committed so the tests and the demo run anywhere
docs/                architecture, evaluation protocol, dataset notes
notebooks/           the original CNN exploration, kept as the experimental record
tests/               66 tests, no network and no dataset required
```

## Requirements

Python 3.10+. Extras are separate on purpose - the API container does not need PyTorch:

```bash
pip install -e ".[vision]"   # MediaPipe + OpenCV: landmark backend, webcam
pip install -e ".[cnn]"      # PyTorch + torchvision: CNN backend, Grad-CAM
pip install -e ".[api]"      # FastAPI + uvicorn: the service
pip install -e ".[data]"     # kagglehub: dataset download
pip install -e ".[dev]"      # pytest + ruff
```

MediaPipe removed the Solutions API after 0.10.14, so the `vision` extra pins
`mediapipe>=0.10.9,<0.10.15`. The landmark backend raises a message saying so if it finds
a newer build.

## Limitations

- Trained on one dataset recorded by one person in one room. Expect a drop on another hand,
  another skin tone, another background - this is a pipeline demonstration, not a product.
- `J` and `Z` are motion letters in ASL; every frame-based model here, including this one,
  treats them as static poses.
- The landmark backend inherits MediaPipe's failure modes (see above) and is limited to one
  hand.
- The blocked split is a proxy for a session holdout, not a true one: the dataset ships no
  session metadata. A genuinely unseen-signer evaluation needs a second dataset.

## License

MIT - see [LICENSE](LICENSE). The dataset is distributed by its own authors under its own
terms and is not redistributed here; see [data/sample/README.md](data/sample/README.md) for
the 56 images used as fixtures.
