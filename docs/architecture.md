# Architecture

## Data flow

```
                  configs/*.yaml  (typed, validated, hashed)
                         │
   data/raw/  ──▶  scan_dataset()  ──▶  catalog: path | label | order
                         │
                    make_split()   ──▶  catalog + split column
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 extract_features()                CatalogImageDataset
 MediaPipe, 42 dims                 100x100 RGB tensors
        │                                 │
 RandomForestClassifier              CNN (3 conv blocks)
        └────────────────┬────────────────┘
                         ▼
              evaluation_artifacts()
                         │
      runs/<ts>-<name>-<hash>/  manifest · config · metrics · model · figures
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   asl predict      asl serve        asl explain
   (image/webcam)   (FastAPI)        (Grad-CAM)
```

## Decisions, and what each one fixes

**One config object, validated at load.** `extra="forbid"` means a typo in a YAML key is an
error at startup rather than a silently ignored setting, and `frozen=True` means nothing
mutates the config halfway through a run. The config is hashed into the run id, so two runs
with the same name and different settings can never be confused.
*Fixes: constants scattered across three scripts and a notebook.*

**The dataset becomes an explicit catalog.** `torchvision.datasets.ImageFolder` infers the
class order from the filesystem at training time; nothing records it. Here the file list,
the label and the split are a DataFrame written into every run as `catalog.csv`.
*Fixes: two different hard-coded label maps in the original code - one of which mapped
index 28 to `del` twice, so a prediction of `nothing` was displayed as `del`.*

**Classes are derived from the data, never hard-coded.** The dataset nominally has 29
classes, but copies in the wild are incomplete (the one this was rebuilt from has 28 - no
`nothing`). `class_names(catalog)` is the single source of truth and is persisted with the
model.

**Split policy is a first-class, swappable object.** See [evaluation.md](evaluation.md).
*Fixes: a random split that inflated the reported score.*

**Runs are immutable.** `runs/<UTC ms timestamp>-<name>-<config hash>/`, created with
`mkdir` that fails if the directory exists. A run holds the config, a manifest with the git
revision and the platform, `metrics.json`, the model and the figures.
*Fixes: results that existed only as a number printed in a notebook cell, and model files
overwritten by the next experiment.*

**Both backends implement one `Recognizer` interface** (`predict_image`, `save`, `load`).
The webcam loop, the CLI and the API are written against that interface alone, so swapping
backends is a config change. `registry.load_recognizer` reads the backend name from the run
manifest, so the caller never has to know which one it is serving.

**Torch and MediaPipe are imported lazily.** The landmark backend runs without PyTorch
installed, the CNN runs without MediaPipe, the API container carries neither CUDA nor
torchvision, and the unit tests need none of them.

**The extractor is injected, not imported.** `extract_features(catalog, extractor)` takes
any callable mapping a BGR frame to a feature vector, which is what lets the pipeline test
run the *real* pipeline against a deterministic stub - no native libraries, no fixtures
downloaded, no flakiness.

**Feature normalisation is scale invariant.** The original code subtracted the bounding-box
minimum, which is invariant to where the hand is but not to how far it is from the camera -
a model trained on arm's-length images degrades when the user sits closer. `bbox`
normalisation divides by the longest bounding-box side as well. Both are available
(`landmarks.normalization`) so the choice can be measured.

**The text composer is pure logic.** Committing a letter requires the same prediction for
`hold_seconds` *and* a rolling mean confidence above a threshold, so one blurred frame
cannot type a wrong character. It contains no OpenCV and no Tkinter, which is why it has
unit tests covering the flicker, low-confidence, `space`, `del` and idle cases.
*Fixes: spelling logic tangled with the GUI event loop and a capture thread, which made it
untestable and made `del` on an empty buffer throw.*

**The service loads a run, not a checkpoint.** `ASL_RUN` points at a run directory, so the
model served and the metrics published are the same artifact by construction. The image
carries no model, which is what makes it deployable without a rebuild.

## Test strategy

66 tests, no network, no dataset, no GPU:

| Area | What is actually asserted |
|---|---|
| `test_config` | YAML round-trip, fingerprint stability, rejection of unknown keys and out-of-range values, and that the *shipped* configs still validate. |
| `test_data` | Natural frame ordering, per-class caps, blocked split holds out the tail, splits never share a file, random split is reproducible, normalisation is translation- and scale-invariant, undetected hands are dropped and counted. |
| `test_runs` | Manifest contents, millisecond ids, refusal to overwrite, metric merging, `latest` resolution, foreign directories ignored. |
| `test_models` | The Random Forest learns a separable problem, probabilities are aligned to the canonical class order, save/load round-trips, CNN output shape and that training actually reduces the loss. |
| `test_pipeline` | The real pipeline end to end on generated images with a stub extractor: run directory, artifacts, metrics, feature cache reuse, and loading the run back to predict. |
| `test_realtime` | Dwell time, confidence gate, flicker, `space`/`del`/idle, empty-buffer delete. |
| `test_api` | 503 without a model, 200 with, 422 when no hand is found, 400 on empty and non-image uploads. |
| `test_cli` | Every command group is reachable and `runs list/show` work on real run directories. |

CI additionally runs the pipeline end to end on the committed sample with the real
MediaPipe extractor, and builds the Docker image and waits for `/health` to answer.
