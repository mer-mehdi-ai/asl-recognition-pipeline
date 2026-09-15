# Dataset

[ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) by *grassknoted*.

- 87 000 training images, 200x200 RGB JPEG
- 29 classes: `A`-`Z`, `space`, `del`, `nothing`, 3 000 images each
- one signer, one room, captured as continuous bursts (see [evaluation.md](evaluation.md))
- a separate 28-image test folder, one image per class - too small to evaluate on, useful
  as a sanity check

## Getting it

```bash
asl data pull            # kagglehub, needs Kaggle credentials
```

Credentials come from `~/.kaggle/kaggle.json` or the `KAGGLE_USERNAME` / `KAGGLE_KEY`
environment variables. The images are never committed: `data/raw/`, `data/interim/` and
`runs/` are all ignored by git.

Point `data.root` at the folder holding the class sub-directories:

```yaml
data:
  root: data/raw/asl_alphabet_train/asl_alphabet_train
```

## Known quirks

- **Nested folder.** The archive contains `asl_alphabet_train/asl_alphabet_train/<class>/`.
  Not a mistake in the config.
- **Incomplete copies.** Copies circulating outside Kaggle are often missing `nothing`.
  Nothing in the code assumes 29 classes; the class list is derived from the directories
  present and stored with the model.
- **`J` and `Z` are movements.** In ASL these two letters are traced in the air. The dataset
  stores single frames, so every model here treats them as static poses - an upper bound
  the data itself imposes.
- **Near-duplicate frames.** The reason the default split is `blocked`.

## The committed sample

`data/sample/` holds 56 images (2 per class, 28 classes) so that `make smoke`, the demo and
CI run on a fresh clone with no Kaggle account. It is a fixture, not a dataset: far too
small to train or evaluate anything. See `data/sample/README.md` for attribution.
