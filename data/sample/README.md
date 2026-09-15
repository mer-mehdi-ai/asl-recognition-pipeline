# Sample fixture

56 images (2 per class, 28 classes) taken from the **ASL Alphabet** dataset by *grassknoted*:

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

They are included so that `make smoke`, the tests and the demo run on a fresh clone without
a Kaggle account. They are a fixture, not a dataset - too small to train or evaluate
anything meaningful. The full dataset is downloaded with `asl data pull` and is never
committed to this repository. Credit and terms for the images belong to the dataset authors.

Rebuild from a full copy:

```bash
asl data sample --per-class 2
```
