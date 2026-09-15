# Evaluation protocol

## The problem with a random split on this dataset

ASL Alphabet contains 3 000 images per class, captured as a continuous burst: the hand does
not move much between `A1.jpg` and `A2.jpg`, the lighting is identical, the background is
identical, the signer is the same person.

A stratified random split therefore does **not** produce an independent test set. It
produces a test set whose every image has a near-twin in the training set. A model can
score extremely well on it by memorising a handful of recording conditions, and the score
says almost nothing about the only thing that matters in practice: what happens on a hand
the model has never seen, in a room it has never seen.

This is the well-known group-leakage failure. The fix is to split by group rather than by
row - here, the group is "the recording session", which the dataset does not label.

## What this repository does instead

Two policies are implemented (`split.strategy` in the config):

| Strategy | How | Use |
|---|---|---|
| `random` | Stratified random split, `sklearn.train_test_split`. | Reproducing the usual published numbers, and measuring the gap. |
| `blocked` | Per class, frames are sorted in natural order (`A1, A2, ... A3000`) and the **last** `test_size` fraction is held out. | Default. The nearest thing to a session holdout the dataset allows. |

`blocked` is not perfect - it is a proxy. If the burst never changed conditions, it is
barely better than random; if conditions drifted over the recording, it is much closer to
an honest estimate. The value of implementing both is that the difference between the two
numbers is itself a measurement: a large gap is evidence of leakage, a small one is
evidence against it.

Measure it:

```bash
asl train --config configs/cnn.yaml                      # blocked (default)
sed 's/strategy: blocked/strategy: random/' configs/cnn.yaml > /tmp/cnn_random.yaml
asl train --config /tmp/cnn_random.yaml
asl runs list
```

## What is reported

Accuracy alone is a poor summary of a 29-class problem with an imbalanced detector in front
of it, so every run writes, for both backends:

- `accuracy`, `macro_f1`, `weighted_f1`, `n_samples`, `n_classes`
- `weakest_classes` - the five lowest per-class F1 scores
- `top_confusions` - the class pairs the model actually mixes up, which is where ASL
  fingerspelling gets interesting (`M`/`N`/`S`/`T` are all a closed fist with the thumb in
  different positions)
- `hand_detection_rate` and `classes_without_detection` for the landmark backend: a class
  MediaPipe never detects silently disappears from the model, and that belongs in the
  metrics rather than in a surprise at demo time
- `artifacts/eval/classification_report.csv`, `confusion_matrix.csv`, `confusion_matrix.png`

## Reproducibility

Every number lives in `runs/<run id>/metrics.json` next to:

- `config.yaml` - the exact configuration, including seeds
- `manifest.json` - git revision, package version, Python version, platform, command line
- `artifacts/catalog.csv` - the exact file list and the split assigned to every image

Run directories are never overwritten: the id carries a millisecond timestamp and a hash of
the config, and `create_run` refuses to write into an existing directory. A published number
can always be traced back to what produced it.

## What would make the evaluation genuinely convincing

Being explicit about what is missing, since none of it is done here:

1. **A second dataset**, recorded by other signers, used as the only test set. This is the
   real test of whether any of these models transfer.
2. **Cross-signer cross-validation**, which needs signer labels this dataset does not have.
3. **A latency and throughput budget** for the real-time path, measured per backend on a
   fixed machine, not just accuracy.
