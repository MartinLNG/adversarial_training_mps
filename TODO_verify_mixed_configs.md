# TODO: Verify mixed NLL alpha-sweep configs

Run these smoke tests on a machine with sufficient CPU/GPU (conda env: `tn_adv_train`).
Each multirun fires 64 jobs (8 lr × 8 alpha); limit to 2 epochs here to just check config validity.

## 2D toy — legendre/d10D6

```bash
# moons
python -m experiments.generative --multirun \
  +experiments=generative/legendre/d10D6/grid_sweep_mixed/moons_4k \
  trainer.generative.max_epoch=2 trainer.generative.patience=5 \
  tracking.mode=disabled

# circles
python -m experiments.generative --multirun \
  +experiments=generative/legendre/d10D6/grid_sweep_mixed/circles_4k \
  trainer.generative.max_epoch=2 trainer.generative.patience=5 \
  tracking.mode=disabled

# spirals
python -m experiments.generative --multirun \
  +experiments=generative/legendre/d10D6/grid_sweep_mixed/spirals_4k \
  trainer.generative.max_epoch=2 trainer.generative.patience=5 \
  tracking.mode=disabled
```

## Time-series — legendre/d3D10c64

```bash
# ecg200
python -m experiments.generative --multirun \
  +experiments=generative/legendre/d3D10c64/grid_sweep_mixed/ecg200 \
  trainer.generative.max_epoch=2 trainer.generative.patience=5 \
  tracking.mode=disabled

# italypowerdemand
python -m experiments.generative --multirun \
  +experiments=generative/legendre/d3D10c64/grid_sweep_mixed/italypowerdemand \
  trainer.generative.max_epoch=2 trainer.generative.patience=5 \
  tracking.mode=disabled

# syntheticcontrol
python -m experiments.generative --multirun \
  +experiments=generative/legendre/d3D10c64/grid_sweep_mixed/syntheticcontrol \
  trainer.generative.max_epoch=2 trainer.generative.patience=5 \
  tracking.mode=disabled

# chlorineconcentration
python -m experiments.generative --multirun \
  +experiments=generative/legendre/d3D10c64/grid_sweep_mixed/chlorineconcentration \
  trainer.generative.max_epoch=2 trainer.generative.patience=5 \
  tracking.mode=disabled
```

## What to check

- Each multirun completes without error across all 64 jobs.
- W&B (or local logs) show `genloss`, `acc`, and `clsloss` all logged.
- Output dirs follow `outputs/grid_sweep/gen/legendre/{arch}/{dataset}_DDMM/`.
- `alpha=0.0` runs behave like pure classification NLL (clsloss ≈ genloss expected).
- `alpha=1.0` runs behave like pure generative NLL.

## After grid sweeps complete

For each (arch, dataset, alpha), find the run with the best `gen/valid/genloss`:
- In W&B: filter by `config.trainer.generative.criterion.kwargs.alpha == X`,
  sort by `summary.gen/valid/genloss` ascending, take the top run's lr.
- Fill `lr: ???` and `alpha: ???` in the corresponding seed_sweep_mixed config.
- Then run the seed sweep:

```bash
python -m experiments.generative --multirun \
  +experiments=generative/legendre/d10D6/seed_sweep_mixed/moons_4k \
  tracking.mode=online
```

(Repeat for each dataset and arch.)
