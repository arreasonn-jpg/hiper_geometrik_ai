# Tekrarlanabilir Deney Sözleşmesi

HGA deneyleri `hga.evaluation.experiment` üzerinden atomik bir koşu kimliği
alabilir. Varsayılan çalışma dizini:

```text
experiments/
└── EXP-0001/
    ├── config.yaml
    ├── manifest.json
    ├── results.json
    ├── stdout.log
    └── model_hash.txt
```

`config.yaml`, YAML 1.2 ile uyumlu kanonik JSON alt-kümesi olarak yazılır.
`manifest.json` en az şu alanları içerir:

```json
{
  "experiment_id": "EXP-0001",
  "git_commit": "...",
  "git_dirty": false,
  "seed": 42,
  "dataset_hash": "...",
  "config_hash": "...",
  "model_hash": "...",
  "python_version": "...",
  "torch_version": "...",
  "device": "cpu",
  "parameters": {},
  "result": "COMPLETED"
}
```

Başarısız koşu da kaybolmaz: manifest `FAILED` olur ve hata türü ile mesajı
`results.json` içine yazılır. `EXP-NNNN` dizini `mkdir` ile atomik alınır;
paralel süreç aynı kimliği alamaz.

## Golden benchmark — beş seed

```bash
python -m hga golden-benchmark \
  --seeds 1,2,3,4,5 \
  --experiment-root experiments \
  --out raporlar/golden_seed_summary.json
```

Golden v1 deterministik olduğundan farklı seed'lerde özdeş sonuç beklenir.
Buradaki `mean ± std`, runner ve veri izolasyonu için bir regresyon kontrolüdür;
eğitilmiş stokastik bir modelin istatistiksel performans kanıtı değildir.

## Python API

```python
from hga.evaluation import run_seed_sweep

summary = run_seed_sweep(
    callback=lambda seed: {"metrics": train_and_evaluate(seed)},
    seeds=[1, 2, 3, 4, 5],
    root="experiments",
    config={"model": "example"},
    dataset_hash="sha256...",
    parameters={"epochs": 10},
    model_path="checkpoint.pt",
)
```

Callback'in döndürdüğü iç içe sayısal metrikler için population standard
deviation (`pstdev`) hesaplanır. Train/test leakage denetimi deney callback'i
çalışmadan önce yapılmalı; golden runner bunu otomatik uygular.
