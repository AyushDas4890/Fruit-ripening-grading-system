Pooled 5-fold grouped CV, mean ± std over 3 seeds, n = 924 crops

| Model | MAE ↓ | Accuracy ↑ | QWK ↑ | Severe errors (≥2 grades) ↓ |
|---|---|---|---|---|
| Small CNN (scratch, softmax) | 0.422 ± 0.019 | 66.1 ± 2.0% | 0.811 ± 0.005 | 7.40 ± 0.31% |
| ResNet18 transfer + softmax | 0.228 ± 0.045 | 79.0 ± 3.8% | 0.910 ± 0.022 | 1.37 ± 0.50% |
| ResNet18 transfer + ordinal (CORAL) | 0.214 ± 0.030 | 79.5 ± 2.5% | 0.924 ± 0.013 | 0.79 ± 0.38% |

Per fruit (mean over seeds)

|                              |   mae |   acc |   qwk |   severe |
|:-----------------------------|------:|------:|------:|---------:|
| ('avocado', 'cnn')           | 0.367 | 0.689 | 0.856 |    0.053 |
| ('avocado', 'tl_ordinal')    | 0.173 | 0.83  | 0.945 |    0.003 |
| ('avocado', 'tl_softmax')    | 0.178 | 0.829 | 0.94  |    0.007 |
| ('strawberry', 'cnn')        | 0.513 | 0.615 | 0.698 |    0.109 |
| ('strawberry', 'tl_ordinal') | 0.281 | 0.739 | 0.876 |    0.016 |
| ('strawberry', 'tl_softmax') | 0.309 | 0.726 | 0.84  |    0.025 |
