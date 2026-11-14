# Original BERT vs domain-adapted BERT (test set, 865 reports)

## Results (mean ± standard deviation over seeds)

| configuration | encoder | seeds | macro-F1 | accuracy | macro-P | macro-R | critical recall | F1 Fire | F1 Capsizing | F1 Flooding |
|---|---|---|---|---|---|---|---|---|---|---|
| bert + ce | base | 2 | 0.898 ± 0.007 | 0.927 ± 0.005 | 0.902 ± 0.009 | 0.895 ± 0.005 | 0.873 ± 0.008 | 0.964 ± 0.007 | 0.857 ± 0.000 | 0.810 ± 0.005 |
| bert + ce | bert-maib-external | 2 | 0.911 ± 0.007 | 0.934 ± 0.001 | 0.917 ± 0.020 | 0.908 ± 0.006 | 0.874 ± 0.007 | 0.964 ± 0.007 | 0.852 ± 0.008 | 0.876 ± 0.047 |
| bert + ce | bert-maib-task | 2 | 0.905 ± 0.002 | 0.935 ± 0.001 | 0.915 ± 0.000 | 0.899 ± 0.002 | 0.843 ± 0.002 | 0.974 ± 0.007 | 0.792 ± 0.032 | 0.839 ± 0.004 |
| bert_bilstm_att + wce | base | 2 | 0.914 ± 0.006 | 0.935 ± 0.003 | 0.925 ± 0.005 | 0.908 ± 0.005 | 0.868 ± 0.016 | 0.969 ± 0.000 | 0.868 ± 0.030 | 0.881 ± 0.011 |
| bert_bilstm_att + wce | bert-maib-external | 2 | 0.917 ± 0.007 | 0.937 ± 0.002 | 0.929 ± 0.004 | 0.909 ± 0.007 | 0.862 ± 0.024 | 0.964 ± 0.007 | 0.868 ± 0.030 | 0.870 ± 0.003 |
| bert_bilstm_att + wce | bert-maib-task | 2 | 0.917 ± 0.001 | 0.934 ± 0.001 | 0.924 ± 0.001 | 0.913 ± 0.001 | 0.885 ± 0.008 | 0.969 ± 0.000 | 0.889 ± 0.000 | 0.883 ± 0.014 |

## Significance: adapted encoder (B) against the original BERT (A)

Paired bootstrap of the seed-averaged macro-F1 difference (2000 resamples); exact McNemar test per seed.

| configuration | encoder (B) vs base (A) | seeds | delta macro-F1 | 95% CI | bootstrap p | McNemar p per seed |
|---|---|---|---|---|---|---|
| bert + ce | bert-maib-external | 2 | +0.0131 | [-0.0006, +0.0270] | 0.063 | 0.108, 0.856 |
| bert_bilstm_att + wce | bert-maib-external | 2 | +0.0023 | [-0.0057, +0.0100] | 0.578 | 1.000, 0.815 |
| bert + ce | bert-maib-task | 2 | +0.0075 | [-0.0103, +0.0220] | 0.37 | 0.150, 0.572 |
| bert_bilstm_att + wce | bert-maib-task | 2 | +0.0023 | [-0.0078, +0.0134] | 0.659 | 0.454, 1.000 |

## Stage 1: MLM perplexity on the validation reports (lower = BERT knows the language better)

| encoder | corpus texts | corpus words | best epoch | perplexity before | perplexity after |
|---|---|---|---|---|---|
| bert-maib-external | 20000 | 2660688 | 4 | 8.21 | 4.91 |
| bert-maib-task | 4033 | 268908 | 9 | 8.21 | 3.22 |
