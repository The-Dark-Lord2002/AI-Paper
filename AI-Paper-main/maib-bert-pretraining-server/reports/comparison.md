# Original BERT vs domain-adapted BERT (test set, 865 reports)

## Results (mean ± standard deviation over seeds)

| configuration | encoder | seeds | macro-F1 | accuracy | macro-P | macro-R | critical recall | F1 Fire | F1 Capsizing | F1 Flooding |
|---|---|---|---|---|---|---|---|---|---|---|
| bert + ce | base | 3 | 0.901 ± 0.008 | 0.929 ± 0.005 | 0.907 ± 0.011 | 0.898 ± 0.006 | 0.867 ± 0.011 | 0.962 ± 0.006 | 0.843 ± 0.024 | 0.826 ± 0.027 |
| bert + ce | bert-maib-external | 3 | 0.901 ± 0.003 | 0.929 ± 0.001 | 0.916 ± 0.011 | 0.893 ± 0.004 | 0.846 ± 0.011 | 0.962 ± 0.006 | 0.805 ± 0.039 | 0.855 ± 0.031 |
| bert + ce | bert-maib-task | 3 | 0.916 ± 0.003 | 0.939 ± 0.001 | 0.928 ± 0.006 | 0.908 ± 0.002 | 0.867 ± 0.003 | 0.973 ± 0.006 | 0.836 ± 0.018 | 0.876 ± 0.018 |
| bert_bilstm_att + wce | base | 3 | 0.910 ± 0.009 | 0.931 ± 0.007 | 0.920 ± 0.011 | 0.905 ± 0.006 | 0.870 ± 0.012 | 0.969 ± 0.000 | 0.864 ± 0.022 | 0.882 ± 0.008 |
| bert_bilstm_att + wce | bert-maib-external | 3 | 0.905 ± 0.011 | 0.930 ± 0.005 | 0.911 ± 0.020 | 0.905 ± 0.004 | 0.868 ± 0.011 | 0.963 ± 0.011 | 0.840 ± 0.052 | 0.865 ± 0.048 |
| bert_bilstm_att + wce | bert-maib-task | 3 | 0.911 ± 0.004 | 0.933 ± 0.004 | 0.921 ± 0.005 | 0.907 ± 0.011 | 0.873 ± 0.022 | 0.963 ± 0.011 | 0.860 ± 0.025 | 0.876 ± 0.038 |

## Significance: adapted encoder (B) against the original BERT (A)

Paired bootstrap of the seed-averaged macro-F1 difference (2000 resamples); exact McNemar test per seed.

| configuration | encoder (B) vs base (A) | seeds | delta macro-F1 | 95% CI | bootstrap p | McNemar p per seed |
|---|---|---|---|---|---|---|
| bert + ce | bert-maib-external | 3 | -0.0000 | [-0.0181, +0.0149] | 0.951 | 0.458, 0.845, 0.736 |
| bert_bilstm_att + wce | bert-maib-external | 3 | -0.0047 | [-0.0155, +0.0051] | 0.355 | 0.804, 0.503, 0.864 |
| bert + ce | bert-maib-task | 3 | +0.0143 | [+0.0007, +0.0277] | 0.042 | 0.024, 0.210, 0.473 |
| bert_bilstm_att + wce | bert-maib-task | 3 | +0.0014 | [-0.0084, +0.0105] | 0.76 | 0.648, 0.523, 0.043 |

## Stage 1: MLM perplexity on the validation reports (lower = BERT knows the language better)

converged = early stopping ended the run; NO = the loss was still falling at the last allowed epoch.
Perplexities of different corpora are not comparable as a measure of quality: TAPT is trained on texts
like the validation reports, so its perplexity is lower without the encoder being better for classification.

| encoder | corpus texts | corpus words | starts from | best epoch | epochs run | converged | perplexity before | perplexity after |
|---|---|---|---|---|---|---|---|---|
| bert-maib-external | 69395 | 9231654 | bert-base-uncased | 3 | 3 | NO | 8.21 | 4.56 |
| bert-maib-task | 4033 | 268908 | bert-base-uncased | 14 | 17 | yes | 8.21 | 3.16 |
