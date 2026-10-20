# Significance tests (test set)

## Per seed

McNemar: exact two-sided test on per-report correctness. CI: 95% paired bootstrap interval of the macro-F1 difference (B - A).

| A | B | seed | A_right_B_wrong | A_wrong_B_right | mcnemar_p | delta_macro_f1 | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|
| tfidf_svm | bert__ce | 1 | 20 | 28 | 0.3123 | 0.008 | -0.0177 | 0.0343 |
| tfidf_svm | bert__ce | 2 | 19 | 33 | 0.0704 | 0.018 | -0.008 | 0.0458 |
| tfidf_svm | bert_bilstm_att__wce | 1 | 13 | 33 | 0.0045 | 0.0339 | 0.0115 | 0.061 |
| tfidf_svm | bert_bilstm_att__wce | 2 | 14 | 30 | 0.0226 | 0.0255 | -0.0002 | 0.0544 |
| tfidf_svm | bert_bilstm_att__wce__augment | 1 | 12 | 31 | 0.0054 | 0.0296 | 0.0074 | 0.0564 |
| tfidf_svm | bert_bilstm_att__wce__augment | 2 | 14 | 31 | 0.0161 | 0.0318 | 0.0056 | 0.0597 |
| bert__ce | bert_bilstm__ce | 1 | 13 | 16 | 0.7111 | 0.012 | -0.0044 | 0.0291 |
| bert__ce | bert_bilstm__ce | 2 | 10 | 8 | 0.8145 | -0.0042 | -0.0256 | 0.0145 |
| bert_bilstm__ce | bert_bilstm_att__ce | 1 | 6 | 11 | 0.3323 | 0.0034 | -0.0142 | 0.0192 |
| bert_bilstm__ce | bert_bilstm_att__ce | 2 | 8 | 9 | 1.0 | 0.0046 | -0.0161 | 0.0256 |
| bert__ce | bert_bilstm_att__ce | 1 | 6 | 14 | 0.1153 | 0.0154 | -0.0028 | 0.0339 |
| bert__ce | bert_bilstm_att__ce | 2 | 12 | 11 | 1.0 | 0.0004 | -0.0235 | 0.0202 |
| bert_bilstm_att__ce | bert_bilstm_att__wce | 1 | 5 | 9 | 0.424 | 0.0105 | -0.0045 | 0.028 |
| bert_bilstm_att__ce | bert_bilstm_att__wce | 2 | 6 | 9 | 0.6072 | 0.0071 | -0.0079 | 0.0249 |
| bert_bilstm_att__ce | bert_bilstm_att__cb | 1 | 4 | 8 | 0.3877 | 0.0027 | -0.0102 | 0.015 |
| bert_bilstm_att__ce | bert_bilstm_att__cb | 2 | 8 | 10 | 0.8145 | 0.0069 | -0.0082 | 0.0262 |
| bert_bilstm_att__ce | bert_bilstm_att__focal | 1 | 9 | 7 | 0.8036 | -0.002 | -0.0199 | 0.0167 |
| bert_bilstm_att__ce | bert_bilstm_att__focal | 2 | 9 | 9 | 1.0 | 0.0019 | -0.0125 | 0.0193 |
| bert_bilstm_att__ce | bert_bilstm_att__ce__oversample | 1 | 9 | 8 | 1.0 | 0.0016 | -0.0097 | 0.0131 |
| bert_bilstm_att__ce | bert_bilstm_att__ce__oversample | 2 | 11 | 11 | 1.0 | 0.0046 | -0.0113 | 0.0234 |
| bert_bilstm_att__ce | bert_bilstm_att__ce__augment | 1 | 11 | 8 | 0.6476 | -0.006 | -0.0205 | 0.0093 |
| bert_bilstm_att__ce | bert_bilstm_att__ce__augment | 2 | 7 | 12 | 0.3593 | 0.0111 | -0.0067 | 0.0313 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__augment | 1 | 7 | 6 | 1.0 | -0.0044 | -0.0165 | 0.0063 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__augment | 2 | 5 | 6 | 1.0 | 0.0063 | -0.0045 | 0.0175 |
| bert_bilstm_att__ce | bert_bilstm_att__wce__augment | 1 | 9 | 12 | 0.6636 | 0.0061 | -0.011 | 0.0248 |
| bert_bilstm_att__ce | bert_bilstm_att__wce__augment | 2 | 8 | 12 | 0.5034 | 0.0133 | -0.0046 | 0.0334 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h384_l1 | 1 | 7 | 5 | 0.7744 | -0.0078 | -0.0255 | 0.0052 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h384_l1 | 2 | 20 | 11 | 0.1496 | -0.0035 | -0.0215 | 0.0174 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h128_l2 | 1 | 16 | 9 | 0.2295 | -0.0138 | -0.03 | 0.0007 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h128_l2 | 2 | 4 | 10 | 0.1796 | 0.0044 | -0.0106 | 0.0234 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h384_l2 | 1 | 12 | 5 | 0.1435 | -0.0061 | -0.0174 | 0.0036 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h384_l2 | 2 | 6 | 8 | 0.7905 | 0.004 | -0.0047 | 0.0132 |
| tfidf_svm | tfidf_svm_oversample | 0 | 5 | 7 | 0.7744 | 0.0006 | -0.0095 | 0.0106 |
| tfidf_svm | tfidf_svm_augment | 0 | 5 | 6 | 1.0 | 0.0046 | -0.0076 | 0.0215 |

## Both seeds together

Difference of the seed-averaged macro-F1 (B - A); 95% paired bootstrap interval (2000 resamples of the test reports, each applied to all seeds) and two-sided bootstrap p-value (0.0 means < 0.0005).

| A | B | seeds | delta_macro_f1 | ci_low | ci_high | bootstrap_p |
|---|---|---|---|---|---|---|
| tfidf_svm | bert__ce | 2 | 0.013 | -0.0111 | 0.0388 | 0.272 |
| tfidf_svm | bert_bilstm_att__wce | 2 | 0.0297 | 0.0068 | 0.0586 | 0.005 |
| tfidf_svm | bert_bilstm_att__wce__augment | 2 | 0.0307 | 0.0091 | 0.0561 | 0.005 |
| bert__ce | bert_bilstm__ce | 2 | 0.0039 | -0.0107 | 0.0184 | 0.575 |
| bert_bilstm__ce | bert_bilstm_att__ce | 2 | 0.004 | -0.0097 | 0.0182 | 0.565 |
| bert__ce | bert_bilstm_att__ce | 2 | 0.0079 | -0.0106 | 0.0246 | 0.376 |
| bert_bilstm_att__ce | bert_bilstm_att__wce | 2 | 0.0088 | -0.0032 | 0.0215 | 0.142 |
| bert_bilstm_att__ce | bert_bilstm_att__cb | 2 | 0.0048 | -0.0072 | 0.0171 | 0.406 |
| bert_bilstm_att__ce | bert_bilstm_att__focal | 2 | -0.0 | -0.0122 | 0.0125 | 0.999 |
| bert_bilstm_att__ce | bert_bilstm_att__ce__oversample | 2 | 0.0031 | -0.0074 | 0.0157 | 0.564 |
| bert_bilstm_att__ce | bert_bilstm_att__ce__augment | 2 | 0.0025 | -0.0095 | 0.0165 | 0.689 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__augment | 2 | 0.001 | -0.007 | 0.0082 | 0.839 |
| bert_bilstm_att__ce | bert_bilstm_att__wce__augment | 2 | 0.0097 | -0.0051 | 0.0262 | 0.176 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h384_l1 | 2 | -0.0057 | -0.0154 | 0.0036 | 0.258 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h128_l2 | 2 | -0.0047 | -0.0163 | 0.0072 | 0.404 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h384_l2 | 2 | -0.001 | -0.0088 | 0.0071 | 0.782 |
| tfidf_svm | tfidf_svm_oversample | 1 | 0.0006 | -0.0086 | 0.01 | 0.86 |
| tfidf_svm | tfidf_svm_augment | 1 | 0.0046 | -0.0076 | 0.0216 | 0.574 |
