# Significance tests (test set, per seed)

McNemar: exact two-sided test on per-report correctness. CI: 95% paired bootstrap interval of the macro-F1 difference (B - A).

| A | B | seed | A_right_B_wrong | A_wrong_B_right | mcnemar_p | delta_macro_f1 | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|
| tfidf_svm | bert__ce | 1 | 20 | 28 | 0.3123 | 0.008 | -0.0177 | 0.0343 |
| tfidf_svm | bert__ce | 2 | 19 | 33 | 0.0704 | 0.018 | -0.008 | 0.0458 |
| bert__ce | bert_bilstm__ce | 1 | 13 | 16 | 0.7111 | 0.012 | -0.0048 | 0.0293 |
| bert__ce | bert_bilstm__ce | 2 | 10 | 8 | 0.8145 | -0.0042 | -0.0274 | 0.0147 |
| bert_bilstm__ce | bert_bilstm_att__ce | 1 | 6 | 11 | 0.3323 | 0.0034 | -0.0151 | 0.0196 |
| bert_bilstm__ce | bert_bilstm_att__ce | 2 | 8 | 9 | 1.0 | 0.0046 | -0.0156 | 0.0251 |
| bert__ce | bert_bilstm_att__ce | 1 | 6 | 14 | 0.1153 | 0.0154 | -0.0038 | 0.0335 |
| bert__ce | bert_bilstm_att__ce | 2 | 12 | 11 | 1.0 | 0.0004 | -0.023 | 0.0204 |
| bert_bilstm_att__ce | bert_bilstm_att__wce | 1 | 5 | 9 | 0.424 | 0.0105 | -0.0042 | 0.0286 |
| bert_bilstm_att__ce | bert_bilstm_att__wce | 2 | 6 | 9 | 0.6072 | 0.0071 | -0.0072 | 0.0251 |
| bert_bilstm_att__ce | bert_bilstm_att__cb | 1 | 4 | 8 | 0.3877 | 0.0027 | -0.0104 | 0.0147 |
| bert_bilstm_att__ce | bert_bilstm_att__cb | 2 | 8 | 10 | 0.8145 | 0.0069 | -0.0076 | 0.0248 |
| bert_bilstm_att__ce | bert_bilstm_att__focal | 1 | 9 | 7 | 0.8036 | -0.002 | -0.0198 | 0.0167 |
| bert_bilstm_att__ce | bert_bilstm_att__focal | 2 | 9 | 9 | 1.0 | 0.0019 | -0.0135 | 0.0201 |
| bert_bilstm_att__ce | bert_bilstm_att__ce__oversample | 1 | 9 | 8 | 1.0 | 0.0016 | -0.01 | 0.0127 |
| bert_bilstm_att__ce | bert_bilstm_att__ce__oversample | 2 | 11 | 11 | 1.0 | 0.0046 | -0.0114 | 0.0246 |
| bert_bilstm_att__ce | bert_bilstm_att__ce__augment | 1 | 11 | 8 | 0.6476 | -0.006 | -0.0206 | 0.0078 |
| bert_bilstm_att__ce | bert_bilstm_att__ce__augment | 2 | 7 | 12 | 0.3593 | 0.0111 | -0.007 | 0.0305 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__augment | 1 | 7 | 6 | 1.0 | -0.0044 | -0.0161 | 0.0064 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__augment | 2 | 5 | 6 | 1.0 | 0.0063 | -0.0042 | 0.0175 |
| bert_bilstm_att__ce | bert_bilstm_att__wce__augment | 1 | 9 | 12 | 0.6636 | 0.0061 | -0.0115 | 0.0261 |
| bert_bilstm_att__ce | bert_bilstm_att__wce__augment | 2 | 8 | 12 | 0.5034 | 0.0133 | -0.004 | 0.0342 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h384_l1 | 1 | 7 | 5 | 0.7744 | -0.0078 | -0.0271 | 0.0056 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h384_l1 | 2 | 20 | 11 | 0.1496 | -0.0035 | -0.0211 | 0.0156 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h128_l2 | 1 | 16 | 9 | 0.2295 | -0.0138 | -0.0293 | -0.0001 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h128_l2 | 2 | 4 | 10 | 0.1796 | 0.0044 | -0.0112 | 0.0222 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h384_l2 | 1 | 12 | 5 | 0.1435 | -0.0061 | -0.0165 | 0.0038 |
| bert_bilstm_att__wce | bert_bilstm_att__wce__h384_l2 | 2 | 6 | 8 | 0.7905 | 0.004 | -0.0043 | 0.0133 |
| tfidf_svm | tfidf_svm_oversample | 0 | 5 | 7 | 0.7744 | 0.0006 | -0.0097 | 0.0101 |
| tfidf_svm | tfidf_svm_augment | 0 | 5 | 6 | 1.0 | 0.0046 | -0.0081 | 0.0224 |
