Critical classes: Fire / Explosion, Capsizing / Listing, Flooding / Foundering

| Configuration | Seeds | Macro-F1 | Critical recall | Accuracy |
|---|---:|---:|---:|---:|
| tfidf_svm | 1 | 0.885 ± 0.000 | 0.827 ± 0.000 | 0.914 ± 0.000 |
| tfidf_svm_oversample | 1 | 0.885 ± 0.000 | 0.834 ± 0.000 | 0.917 ± 0.000 |
| tfidf_svm_augment | 1 | 0.889 ± 0.000 | 0.850 ± 0.000 | 0.916 ± 0.000 |
| bert + ce | 2 | 0.898 ± 0.007 | 0.873 ± 0.008 | 0.927 ± 0.005 |
| bert_bilstm + ce | 2 | 0.901 ± 0.004 | 0.851 ± 0.040 | 0.928 ± 0.001 |
| bert_bilstm_att + ce | 2 | 0.905 ± 0.004 | 0.845 ± 0.016 | 0.931 ± 0.002 |
| bert_bilstm_att + wce | 2 | 0.914 ± 0.006 | 0.868 ± 0.016 | 0.935 ± 0.003 |
| bert_bilstm_att + focal | 2 | 0.905 ± 0.001 | 0.858 ± 0.012 | 0.930 ± 0.001 |
| bert_bilstm_att + cb | 2 | 0.910 ± 0.001 | 0.857 ± 0.000 | 0.935 ± 0.004 |
| bert_bilstm_att + ce [oversample] | 2 | 0.909 ± 0.001 | 0.851 ± 0.008 | 0.931 ± 0.002 |
| bert_bilstm_att + ce [augment] | 2 | 0.908 ± 0.009 | 0.862 ± 0.008 | 0.932 ± 0.004 |
| bert_bilstm_att + wce [augment] | 2 | 0.915 ± 0.002 | 0.868 ± 0.016 | 0.935 ± 0.002 |
| bert_bilstm_att + wce [h384_l1] | 2 | 0.909 ± 0.003 | 0.862 ± 0.007 | 0.929 ± 0.009 |
| bert_bilstm_att + wce [h128_l2] | 2 | 0.910 ± 0.007 | 0.873 ± 0.008 | 0.935 ± 0.007 |
| bert_bilstm_att + wce [h384_l2] | 2 | 0.913 ± 0.001 | 0.868 ± 0.016 | 0.932 ± 0.004 |

Is BERT better than the classic baseline?
- bert + ce vs tfidf_svm: macro-F1 +0.013, critical recall +0.045 (seed std 0.007: larger than seed-to-seed noise)
- bert_bilstm + ce vs bert + ce: macro-F1 +0.004, critical recall -0.022 (seed std 0.007: within seed-to-seed noise)

RQ1 - Does attention over the BiLSTM states help? (proposal step 3)
- bert_bilstm_att + ce vs bert_bilstm + ce: macro-F1 +0.004, critical recall -0.005 (seed std 0.004: within seed-to-seed noise)

RQ2 - Which imbalance treatment helps? (proposal step 4)
- bert_bilstm_att + wce vs bert_bilstm_att + ce: macro-F1 +0.009, critical recall +0.022 (seed std 0.006: larger than seed-to-seed noise)
- bert_bilstm_att + focal vs bert_bilstm_att + ce: macro-F1 -0.000, critical recall +0.013 (seed std 0.004: within seed-to-seed noise)
- bert_bilstm_att + cb vs bert_bilstm_att + ce: macro-F1 +0.005, critical recall +0.011 (seed std 0.004: larger than seed-to-seed noise)

Oversampling the training set (supervisor's request)
- tfidf_svm_oversample vs tfidf_svm: macro-F1 +0.001, critical recall +0.007 (seed std 0.000: single runs, no seed-to-seed noise available)
- bert_bilstm_att + ce [oversample] vs bert_bilstm_att + ce: macro-F1 +0.003, critical recall +0.005 (seed std 0.004: within seed-to-seed noise)
- bert_bilstm_att + ce [oversample] vs bert_bilstm_att + wce: macro-F1 -0.006, critical recall -0.017 (seed std 0.006: within seed-to-seed noise)

Data augmentation with a language model (supervisor's request, step 7)
- tfidf_svm_augment vs tfidf_svm: macro-F1 +0.005, critical recall +0.022 (seed std 0.000: single runs, no seed-to-seed noise available)
- bert_bilstm_att + ce [augment] vs bert_bilstm_att + ce: macro-F1 +0.003, critical recall +0.017 (seed std 0.009: within seed-to-seed noise)
- bert_bilstm_att + ce [augment] vs bert_bilstm_att + ce [oversample]: macro-F1 -0.001, critical recall +0.012 (seed std 0.009: within seed-to-seed noise)
- bert_bilstm_att + wce [augment] vs bert_bilstm_att + wce: macro-F1 +0.001, critical recall +0.000 (seed std 0.006: within seed-to-seed noise)

Bigger BiLSTM (supervisor's question), all with wce; the default is 128 units per direction, 1 layer
- bert_bilstm_att + wce [h384_l1] vs bert_bilstm_att + wce: macro-F1 -0.006, critical recall -0.006 (seed std 0.006: within seed-to-seed noise)
- bert_bilstm_att + wce [h128_l2] vs bert_bilstm_att + wce: macro-F1 -0.005, critical recall +0.005 (seed std 0.007: within seed-to-seed noise)
- bert_bilstm_att + wce [h384_l2] vs bert_bilstm_att + wce: macro-F1 -0.001, critical recall +0.000 (seed std 0.006: within seed-to-seed noise)
