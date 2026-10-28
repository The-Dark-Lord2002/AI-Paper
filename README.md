# دسته‌بندی گزارش‌های حوادث دریایی با استفاده از مدل‌های زبانی

**Classification of Marine Incident Reports Using Language Models**

پایان‌نامه‌ی کارشناسی ارشد، دانشگاه صنعتی شاهرود.
دانشجو: محمد مهدی عرب · استاد راهنما: دکتر هدی مشایخی · استاد مشاور: دکتر مریم خدابخش

## ساختار مخزن

```
AI-Paper/
├── maib-thesis/                     کد پایان‌نامه (هر گام پروپوزال یک فایل)
│   ├── config.py, step1 … step6, models.py, losses.py, common.py
│   ├── MAIB_Pipeline.ipynb          اجرای همه‌ی گام‌ها در یک نوت‌بوک
│   ├── results/                     نتیجه‌ی هر اجرا (JSON)
│   └── reports/                     جدول‌ها و نمودارها
├── maib-bert-pretraining/           پروژه‌ی مستقل: پیش‌آموزش BERT روی متون دریایی (MLM) و سپس fine-tuning
├── docs/
│   ├── proposal/                    پروپوزال (PDF و Word)
│   └── papers/                      مقاله‌های مرتبط؛ مقاله‌ی مرجع با برچسب REFERENCE
└── presentations/
    ├── 1405-7-19_progress-meeting/  گزارش Word، اسلایدها و راهنمای آمادگی جلسه
    ├── 1405-7-21_results/            نتایج کامل برای استاد راهنما (PDF)
    ├── 1405-7-21_full-report/        مرور کامل کار برای خود دانشجو
    └── _build/                      اسکریپت‌هایی که این فایل‌ها را از نتایج می‌سازند
```

## نتایج فعلی

دادگان عمومی MAIB، ۹ کلاس، ۸۶۵ گزارش آزمون. میانگین ± انحراف معیار روی ۲ seed.

| پیکربندی | Macro-F1 | Recall کلاس‌های بحرانی | Accuracy |
|---|---:|---:|---:|
| TF-IDF + SVM | 0.885 | 0.827 | 0.914 |
| TF-IDF + SVM + oversampling | 0.885 | 0.834 | 0.917 |
| TF-IDF + SVM + افزایش داده با مدل زبانی | 0.889 | 0.850 | 0.916 |
| BERT | 0.898 ± 0.007 | **0.873 ± 0.008** | 0.927 |
| BERT + BiLSTM (مقاله‌ی مرجع) | 0.901 ± 0.004 | 0.851 ± 0.040 | 0.928 |
| BERT + BiLSTM + Attention | 0.905 ± 0.004 | 0.845 ± 0.016 | 0.931 |
| + Attention + wce | 0.914 ± 0.006 | 0.868 ± 0.016 | **0.935** |
| + Attention + focal | 0.905 ± 0.001 | 0.858 ± 0.012 | 0.930 |
| + Attention + cb | 0.910 ± 0.001 | 0.857 ± 0.000 | **0.935** |
| + Attention + oversampling | 0.909 ± 0.001 | 0.851 ± 0.008 | 0.931 |
| + Attention + افزایش داده | 0.908 ± 0.009 | 0.862 ± 0.008 | 0.932 |
| **+ Attention + wce + افزایش داده (بهترین)** | **0.915 ± 0.002** | 0.868 ± 0.016 | **0.935** |
| + Attention + wce، BiLSTM ۳۸۴×۱ | 0.909 ± 0.003 | 0.862 ± 0.007 | 0.929 |
| + Attention + wce، BiLSTM ۱۲۸×۲ | 0.910 ± 0.007 | **0.873 ± 0.008** | **0.935** |
| + Attention + wce، BiLSTM ۳۸۴×۲ | 0.913 ± 0.001 | 0.868 ± 0.016 | 0.932 |

- همه‌ی مدل‌های BERT از خط مبنای کلاسیک بهترند.
- BiLSTM و attention با تنظیمات یکسان بهبودی بیشتر از نوسان بین seedها ندادند (RQ1)؛ BiLSTM بزرگ‌تر (۳۸۴ واحد یا ۲ لایه) هم کمکی نکرد.
- تابع زیان وزن‌دار (`wce`) بیشترین بهبود را داد (RQ2). بهترین Macro-F1 (0.915) از wce همراه با افزایش داده است؛ برتری‌اش بر wce تنها (0.001) در حد نوسان است. عدد پررنگ: بیشترین مقدار هر ستون.
- oversampling روی هیچ‌کدام از دو مدل بهبود روشنی نداد.
- افزایش داده با مدل زبانی (بازنویسی گزارش‌های کمیاب) SVM را کمی بهتر کرد، ولی روی BERT بهبودی بیشتر از نوسان بین seedها نداد ([گزارش کامل](presentations/1405-7-21_full-report/report_fa.md)).

جزئیات کامل در [`maib-thesis/reports/main_table.md`](maib-thesis/reports/main_table.md) و گزارش Word در `presentations/` است.

## اجرا

```bash
cd maib-thesis
pip install -r requirements.txt
python step1_prepare_data.py
```

دستورهای کامل و توضیح هر فایل در [`maib-thesis/README.md`](maib-thesis/README.md) آمده است.
