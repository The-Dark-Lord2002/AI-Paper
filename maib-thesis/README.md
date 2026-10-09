# دسته‌بندی گزارش‌های حوادث دریایی با مدل‌های زبانی

کد پایان‌نامه، دقیقاً مطابق بخش «روش تحقیق» پروپوزال (`../Proposal.pdf`).

## ساختار: هر گام پروپوزال یک فایل

| فایل | گام پروپوزال | کار |
|---|---|---|
| `config.py` | — | **همه‌ی تنظیمات در یک جا** (مدل، learning rate، seedها، فهرست آزمایش‌ها) |
| `step1_prepare_data.py` | گام ۱ | بارگذاری، پاک‌سازی، توزیع کلاس‌ها، تقسیم ۷۰/۱۵/۱۵ |
| `step2_tfidf_svm.py` | گام ۲-الف | خط مبنای کلاسیک TF-IDF + SVM |
| `models.py` | گام‌های ۲-ب، ۲-ج و ۳ | سه مدل: `bert`، `bert_bilstm` (مقاله‌ی مرجع)، `bert_bilstm_att` (روش پیشنهادی) |
| `losses.py` | گام ۴ | چهار تابع زیان: `ce`، `wce`، `focal`، `cb` |
| `step3_train.py` | — | آموزش و ارزیابی **یک** پیکربندی |
| `step4_run_experiments.py` | گام‌های ۲ تا ۴ | اجرای کل جدول آزمایش‌ها (۱۸ اجرا) |
| `step5_report.py` | گام ۵ | جدول نتایج، recall هر کلاس، ماتریس درهم‌ریختگی |
| `step6_attention.py` | گام ۵ | تحلیل کیفی وزن‌های توجه |
| `common.py` | — | توابع کمکی: خواندن داده، محاسبه‌ی معیارها |
| `MAIB_Pipeline.ipynb` | — | اجرای همه‌ی گام‌ها به ترتیب (روی لپ‌تاپ یا Kaggle) |

**ترتیب خواندن کد برای فهمیدن پروژه:** `config.py` ← `step1` ← `step2` ← `models.py` ← `losses.py` ← `step3` ← `step4` ← `step5` ← `step6`

## جدول آزمایش‌ها (`config.EXPERIMENTS`)

| مدل | زیان | گام | سؤال |
|---|---|---|---|
| `bert` | `ce` | ۲-ب | BERT با طبقه‌بند ساده |
| `bert_bilstm` | `ce` | ۲-ج | بازتولید مقاله‌ی مرجع |
| `bert_bilstm_att` | `ce` | ۳ | **RQ1:** آیا توجه کمک می‌کند؟ |
| `bert_bilstm_att` | `wce` | ۴-الف | **RQ2:** وزن معکوس فراوانی |
| `bert_bilstm_att` | `focal` | ۴-ب | **RQ2:** زیان کانونی |
| `bert_bilstm_att` | `cb` | ۴-ج | **RQ2:** تعداد مؤثر نمونه |

هر ردیف با ۳ seed اجرا می‌شود، پس ۱۸ اجرا داریم. به‌علاوه‌ی TF-IDF + SVM (گام ۲-الف).

## اجرا

### روی Kaggle (پیشنهادی، حدود ۳ تا ۴ ساعت)
1. kaggle.com → Code → New Notebook → File → Import Notebook → فایل `MAIB_Pipeline.ipynb` را آپلود کن.
2. پنل سمت راست: Accelerator = **GPU T4**، Internet = **On** (برای GPU باید شماره‌ی تلفن تأیید شده باشد).
3. **Save Version → Save & Run All**. اجرا در پس‌زمینه انجام می‌شود و می‌توانی مرورگر را ببندی.
4. پایان کار: تب Output → `outputs.zip` را دانلود کن و داخل همین پوشه باز کن.

### روی لپ‌تاپ
```bash
pip install -r requirements.txt
python step1_prepare_data.py
python step2_tfidf_svm.py
python step4_run_experiments.py          # حدود ۲۰ ساعت روی GTX 1650 Ti
python step5_report.py
python step6_attention.py
```
اگر اجرای `step4` قطع شد، همان دستور را دوباره بزن تا از جایی که مانده بود ادامه پیدا کند.

## خروجی‌ها

| فایل | محتوا |
|---|---|
| `reports/data_summary.json`، `class_distribution.png` | آمار دادگان (فصل ۳ پایان‌نامه) |
| `reports/main_table.md` | **جدول اصلی** و جواب RQ1 و RQ2 |
| `reports/per_class_recall.csv` / `.png` | recall هر کلاس، نادرترین کلاس اول |
| `reports/confusion_matrix.png` | ماتریس درهم‌ریختگی بهترین مدل |
| `reports/attention_examples.html`، `attention_top_words.csv` | تحلیل وزن‌های توجه |
| `results/*.json` | نتیجه‌ی کامل هر اجرا (تنظیمات، تاریخچه‌ی آموزش، پیش‌بینی‌ها) |

## تصمیم‌های روش‌شناختی (برای فصل ۳)

1. **۹ کلاس به‌جای ۱۱:** دو کلاس `Hull Failure` (۳ گزارش) و `Non-accidental Event` (۲ گزارش) حذف شدند. با این تعداد، مجموعه‌ی آزمون فقط ۰ یا ۱ نمونه از آن‌ها داشت و یک پیش‌بینی به‌تنهایی Macro-F1 را چند واحد جابه‌جا می‌کرد. آستانه در `config.MIN_REPORTS_PER_CLASS` قابل تغییر است.
2. **یک تقسیم ثابت برای همه‌ی مدل‌ها** (`SPLIT_SEED = 42`). seed آموزش (۱، ۲، ۳) جدا از seed تقسیم است.
3. **انتخاب بهترین epoch فقط با validation.** مجموعه‌ی آزمون فقط یک بار، در پایان هر اجرا، استفاده می‌شود.
4. **وزن‌های کلاس فقط از داده‌ی آموزش** محاسبه می‌شوند.
5. **دستور آموزش یکسان برای همه‌ی مدل‌ها:** learning rate برابر 2e-5 برای BERT و 1e-3 برای لایه‌های جدید، batch مؤثر ۳۲ (مثل مقاله‌ی مرجع)، دقت مختلط و انباشت گرادیان. با این کار تفاوت نتایج فقط از معماری یا تابع زیان می‌آید.
   - مقاله‌ی مرجع learning rate را 1e-6 گزارش کرده است. این تنظیم برای همگرایی ده‌ها epoch لازم دارد. برای بازتولید دقیق آن، دستور اختیاری زیر را اجرا کن. نتیجه در جدول با برچسب `[paper_lr]` می‌آید:
     ```bash
     python step4_run_experiments.py --paper-settings
     ```
6. **معیار اصلی:** Macro-F1 و recall هر کلاس، به‌ویژه سه کلاس ایمنی‌بحرانی (`config.CRITICAL_CLASSES`). Accuracy فقط برای مقایسه با ادبیات گزارش می‌شود.
7. **میانگین ± انحراف معیار روی ۳ seed.** `step5_report.py` هر تفاوت را با انحراف معیار seedها مقایسه می‌کند و می‌گوید تفاوت بزرگ‌تر از نوسان تصادفی است یا نه.
