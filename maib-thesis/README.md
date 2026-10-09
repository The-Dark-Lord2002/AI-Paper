# دسته‌بندی گزارش‌های حوادث دریایی با مدل‌های زبانی

پیاده‌سازی پایان‌نامه: **BERT + BiLSTM + لایه‌ی توجه + توابع هزینه‌ی حساس به عدم توازن**
روی داده‌ی MAIB ([baker-street/maib-incident-reports-5K](https://huggingface.co/datasets/baker-street/maib-incident-reports-5K)) — ۵٬۷۶۸ گزارش، ۱۱ کلاس.

## ساختار

```
maib/data.py          بارگذاری، پاک‌سازی (یکسان‌سازی برچسب، حذف تکراری)، تقسیم ثابت 70/15/15
maib/model.py         BERT + head: cls | mean | attention | bilstm | bilstm_att
maib/losses.py        ce | wce | focal | cb | cb_focal | la
maib/metrics.py       macro-F1 (معیار اصلی)، balanced acc، per-class، confusion matrix
prepare_data.py       مرحله ۱: آماده‌سازی و تحلیل اکتشافی داده
baseline.py           مرحله ۲: TF-IDF + LogReg / SVM
train.py              آموزش یک پیکربندی
run_experiments.py    اجرای جدول آزمایش‌ها (قابل ادامه بعد از قطعی)
aggregate.py          جدول نتایج: میانگین ± انحراف معیار + آزمون معناداری bootstrap
explain_attention.py  تفسیرپذیری: کلمات پروزن در تصمیم مدل
```

## نصب

```bash
pip install -r requirements.txt
# PyTorch با CUDA را از pytorch.org مطابق نسخه‌ی درایور نصب کن
```

## ترتیب اجرا

```bash
# ۱) داده (یک‌بار). اگر HF در دسترس نیست، فایل jsonl را دستی دانلود کن و --data بده
python prepare_data.py

# ۲) مبنای کلاسیک (چند ثانیه روی CPU)
python baseline.py

# ۳) تست سریع pipeline روی GPU (یک اجرا، یک epoch)
python run_experiments.py --preset smoke --batch 8 --grad_accum 2

# ۴) آزمایش اصلی پروپوزال: {cls, bilstm, bilstm_att} × {ce, wce, focal, cb} × 3 seed = 36 اجرا
python run_experiments.py --preset main --batch 8 --grad_accum 2

# ۴-ب) بازتولید با تنظیمات دقیق مقاله‌ی مرجع (lr=1e-6، batch مؤثر ۳۲)
python run_experiments.py --preset paper --batch 8 --grad_accum 4

# ۵) جدول نتایج
python aggregate.py

# ۶) تفسیرپذیری با بهترین پیکربندی
python train.py --head bilstm_att --loss focal --seed 1 --keep_ckpt --batch 8 --grad_accum 2
python explain_attention.py --ckpt checkpoints/bert-base-uncased__bilstm_att__focal__s1.pt
```

## تنظیمات برای GTX 1650 Ti (4GB VRAM)

- `--batch 8 --grad_accum 2` → batch مؤثر ۱۶. اگر خطای `CUDA out of memory` گرفتی: `--batch 4 --grad_accum 4`.
- اگر باز هم کم آمد: `--grad_ckpt` (حافظه‌ی کمتر، حدود ۳۰٪ کندتر).
- `--max_len 192` کافی است؛ متن‌ها کوتاه‌اند. درصد متن‌های بریده‌شده در ابتدای هر اجرا چاپ می‌شود؛ اگر بالای ۵٪ بود، ۲۵۶ کن.
- fp16 خودکار روی GPU فعال است.
- اجراها قابل ادامه‌اند: اگر لپ‌تاپ خاموش شد، همان دستور را دوباره بزن.

## طراحی آزمایش (برای فصل ۴)

| سؤال پژوهشی | مقایسه | preset |
|---|---|---|
| RQ1: آیا توجه روی BiLSTM از حالت پایانی بهتر است؟ | bilstm در برابر bilstm_att | `main` |
| RQ2: کدام loss برای رده‌های نادر بهتر است؟ | ce / wce / focal / cb | `main` |
| بازتولید مرجع | cls / mean / bilstm با lr=1e-6 | `paper` |
| Ablation: آیا BiLSTM لازم است؟ | mean / attention روی BERT | `ablation` |
| تعمیم (اختیاری) | RoBERTa / DeBERTa-v3 | `models` |

قواعد روش‌شناختی که کد رعایت می‌کند:
- یک تقسیم ثابت برای همه‌ی مدل‌ها (`splits/`)؛ seed آموزش جدا از seed تقسیم.
- انتخاب مدل فقط با val (macro-F1)؛ test فقط یک‌بار در پایان هر اجرا.
- وزن‌های loss فقط از توزیع train.
- گزارش میانگین ± انحراف معیار روی ۳ seed و آزمون bootstrap جفت‌شده.

## منابع روش‌ها

- Devlin et al. (2019). BERT. NAACL.
- Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.
- Cui et al. (2019). Class-Balanced Loss Based on Effective Number of Samples. CVPR.
- Menon et al. (2021). Long-tail Learning via Logit Adjustment. ICLR.
- Munaev (2025). MAIB Incident Type Dataset. Hugging Face.
