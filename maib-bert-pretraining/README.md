# پیش‌آموزش BERT روی متون دریایی (بدون ناظر) و سپس آموزش با ناظر

این پوشه یک پروژه‌ی **مستقل** است و به کد مقاله (`maib-thesis/`) وابسته نیست. می‌توانید همین پوشه را به‌تنهایی روی سرور دانشگاه بگذارید. کد آموزش با ناظر، مدل‌ها و تابع‌های زیان عیناً از کد مقاله کپی شده‌اند تا نتایج دو پروژه قابل مقایسه باشند.

---

## ۱. این کار چیست؟ (به زبان ساده)

BERT اصلی با ویکی‌پدیا و کتاب‌ها آموزش دیده است و واژه‌های دریایی مثل *allided*، *girting*، *ingress* یا *pilot ladder* را کم دیده است. ایده این است که قبل از طبقه‌بندی، **زبان دریایی را به BERT یاد بدهیم.** این کار دو مرحله دارد:

| مرحله | نام | برچسب لازم است؟ | چه اتفاقی می‌افتد |
|---|---|---|---|
| ۱ | **بدون ناظر** (Masked Language Modeling) | خیر | ۱۵٪ کلمه‌های هر متن دریایی پنهان می‌شوند و BERT باید آن‌ها را حدس بزند |
| ۲ | **با ناظر** (fine-tuning) | بله | همان آموزش مقاله با ۹ کلاس حادثه، ولی شروع از BERT دریایی |

**مثال مرحله‌ی ۱ («ماسک کردن روی داده‌ی متنی»):**

```
متن اصلی:     the vessel ran aground on a sandbank
ورودی BERT:   the vessel ran [MASK] on a sandbank
BERT باید حدس بزند:  aground
```

جواب هر سؤال خود متن است، پس **هیچ برچسبی لازم نیست.** از ۱۵٪ کلمه‌هایی که انتخاب می‌شوند، ۸۰٪ با `[MASK]` جایگزین می‌شوند، ۱۰٪ با یک کلمه‌ی تصادفی، و ۱۰٪ دست‌نخورده می‌مانند (Devlin et al., 2019). کد این کار در تابع `mask_tokens` در `step3_pretrain_mlm.py` است و خط‌به‌خط توضیح داده شده است.

**سؤال پژوهش:** آیا BERTی که زبان دریایی را یاد گرفته، گزارش‌ها، به‌خصوص کلاس‌های نادر و حیاتی، را بهتر از BERT اصلی طبقه‌بندی می‌کند؟

**نام علمی:**
- **DAPT** (Domain-Adaptive Pre-Training): مرحله‌ی ۱ روی یک دیتاست دیگرِ همین حوزه.
- **TAPT** (Task-Adaptive Pre-Training): مرحله‌ی ۱ روی متن‌های آموزشیِ خودِ دیتاست.

هر دو در Gururangan et al. (2020)، *Don't Stop Pretraining*، آمده‌اند و هر دو در این کد پیاده شده‌اند. این همان «domain-adapted encoder» است که در بخش محدودیت‌های مقاله به‌عنوان کار آینده آمده است (Black et al., 2026).

---

## ۲. چرا دیتاست دیگر؟ (نشت داده)

اگر مرحله‌ی ۱ روی همان گزارش‌هایی انجام شود که بعداً روی آن‌ها تست می‌کنیم، BERT متن گزارش‌های تست را **قبل از تست دیده است**، حتی بدون برچسبشان. در این صورت نمره‌ی تست بیش از حد خوش‌بینانه می‌شود.

برای جلوگیری از این مشکل، `step2_build_corpus.py` هر متنی را که با یکی از گزارش‌های دیتاست برچسب‌دار (train، val یا test) یکسان یا تقریباً یکسان باشد حذف می‌کند:
- شباهت با ۸-گرم‌های کلمه سنجیده می‌شود.
- آستانه‌ی حذف ۵۰٪ است.
- تعداد متن‌های حذف‌شده در `reports/corpus_summary.json` ثبت می‌شود.

**TAPT** فقط از متن گزارش‌های **train** استفاده می‌کند، که مجاز است. انتخاب بهترین epoch در مرحله‌ی ۱ با loss روی متن گزارش‌های **val** انجام می‌شود. گزارش‌های test هرگز خوانده نمی‌شوند.

---

## ۳. داده‌ی «دیگر» را از کجا بیاوریم؟

**دیتاست برچسب‌دار ما:** `baker-street/maib-incident-reports-5K` (Munaev, 2025)
- «2025» سال انتشار دیتاست است، **نه سال حوادث.**
- طبق صفحه‌ی دیتاست، متن‌ها از خلاصه‌ی رخدادهای MAIB ساخته شده‌اند: Short Description + Description، با برچسب Main Event L1.
- سال‌های دقیق حوادث در صفحه‌ی دیتاست ذکر نشده است.

**منبع ۱ (پیشنهاد اصلی): MAIB Data Portal**
- آدرس: https://maps.dft.gov.uk/maib-data-portal/
- جدول occurrences را با فرمت CSV دانلود کنید.
- طبق اخبار به‌روزرسانی پورتال، داده‌های 2021 تا 2025 در دسترس است و 2020 در حال اضافه شدن است. سال‌های فعلی را خودتان در پورتال چک کنید.
- بخشی از این سال‌ها احتمالاً همان گزارش‌های دیتاست ما هستند. **مهم نیست:** step 2 آن‌ها را خودکار حذف می‌کند و می‌گوید چند متن باقی ماند.

**منبع ۲ (برای متن بیشتر): نشریه‌ها و گزارش‌های MAIB روی gov.uk**
- MAIB Safety Digest، به‌صورت PDF، از دهه‌ی ۱۹۹۰.
- گزارش‌های کامل تحقیق حوادث، به‌صورت PDF.
- هر دو متن دریایی بسیار مرتبط هستند.

`step0_download_sources.py` منبع ۲ (Safety Digestها) و گزارش‌های تحقیق MAIB تا 2020 را خودکار دانلود می‌کند. فایل‌های دیگر (csv، xlsx، txt یا pdf) را دستی در پوشه‌ی `corpus_sources/` بگذارید. استفاده از داده‌ی MAIB تحت Open Government Licence آزاد است؛ منبع را در پایان‌نامه ذکر کنید.

**هدف:** حداقل حدود 2,000 متن یا 200 هزار کلمه بعد از حذف تکراری‌ها. اگر کمتر شد، PDFهای Safety Digest را اضافه کنید.

---

## ۴. طرح آزمایش

| | |
|---|---|
| Encoderها | `base` (BERT اصلی)، `bert-maib-task` (TAPT)، `bert-maib-external` (DAPT)، `bert-maib-external-then-task` (اول DAPT، بعد TAPT روی همان مدل) |
| پیکربندی‌ها | `bert + ce` و `bert_bilstm_att + wce` (مدل اصلی مقاله) |
| seedها | 1، 2 و 3 (seedهای 1 و 2ِ BERT اصلی همان اجراهای مقاله هستند) |
| تقسیم داده | دقیقاً تقسیم مقاله: 4,033 / 864 / 865؛ step 1 این را چک می‌کند |
| معیارها | Macro-F1، Recall کلاس‌های حیاتی، F1 هر کلاس حیاتی، و perplexity در مرحله‌ی ۱ |
| آزمون آماری | McNemar برای هر seed، و بوت‌استرپ جفتی روی میانگین seedها (مثل مقاله) |

**معیار موفقیت، که قبل از اجرا تعیین شده است:** encoder دریایی در Macro-F1 یا Recall کلاس‌های حیاتی بهبودی بیشتر از نوسان seed داشته باشد، با p < 0.05 در بوت‌استرپ. اگر بهبود معنادار نبود، این هم یک نتیجه‌ی قابل گزارش است.

---

## ۵. اجرا (با uv)

**یک بار، نصب:**
```bash
cd maib-bert-pretraining
uv sync          # محیط .venv و همه‌ی کتابخانه‌ها، از جمله PyTorch با CUDA (حدود ۲ تا ۳ گیگابایت دانلود)
uv run python -c "import torch; print(torch.cuda.is_available())"     # باید True چاپ کند
```

**اگر `uv sync` با خطای timeout متوقف شد** (اینترنت کند یا محدود):
```bash
UV_HTTP_TIMEOUT=300 uv sync                       # ۱) همان کار، با صبر بیشتر
uv sync --default-index <آدرس-میرور>/simple        # ۲) از یک میرور PyPI داخلی
```
آدرس دقیق میرور را از صفحه‌ی میرورهای Runflare یا Liara بردارید (فهرست میرورهای ایران: https://github.com/MiravaOrg/Mirava).
برای اینکه همیشه از میرور استفاده شود، در فایل `~/.config/uv/uv.toml` بنویسید:
```toml
[[index]]
url = "<آدرس-میرور>/simple"
default = true
```
روی لینوکس PyTorch از همان PyPI (یا میرور) نصب می‌شود و CUDA را دارد؛ روی ویندوز از سرور PyTorch.

**یا با Miniconda** (همه‌ی بسته‌ها از conda-forge، بدون PyPI؛ conda خودش نسخه‌ی CUDA سازگار با درایور را انتخاب می‌کند):
```bash
conda config --set remote_read_timeout_secs 600   # یک بار: صبر بیشتر روی اینترنت کند
conda config --set remote_max_retries 10
conda env create -f environment.yml               # یک بار، حدود ۳ گیگابایت
conda activate maib-bert-pretraining              # در هر ترمینال جدید
python -c "import torch; print(torch.cuda.is_available())"   # باید True باشد
```
با conda همه‌ی دستورهای پایین را **بدون `uv run`** اجرا کنید، مثلاً `python run_pipeline.py`.

**کل آزمایش با یک دستور:**
```bash
uv run python run_pipeline.py
```
این دستور به ترتیب این کارها را انجام می‌دهد. **اگر وسط کار قطع شد** (قطع برق، Ctrl+C، خواب لپ‌تاپ)، همان دستور را دوباره بزنید: اجراهای تمام‌شده رد می‌شوند و اجرای نیمه‌کاره از **آخرین epoch تمام‌شده** ادامه پیدا می‌کند، با همان نتیجه‌ای که بدون قطع شدن می‌گرفت. وضعیت هر اجرا بعد از هر epoch در یک فایل `*.resume.pt` (حدود ۲ گیگابایت، در `models/` یا `results/`) ذخیره و در پایان اجرا پاک می‌شود؛ پس حدود ۲ گیگابایت جای خالی لازم است. اگر تنظیمات را عوض کنید، کد ادامه نمی‌دهد و می‌گوید فایل را پاک کنید.

| ترتیب | اسکریپت | کار |
|---|---|---|
| ۱ | `step0` | دانلود گزارش‌ها و Safety Digestهای MAIB تا 2020 (فقط اگر `corpus_sources/` خالی باشد) |
| ۲ | `step1` | دیتاست برچسب‌دار شما با تقسیم مقاله؛ اگر test با مقاله یکی نباشد، **متوقف می‌شود** |
| ۳ | `step2` | ساخت corpus: تمیز کردن PDFها (سرصفحه، شماره‌ی صفحه، فهرست، آدرس وب) و حذف هر متنی که در دیتاست شما هست |
| ۴ | `step3` | مرحله‌ی ۱، خودنظارتی (ماسک کردن): یک بار روی train شما (TAPT) و یک بار روی دیتاست دیگر (DAPT) |
| ۵ | `step5` | مرحله‌ی ۲، با ناظر: fine-tuning روی train شما و تست روی test شما، برای BERT اصلی و هر دو BERT دریایی |
| ۶ | `step6` | جدول مقایسه و آزمون‌های معناداری → `reports/comparison.md` |

**گزینه‌ها:**
```bash
uv run python run_pipeline.py --corpora task                # فقط با دیتاست خودتان (بدون دانلود)
uv run python run_pipeline.py --since 2010 --max-reports 100   # دانلود کمتر برای شروع
uv run python run_pipeline.py --max-texts 20000 --mlm-epochs 3 # مرحله‌ی ۱ کوتاه و ناقص، فقط برای آزمایش سریع
uv run python run_pipeline.py --skip-base                   # BERT اصلی از نتایج مقاله (پایین را ببینید)
```

**اجرای تک‌تک مراحل** هم ممکن است، مثلاً:
```bash
uv run python step0_download_sources.py --dry-run           # فقط فهرست فایل‌ها، بدون دانلود
uv run python step3_pretrain_mlm.py --corpus task
uv run python step4_finetune.py --model bert_bilstm_att --loss wce --seed 1 --encoder models/bert-maib-external
```

بدون uv: `pip install -r requirements.txt` و بعد همان دستورها بدون `uv run`.

**زمان تقریبی روی GTX 1650 Ti (4 GB):**
- step 3: حدود ۴ دقیقه برای هر 1,000 متن در هر epoch. مثلاً 5,000 متن × 10 epoch حدود ۳ ساعت، ولی توقف زودهنگام معمولاً زودتر تمامش می‌کند.
- step 5: حدود ۷۵ دقیقه برای هر اجرا. ۳ encoder × ۲ پیکربندی × ۲ seed = ۱۲ اجرا، یعنی حدود ۱۵ ساعت.

**صرفه‌جویی در زمان:** چهار نتیجه‌ی BERT اصلی را از مقاله کپی کنید تا آن‌ها دوباره اجرا نشوند. کد آموزش و تقسیم داده یکسان است، پس این کار مجاز است، به شرطی که step 1 پیام `[ok]` داده باشد:
- `maib-thesis/results/bert__ce__seed1.json` و `bert__ce__seed2.json`
- `maib-thesis/results/bert_bilstm_att__wce__seed1.json` و `bert_bilstm_att__wce__seed2.json`

این چهار فایل را در `results/` همین پوشه کپی کنید و بعد این را اجرا کنید:
```bash
uv run python run_pipeline.py --skip-base
```
اسکریپت `step6_compare.py` فایل‌های بدون فیلد encoder را «base» حساب می‌کند. اگر فایلی روی test دیگری ساخته شده باشد، خطا می‌دهد.

---

## ۵-الف. اجرای اصولی (کامل) — دستور اصلی

**اصل:** تعداد epoch مرحله‌ی ۱ را دستی انتخاب نمی‌کنیم. آموزش آن‌قدر ادامه پیدا می‌کند که loss روی متن گزارش‌های val دیگر بهتر نشود (early stopping). بهترین epoch نگه داشته می‌شود. مرحله‌ی ۲ برای همه‌ی encoderها دقیقاً همان تنظیمات مقاله است. فقط encoder عوض می‌شود.

| encoder | متن مرحله‌ی ۱ | شروع از | سقف epoch | patience | زمان تقریبی روی GTX 1650 Ti |
|---|---|---|---|---|---|
| `bert-maib-task` (TAPT) | train دیتاست شما (۴,۰۳۳ متن) | BERT اصلی | 50 | 3 | حدود ۲۵ دقیقه برای هر epoch |
| `bert-maib-external` (DAPT) | **همه‌ی** متن‌های MAIB قدیمی (حدود ۶۹ هزار) | BERT اصلی | 8 | 2 | حدود ۶ ساعت برای هر epoch |
| `bert-maib-external-then-task` | train دیتاست شما | مدل DAPT | 50 | 3 | حدود ۲۵ دقیقه برای هر epoch |

- سقف epoch فقط یک حد بالاست. اگر اجرایی به سقف رسید و هنوز بهتر می‌شد، کد هشدار **NOT CONVERGED** می‌دهد. در جدول `comparison.md` هم ستون `converged` مقدار `NO` می‌گیرد. در این صورت سقف را در `config.MLM_EPOCHS` بالاتر ببرید.
- این تنظیمات در `config.py` هستند: `MLM_EPOCHS`، `MLM_PATIENCE` و `SEEDS`.

**قدم ۱، فقط یک بار: نتایج اجرای سبک قبلی را کنار بگذارید.** هیچ چیزی پاک نمی‌شود:
```bash
python archive_previous_run.py light          # فقط نشان می‌دهد چه چیزهایی منتقل می‌شوند
python archive_previous_run.py light --yes    # منتقل می‌کند به archive/light/
```
بدون این قدم، pipeline مدل‌ها و نتیجه‌های قبلی را می‌بیند و آن‌ها را دوباره نمی‌سازد. نتایج BERT اصلی که از مقاله کپی شده‌اند، سر جایشان می‌مانند.

**قدم ۲: اجرا (بدون `--skip-base`، بدون `--max-texts` و بدون `--mlm-epochs`):**
```bash
source .venv/bin/activate
nohup python run_pipeline.py > run_full.log 2>&1 &
tail -f run_full.log          # دیدن پیشرفت؛ با Ctrl+C فقط tail بسته می‌شود، آموزش ادامه دارد
```

**ترتیب اجرا:**
1. BERT اصلی با seed 3 (seedهای 1 و 2 از مقاله کپی شده‌اند).
2. TAPT و بعد fine-tune آن.
3. DAPT و بعد fine-tune آن.
4. DAPT ← TAPT و بعد fine-tune آن.

بعد از هر encoder، جدول `reports/comparison.md` به‌روز می‌شود. پس نتیجه‌ی TAPT را حدود یک روز بعد از شروع می‌بینید، قبل از اینکه DAPT تمام شود.

**زمان کل:** حدود ۳ تا ۴ روز. شامل ۲۰ اجرای fine-tune (هر کدام حدود ۷۵ دقیقه) و مرحله‌ی ۱ برای سه encoder است.

**متوقف کردن:**
```bash
pkill -f run_pipeline; pkill -f step3_pretrain_mlm; pkill -f step5_run_all
```
**ادامه دادن:** همان دستور قدم ۲ را بزنید. اجراهای تمام‌شده رد می‌شوند و اجرای نیمه‌کاره از آخرین epoch تمام‌شده ادامه پیدا می‌کند. اگر corpus تغییری نکرده باشد، ساخت دوباره‌ی آن (step 2) هم رد می‌شود. **تنظیمات را وسط کار عوض نکنید.**

---

**اجرا روی GPU رایگان Kaggle یا Colab:** راهنمای قدم‌به‌قدم در [`cloud/README.md`](cloud/README.md) است.

---

## ۶. ساختار پوشه

| فایل | کار |
|---|---|
| `config.py` | همه‌ی تنظیمات |
| `run_pipeline.py` | **کل آزمایش با یک دستور**، به ترتیب و قابل ادامه بعد از قطع شدن |
| `step0_download_sources.py` | دانلود خودکار گزارش‌ها و Safety Digestهای MAIB تا 2020 از gov.uk |
| `pyproject.toml` | تعریف محیط برای uv |
| `environment.yml` | تعریف محیط برای conda (Miniconda) |
| `step1_prepare_data.py` | دیتاست برچسب‌دار با تقسیم مقاله، و چک یکسان بودن test |
| `step2_build_corpus.py` | ساخت corpus بدون برچسب و حذف هم‌پوشانی با دیتاست برچسب‌دار |
| `step3_pretrain_mlm.py` | مرحله‌ی ۱: Masked Language Modeling |
| `step4_finetune.py` | مرحله‌ی ۲: یک اجرای fine-tuning (کپی کد مقاله) |
| `step5_run_all.py` | همه‌ی اجراهای مرحله‌ی ۲ |
| `step6_compare.py` | جدول مقایسه و آزمون‌های معناداری |
| `cloud/` | notebookهای آماده برای Kaggle و Colab، اسکریپت ساخت zip برای آپلود، و راهنما |
| `archive_previous_run.py` | انتقال نتایج یک اجرای قبلی به `archive/` (بدون پاک کردن) |
| `models.py`، `losses.py`، `common.py` | مدل‌ها، تابع‌های زیان و ابزارها (از کد مقاله) |
| `corpus_sources/` | فایل‌های داده‌ی دیگر (ورودی شما) |
| `results/`، `reports/` | خروجی‌ها؛ commit شوند |
| `data/`، `corpus/`، `models/` | ساخته‌شده توسط کد؛ commit نشوند (`.gitignore`). هر مدل حدود 440 MB است |

---

## ۷. نکته‌های صادقانه

- **اندازه‌ی corpus مهم است.** با چند هزار متن کوتاه، بهبود ممکن است کم باشد. Gururangan et al. بهبود بزرگ‌تر را با داده‌ی بیشتر گزارش کرده‌اند، ولی TAPT حتی با داده‌ی کم هم اغلب کمک کرده است.
- **واژگان BERT عوض نمی‌شود.** کلمه‌ای مثل *allided* همچنان به چند تکه شکسته می‌شود. مرحله‌ی ۱ کاربرد این تکه‌ها را در متن دریایی یاد می‌دهد، نه کلمه‌ی جدید.
- **seed:** با ۲ seed، انحراف معیار تقریباً بی‌معنی است. پیش‌فرض الان ۳ seed است. اگر تفاوت‌ها هنوز نزدیک به مرز معناداری بودند، 4 و 5 را هم به `config.SEEDS` اضافه کنید.
- **هیچ تنظیمی را با test انتخاب نکنید.** انتخاب epoch در هر دو مرحله فقط با val انجام می‌شود.

## منابع
- Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
- Gururangan, S. et al. (2020). Don't stop pretraining: Adapt language models to domains and tasks. *ACL*.
- Munaev, I. (2025). *MAIB Incident Type Dataset.* Hugging Face.
- MAIB Data Portal: https://maps.dft.gov.uk/maib-data-portal/
