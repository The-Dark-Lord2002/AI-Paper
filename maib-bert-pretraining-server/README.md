# نسخه‌ی سرور: پیش‌آموزش BERT (MLM) و fine-tune، روی Kaggle، Colab یا یک سرور GPU

این پوشه نسخه‌ی **سرور** پروژه‌ی `../maib-bert-pretraining/` (نسخه‌ی لپ‌تاپ) است. پوشه‌ی لپ‌تاپ دست نخورده است.
- توضیح روش (MLM، DAPT/TAPT، جلوگیری از نشت داده، طرح آزمایش): README پوشه‌ی لپ‌تاپ.
- این فایل: فقط **چطور روی سرور اجرا کنیم**.

## تفاوت با نسخه‌ی لپ‌تاپ

| | لپ‌تاپ | سرور |
|---|---|---|
| batch مرحله‌ی ۱ (MLM) | 8 × 4 (accumulation) | **32 × 1**، چون GPU سرور ۱۶ گیگ حافظه دارد |
| batch مؤثر و بقیه‌ی تنظیمات | 32 | **همان** (32)، پس آموزش معادل است |
| ماسک‌های ارزیابی و perplexity | ثابت | **عیناً همان** |
| fine-tune (مرحله‌ی ۲) | تنظیمات مقاله، 8 × 4 | **عیناً همان**، تا با اجراهای BERT اصلی مقاله مقایسه‌پذیر باشد |
| توقف زودهنگام | هر بهبودی | فقط بهبود حداقل ۰.۰۱ در loss (حدود ۱٪ perplexity): epochهای بی‌اثر روی سرور پولی اجرا نمی‌شوند |
| فایل‌های اضافه | — | `install_server.sh`، `kaggle_notebook.ipynb`، `colab_notebook.ipynb`، `make_upload_zip.sh` |

خود آزمایش هم همان است:
- مرحله‌ی ۱ تا همگرایی (early stopping).
- سه encoder: TAPT، DAPT، و DAPT ← TAPT.
- ۳ seed.
- جدول مقایسه و آزمون‌های معناداری.

---

## کدام سرور؟

| گزینه | هزینه | GPU | زمان کل تقریبی | نکته |
|---|---|---|---|---|
| **Kaggle** | **رایگان** | T4، ۱۶ گیگ | حدود ۲۰ تا ۲۵ ساعت GPU؛ با سهمیه‌ی ۳۰ ساعت در هفته، حدود ۱ هفته | اجرا در پس‌زمینه؛ می‌توانید مرورگر را ببندید. **پیشنهاد اول.** |
| Google Colab | رایگان | T4 | همان | تب باید باز بماند؛ قطع‌شدن زیاد |
| سرور اجاره‌ای (مثل Vast.ai یا RunPod، یا سرویس‌های داخلی) | تقریباً ۰.۲ تا ۰.۵ دلار در ساعت | RTX 3090/4090 یا A10 | حدود ۱۰ تا ۱۵ ساعت، یعنی **چند دلار** | پرداخت از ایران ممکن است سخت باشد؛ سرویس‌های داخلی را مقایسه کنید |
| سرور دانشگاه | رایگان | — | — | (فعلاً در دسترس نیست) |

**هشدار برای کاربران ایران:**
- Kaggle و Colab متعلق به Google هستند و ممکن است IP ایران را مسدود کنند.
- Kaggle برای GPU تأیید شماره‌ی تلفن می‌خواهد و ممکن است شماره‌ی ایران را قبول نکند.
- **اول ثبت‌نام و تأیید را امتحان کنید.**

زمان‌ها تخمینی‌اند. زمان واقعی مرحله‌ی ۱ به این بستگی دارد که early stopping کِی آن را متوقف کند.

---

## قدم ۱ (برای همه‌ی گزینه‌ها): ساخت فایل آپلود، روی لپ‌تاپ

```bash
cd ~/Mohammad/arshad/paper/AI-Paper
git pull origin claude/awesome-newton-43m2x0
cd maib-bert-pretraining-server
bash make_upload_zip.sh            # می‌سازد: ../maib-server.zip
```

این اسکریپت پوشه‌ی لپ‌تاپ را **فقط می‌خواند و چیزی در آن تغییر نمی‌دهد.** این‌ها را از آن برمی‌دارد:

| چه چیزی | از کجا | چرا |
|---|---|---|
| `data/` | پوشه‌ی لپ‌تاپ | تقسیم مقاله |
| `corpus/` | پوشه‌ی لپ‌تاپ | متن‌های تمیزشده؛ سرور دیگر ۱.۵ گیگ PDF دانلود نمی‌کند |
| ۴ نتیجه‌ی BERT اصلی | `maib-thesis/results` | سرور فقط seed 3 را آموزش می‌دهد |

اگر چیزی پیدا نشد، اسکریپت می‌گوید. در آن صورت سرور خودش آن را می‌سازد، که به اینترنت روی سرور نیاز دارد.

---

## قدم ۲-الف: Kaggle (پیشنهاد اصلی)

**یک بار:**
1. در https://www.kaggle.com ثبت‌نام کنید.
2. Settings → **Phone verification** را انجام دهید. بدون آن، GPU و اینترنت فعال نمی‌شوند.
3. **Datasets → New Dataset**: فایل `maib-server.zip` را آپلود کنید (Private).
4. **Code → New Notebook → File → Import Notebook**: فایل `kaggle_notebook.ipynb` را انتخاب کنید.
5. در پنل سمت راست:
   - **Accelerator → GPU T4 x2** را انتخاب کنید. P100 را انتخاب نکنید؛ نسخه‌های جدید PyTorch روی آن کار نمی‌کنند.
   - **Internet → On** را بزنید؛ برای دانلود `bert-base-uncased` لازم است.
   - **Add Input** → دیتاستی را که ساختید اضافه کنید.

**هر جلسه:**
6. **Save Version → Save & Run All (Commit)** را بزنید. اجرا در پس‌زمینه است و **می‌توانید مرورگر را ببندید.** لاگ در Versions → Logs است.
7. بعد از ۱۰.۵ ساعت، notebook خودش با خیال راحت متوقف می‌شود. همه‌ی فایل‌ها، از جمله checkpoint نیمه‌کاره، در **Output** همان نسخه ذخیره می‌شوند.

**جلسه‌ی بعد:**
8. notebook را باز کنید. **Add Input → Your Work → Notebooks** و **خروجی همین notebook** را اضافه کنید.
9. دوباره **Save & Run All (Commit)** را بزنید.
   - سلول اول نسخه‌ای را که بیشترین پیشرفت را دارد خودکار پیدا می‌کند.
   - آموزش از آخرین epoch تمام‌شده ادامه پیدا می‌کند.
10. این کار را تکرار کنید تا در لاگ `Done in ... h` را ببینید.

**گرفتن نتیجه:** تب **Output → Download**. پوشه‌های `results/` و `reports/` را نگه دارید؛ جدول نهایی در `reports/comparison.md` است.

---

## قدم ۲-ب: Google Colab

1. فایل `maib-server.zip` را در **ریشه‌ی My Drive** آپلود کنید.
2. در https://colab.research.google.com از **File → Upload notebook** فایل `colab_notebook.ipynb` را باز کنید.
3. **Runtime → Change runtime type → T4 GPU** را انتخاب کنید.
4. **Runtime → Run all** را بزنید.
   - پروژه در Drive است، پس با قطع شدن Colab چیزی از دست نمی‌رود.
   - **تب را باز نگه دارید.**
5. بعد از قطع شدن، دوباره **Run all** را بزنید. کار از آخرین epoch ادامه پیدا می‌کند.

---

## قدم ۲-ج: سرور GPU اجاره‌ای یا دانشگاه (لینوکس با SSH)

```bash
# روی لپ‌تاپ: فرستادن فایل
scp ../maib-server.zip root@IP:~

# روی سرور: نصب و چک کامل، قبل از شروع آموزش (حدود ۱۰ دقیقه)
ssh root@IP
unzip maib-server.zip && cd maib-bert-pretraining-server
bash install_server.sh
```

`install_server.sh` این‌ها را چک می‌کند تا ساعت‌های پولی صرف خطایابی نشود:
- درایور NVIDIA نصب است یا نه.
- **PyTorch مناسبِ همان درایور** را نصب می‌کند. PyTorch پیش‌فرض pip به درایور ۵۸۰ یا بالاتر نیاز دارد. اگر درایور سرور قدیمی‌تر باشد، نسخه‌ی سازگار از سایت PyTorch نصب می‌شود.
- GPU واقعاً با float16 محاسبه می‌کند یا نه.
- BERT از Hugging Face دانلود می‌شود یا نه.
- فایل‌های داده سر جایشان هستند یا نه.

همه‌ی خط‌ها باید `[ok]` باشند.

**اجرا (داخل tmux، تا با قطع شدن SSH متوقف نشود):**
```bash
tmux new -s maib
source .venv/bin/activate

# الف) اجرای کامل (بدون محدودیت بودجه)
python run_pipeline.py 2>&1 | tee -a run_full.log

# ب) با بودجه‌ی محدود: DAPT حداکثر ۳ epoch، بدون encoder سوم، و توقف خودکار بعد از ۸.۵ ساعت
timeout -s INT 8.5h python run_pipeline.py --corpora external task --external-epochs 3 2>&1 | tee -a run_full.log
```
- خروج از tmux بدون متوقف کردن آموزش: `Ctrl+B` و بعد `D`.
- برگشت به tmux: `tmux attach -t maib`.
- دیدن سرعت: `grep "\[epoch" run_full.log | tail`.

**در پایان، روی لپ‌تاپ همه‌چیز را دانلود کنید.** مدل‌ها و checkpointهای نیمه‌کاره را هم بگیرید تا اگر کاری مانده بود روی لپ‌تاپ ادامه پیدا کند:
```bash
rsync -av --exclude .venv --exclude corpus root@IP:~/maib-bert-pretraining-server/ server_run/
```

**مهم در سرور اجاره‌ای:**
- **تا وقتی سرور روشن است، هزینه حساب می‌شود.** بعد از گرفتن نتیجه‌ها، سرور را خاموش یا حذف کنید.
- **قبل از حذف**، `results/` و `reports/` را دانلود کنید. اگر مدل‌ها را برای کارهای بعدی می‌خواهید، `models/` را هم دانلود کنید (هر مدل حدود ۴۴۰ مگابایت).

---

## قدم ۲-د: تقسیم کار — پیش‌آموزش روی سرور GPU، fine-tune روی سرور CPU دانشگاه

مرحله‌ی ۱ (MLM) به GPU نیاز دارد. مرحله‌ی ۲ (۲۰ اجرای fine-tune) روی یک CPU پرهسته هم شدنی است، مثلاً سرور دانشگاه با ۸۰ هسته. پس:

**الف) روی سرور GPU (اجاره‌ای، حدود ۵ تا ۶ ساعت): فقط پیش‌آموزش**
```bash
bash install_server.sh
tmux new -s maib && source .venv/bin/activate
python step3_pretrain_mlm.py --corpus task 2>&1 | tee -a mlm.log                  # TAPT، ارزان، اول
python step3_pretrain_mlm.py --corpus external --epochs 3 2>&1 | tee -a mlm.log    # DAPT
```
بعد `models/` و `reports/` را دانلود کنید و **سرور را حذف کنید**:
```bash
rsync -av root@IP:~/maib-bert-pretraining-server/models/  models/
rsync -av root@IP:~/maib-bert-pretraining-server/reports/ reports/
```

**ب) روی سرور CPU دانشگاه: fine-tuneها، چند اجرا هم‌زمان**

پوشه را با `models/` و `reports/` به سرور ببرید و نصب کنید. نسخه‌ی CPU پایتورچ حجم کمتری دارد:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**اول یک تست ۱۰ دقیقه‌ای سرعت:**
```bash
OMP_NUM_THREADS=20 python step4_finetune.py --model bert_bilstm_att --loss wce --seed 3
```
عدد `s/it` روی نوار پیشرفت را بخوانید و با Ctrl+C متوقف کنید. **زمان هر اجرا ≈ s/it × ۵۰۵ × ۵ ثانیه.**

**اجرای همه‌ی fine-tuneها:**
```bash
tmux new -s maib && source .venv/bin/activate
python run_pipeline.py --corpora task external --workers 4 --threads 20 2>&1 | tee -a run_full.log
```
- `--workers 4 --threads 20` یعنی ۴ اجرا هم‌زمان، هر کدام با ۲۰ هسته (روی ۸۰ هسته). یک پروسه‌ی PyTorch نمی‌تواند ۸۰ هسته را خوب پر کند، ولی چند پروسه‌ی هم‌زمان می‌توانند.
- مرحله‌ی ۱ رد می‌شود، چون مدل‌ها در `models/` هستند. فقط fine-tuneها اجرا می‌شوند: BERT اصلی با seed 3، TAPT و DAPT.
- لاگ هر اجرا در `logs/<run>.log` است. وقتی هر اجرا تمام شود، نتیجه‌ی test آن در لاگ اصلی چاپ می‌شود.
- **هر اجرا دقیقاً همان نتیجه‌ای را می‌دهد که تنها اجرا می‌شد** (تست شده).
- **حافظه:** هر اجرا حدود ۴ تا ۶ گیگ RAM می‌خواهد. با ۴ اجرای هم‌زمان، حدود ۲۴ گیگ.
- **متوقف کردن:** Ctrl+C. همه‌ی اجراها متوقف می‌شوند و دفعه‌ی بعد از آخرین epoch تمام‌شده‌شان ادامه پیدا می‌کنند. اگر کسی سرور را لازم داشت، کمتر بگیرید، مثلاً `--workers 2 --threads 20`.

---

## نکته‌های مهم

- **تنظیمات `config.py` را وسط کار عوض نکنید.** checkpoint فقط با همان تنظیمات ادامه پیدا می‌کند.
- **هم‌زمان روی دو جا اجرا نکنید**، مثلاً لپ‌تاپ و Kaggle با هم.
- اگر وسط کار مجبور شدید سرویس را عوض کنید (مثلاً از Kaggle به Colab)، کل پوشه‌ی خروجی را، از جمله `models/` و فایل‌های `*.resume.pt`، به سرویس جدید ببرید. کار از همان‌جا ادامه پیدا می‌کند.
- نام GPU در فایل‌های نتیجه ثبت می‌شود (`"gpu": "Tesla T4"`). در پایان‌نامه بنویسید مرحله‌ی ۱ و ۲ روی چه GPUای اجرا شده‌اند.
- نتایج روی GPUهای مختلف ممکن است در رقم‌های آخر کمی فرق کنند. این طبیعی است.
