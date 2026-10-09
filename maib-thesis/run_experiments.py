"""
اجرای جدول آزمایش‌های پروپوزال، با قابلیت ادامه (resume):
اجراهای تمام‌شده دوباره اجرا نمی‌شوند؛ اگر لپ‌تاپ خاموش شد، همان دستور را دوباره بزن.

    python run_experiments.py --preset smoke        # تست سریع (۱ اجرا، ۱ epoch)
    python run_experiments.py --preset main         # آزمایش اصلی: ۳ head × ۴ loss × ۳ seed = ۳۶ اجرا
    python run_experiments.py --preset paper        # بازتولید با تنظیمات دقیق مقاله‌ی مرجع (lr=1e-6)
    python run_experiments.py --preset ablation     # آیا BiLSTM لازم است؟ توجه مستقیم روی BERT
"""
import argparse
import itertools
import traceback
from pathlib import Path

from train import get_parser, run, run_name

PRESETS = {
    # گام ۲ تا ۴ پروپوزال: هر head با هر loss، هر کدام ۳ بار
    #   cls        = BERT ساده (گام ۲-ب)
    #   bilstm     = BERT + BiLSTM مقاله‌ی مرجع (گام ۲-ج)
    #   bilstm_att = روش پیشنهادی (گام ۳)
    #   ce = بدون اقدام | wce = وزن معکوس فراوانی | focal = کانونی | cb = تعداد مؤثر (گام ۴)
    "main": dict(models=["bert-base-uncased"], heads=["cls", "bilstm", "bilstm_att"],
                 losses=["ce", "wce", "focal", "cb"], seeds=[1, 2, 3], overrides={}),
    # بازتولید وفادار به مقاله: lr=1e-6 برای همه‌ی لایه‌ها، batch مؤثر ۳۲، epochهای زیاد
    "paper": dict(models=["bert-base-uncased"], heads=["cls", "mean", "bilstm"],
                  losses=["ce"], seeds=[1],
                  overrides=dict(lr=1e-6, head_lr=1e-6, epochs=30, patience=5,
                                 results_dir="results_paper")),
    # ablation: سهم BiLSTM از سهم توجه جدا شود
    "ablation": dict(models=["bert-base-uncased"], heads=["mean", "attention"],
                     losses=["focal"], seeds=[1, 2, 3], overrides={}),
    # تعمیم به رمزگذارهای دیگر (اختیاری، اگر وقت ماند)
    "models": dict(models=["roberta-base", "microsoft/deberta-v3-base"], heads=["bilstm_att"],
                   losses=["focal"], seeds=[1, 2, 3], overrides={}),
    "smoke": dict(models=["bert-base-uncased"], heads=["bilstm_att"], losses=["focal"],
                  seeds=[1], overrides=dict(epochs=1, results_dir="results_smoke")),
}

ap = argparse.ArgumentParser(parents=[get_parser()], add_help=True, conflict_handler="resolve")
ap.add_argument("--preset", choices=PRESETS, default="main")
ap.add_argument("--losses", nargs="*", help="override preset losses")
ap.add_argument("--heads", nargs="*", help="override preset heads")
ap.add_argument("--models", nargs="*", help="override preset models")
ap.add_argument("--seeds", nargs="*", type=int, help="override preset seeds")
args = vars(ap.parse_args())

preset = dict(PRESETS[args.pop("preset")])
for k in ("losses", "heads", "models", "seeds"):
    v = args.pop(k)
    if v:
        preset[k] = v
parser_defaults = vars(get_parser().parse_args([]))
for k, v in preset["overrides"].items():
    if args.get(k) == parser_defaults.get(k):      # فقط اگر کاربر خودش آن را عوض نکرده باشد
        args[k] = v

combos = list(itertools.product(preset["models"], preset["heads"], preset["losses"], preset["seeds"]))
print(f"{len(combos)} runs planned -> {args['results_dir']}/")
for i, (m, head, loss, seed) in enumerate(combos, 1):
    cfg = dict(args, model=m, head=head, loss=loss, seed=seed)
    name = run_name(cfg)
    if (Path(cfg["results_dir"]) / f"{name}.json").exists():
        print(f"[{i}/{len(combos)}] skip (done): {name}")
        continue
    print(f"\n[{i}/{len(combos)}] ===== {name} =====")
    try:
        run(cfg)
    except Exception:
        traceback.print_exc()
        print(f"[FAILED] {name} — continuing with the next run")
