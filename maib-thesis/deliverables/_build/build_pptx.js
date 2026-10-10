// Persian RTL presentation built from data.json (made by build_data.py).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const [, , DATA_DIR, REPO, OUTFILE] = process.argv;
const D = JSON.parse(fs.readFileSync(path.join(DATA_DIR, "data.json"), "utf8"));
const cfg = (n) => D.configs.find((c) => c.name === n);
const has = (n) => Boolean(cfg(n));
const f3 = (x) => x.toFixed(3);
const sgn = (x) => (Math.abs(x) < 0.0005 ? "±0.000" : (x > 0 ? "+" : "−") + Math.abs(x).toFixed(3));
const d = (a, b, k = "macro_f1") => cfg(b)[k][0] - cfg(a)[k][0];
const pm = (ms) => (ms[1] > 0 ? `${f3(ms[0])} ± ${f3(ms[1])}` : f3(ms[0]));
const S = D.summary;
const REF = "bert_bilstm + ce", ATT = "bert_bilstm_att + ce", WCE = "bert_bilstm_att + wce";
const OS_BERT = "bert_bilstm_att + ce [oversample]", OS_SVM = "tfidf_svm_oversample";

// ---------------------------------------------------------------- palette (maritime: navy + sea teal, safety orange accent)
const NAVY = "14213D", TEAL = "0F7C8C", ORANGE = "E36414", INK = "1B1B1B", MUTED = "5B6770", TINT = "EEF3F6", ICE = "CFE3EA", WHITE = "FFFFFF";
const FONT = "Arial";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";               // 10 x 5.625 in
pres.rtlMode = true;
pres.title = "MAIB incident classification - progress report";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

pres.defineSlideMaster({
  title: "CONTENT", background: { color: WHITE },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.28, w: 9.0, h: 0.7,
    fontFace: FONT, fontSize: 26, bold: true, color: NAVY, align: "right", valign: "middle", rtlMode: true, margin: 0 }, text: "" } }],
  slideNumber: { x: 0.35, y: 5.2, w: 0.5, h: 0.3, fontFace: FONT, fontSize: 10, color: MUTED },
});
pres.defineSlideMaster({ title: "DARK", background: { color: NAVY },
  slideNumber: { x: 0.35, y: 5.2, w: 0.5, h: 0.3, fontFace: FONT, fontSize: 10, color: ICE } });

// ---------------------------------------------------------------- helpers
const T = (s, text, o) => s.addText(text, { isTextBox: true, fontFace: FONT, rtlMode: true, lang: "fa-IR", align: "right",
  valign: "top", margin: 0.05, fontSize: 15, color: INK, ...o });
const title = (s, text) => s.addText(text, { placeholder: "title" });
const card = (s, x, y, w, h, fill = TINT) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
const bullets = (s, items, o) => T(s, items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: 6 } })), o);
const img = (s, file, x, y, w) => {
  const b = fs.readFileSync(file); const iw = b.readUInt32BE(16), ih = b.readUInt32BE(20);
  s.addImage({ path: file, x, y, w, h: w * ih / iw });
  return w * ih / iw;
};
// RTL table: logical first column is drawn on the right
function table(s, header, rows, o) {
  const cell = (t, head, hi) => ({ text: t, options: { bold: head || hi, color: head ? WHITE : INK, fill: { color: head ? NAVY : (hi ? "FCE7DB" : WHITE) },
    align: "center", valign: "middle", rtlMode: true, fontFace: FONT, fontSize: o.fontSize || 12 } });
  const data = [header.map((h) => cell(h, true)).reverse(),
    ...rows.map((r) => r.cells.map((t) => cell(t, false, r.hi)).reverse())];
  s.addTable(data, { x: o.x, y: o.y, w: o.w, colW: [...o.colW].reverse(), border: { type: "solid", pt: 0.5, color: "C9D3DA" }, rowH: o.rowH || 0.3 });
}
const SHORT = { "tfidf_svm": "TF-IDF + SVM", "tfidf_svm_oversample": "SVM + oversampling", "bert + ce": "BERT",
  "bert_bilstm + ce": "BERT + BiLSTM (مرجع)", "bert_bilstm_att + ce": "+ Attention", "bert_bilstm_att + wce": "+ Attention + wce",
  "bert_bilstm_att + focal": "+ Attention + focal", "bert_bilstm_att + cb": "+ Attention + cb", "bert_bilstm_att + ce [oversample]": "+ Attention + oversampling" };

// ================================================================ 1. title
let s = pres.addSlide({ masterName: "DARK" });
T(s, "دانشگاه صنعتی شاهرود — دانشکده مهندسی کامپیوتر", { x: 0.6, y: 0.6, w: 8.8, h: 0.4, fontSize: 14, color: ICE });
T(s, "دسته‌بندی گزارش‌های حوادث دریایی با مدل‌های زبانی", { x: 0.6, y: 1.35, w: 8.8, h: 1.35, fontSize: 32, bold: true, color: WHITE });
T(s, "گزارش پیشرفت: نتایج کامل آزمایش‌ها و بررسی oversampling", { x: 0.6, y: 2.85, w: 8.8, h: 0.5, fontSize: 18, color: "7FD1DC" });
T(s, "محمد مهدی عرب   |   استاد راهنما: دکتر هدی مشایخی   |   مهر ۱۴۰۵", { x: 0.6, y: 4.4, w: 8.8, h: 0.4, fontSize: 14, color: ICE });
s.addNotes("سلام. این گزارش دو کاری را که جلسه‌ی قبل خواسته شد نشان می‌دهد: نتیجه‌ی کامل آزمایش‌ها و بررسی oversampling.");

// ================================================================ 2. main message
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "پیام اصلی در یک نگاه");
const stats = [
  ["tfidf_svm", "خط مبنای کلاسیک", "TF-IDF + SVM", TEAL],
  [REF, "بازتولید مقاله‌ی مرجع", "BERT + BiLSTM", TEAL],
  [WCE, "روش پیشنهادی", "BiLSTM + Attention + wce", ORANGE],
];
stats.forEach(([n, lab, sub, col], i) => {
  const x = 9.5 - (i + 1) * 3.0 + 0.15;        // first card on the right
  card(s, x, 1.2, 2.7, 1.75);
  s.addText(f3(cfg(n).macro_f1[0]), { isTextBox: true, x, y: 1.3, w: 2.7, h: 0.8, fontFace: FONT, fontSize: 40, bold: true, color: col, align: "center", margin: 0 });
  T(s, lab, { x, y: 2.1, w: 2.7, h: 0.35, fontSize: 14, bold: true, align: "center" });
  T(s, sub, { x, y: 2.47, w: 2.7, h: 0.35, fontSize: 11, color: MUTED, align: "center" });
});
T(s, "Macro-F1 روی مجموعه‌ی آزمون (۸۶۵ گزارش)، میانگین ۲ بذر", { x: 0.5, y: 3.0, w: 9.0, h: 0.3, fontSize: 11, color: MUTED, align: "center" });
bullets(s, [
  `بیشترین بهبود از تابع زیان وزن‌دار (wce) آمد: ${sgn(d(ATT, WCE))} نسبت به همان مدل با ce`,
  `BiLSTM (${sgn(d("bert + ce", REF))}) و attention (${sgn(d(REF, ATT))}) با تنظیمات یکسان بهبودی بیش از نوسان بذرها ندادند`,
  "سقف حدود ۰٫۹۱ بیشتر از ابهام برچسب گزارش‌های چندرویدادی می‌آید تا ضعف مدل",
], { x: 0.7, y: 3.45, w: 8.6, h: 1.6, fontSize: 15 });
s.addNotes("سه عدد اصلی: خط مبنای کلاسیک ۰٫۸۸۵، بازتولید مقاله‌ی مرجع ۰٫۹۰۱، و روش پیشنهادی ۰٫۹۱۴. نکته‌ی مهم این است که بهبود اصلی از تابع زیان وزن‌دار آمد، نه از لایه‌های اضافه.");

// ================================================================ 3. last meeting's tasks
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "وظایف جلسه‌ی قبل");
[[ "۱. نتیجه و گزارش کامل", [`${D.per_run.length} اجرای آموزش BERT و خط مبنای SVM`, "۶ پیکربندی × ۲ بذر، تنظیمات یکسان", "جدول‌ها، تحلیل attention و تحلیل خطا", "گزارش کامل Word پیوست است"]],
  [ "۲. بررسی oversampling", ["روی SVM و روی روش پیشنهادی", "فقط داده‌ی آموزش؛ آزمون دست‌نخورده", "مقایسه با بدون اقدام و با wce", has(OS_BERT) ? "انجام شد؛ نتایج در اسلاید ۱۰" : "پیاده‌سازی شد؛ اجرا در جریان"]],
].forEach(([h, items], i) => {
  const x = i === 0 ? 5.1 : 0.5;
  card(s, x, 1.25, 4.4, 3.6);
  T(s, h, { x: x + 0.25, y: 1.45, w: 3.9, h: 0.5, fontSize: 20, bold: true, color: i === 0 ? TEAL : ORANGE });
  bullets(s, items, { x: x + 0.25, y: 2.1, w: 3.9, h: 2.6, fontSize: 15 });
});

// ================================================================ 4. data
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "دادگان: ۹ رده، به‌شدت نامتوازن");
img(s, path.join(REPO, "reports", "class_distribution.png"), 0.4, 1.15, 5.4);
bullets(s, [
  `${S.raw_reports.toLocaleString("en")} گزارش عمومی MAIB، برچسب «Main Event L1»`,
  `۲ رده با کمتر از ۲۰ گزارش حذف شد (۳ و ۲ گزارش)`,
  `${S.kept_classes} رده، ${S.words_per_report.count.toLocaleString("en")} گزارش`,
  `نسبت عدم توازن: ${S.imbalance_ratio}`,
  `تقسیم ثابت ۷۰/۱۵/۱۵: ${S.split.train.toLocaleString("en")} / ${S.split.val} / ${S.split.test}`,
  `میانه‌ی طول گزارش: ${S.words_per_report["50%"]} واژه`,
], { x: 6.0, y: 1.3, w: 3.5, h: 3.7, fontSize: 14 });

// ================================================================ 5. design
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "طراحی آزمایش: در هر گام فقط یک جزء تغییر می‌کند");
const steps = [["TF-IDF + SVM", "خط مبنا (۲-الف)"], ["BERT", "۲-ب"], ["+ BiLSTM", "مقاله‌ی مرجع (۲-ج)"], ["+ Attention", "روش پیشنهادی (۳)"], ["+ wce / focal / cb\n+ oversampling", "نامتوازنی (۴)"]];
steps.forEach(([a, b], i) => {
  const w = 1.62, gap = 0.22, x = 9.5 - (i + 1) * w - i * gap;
  card(s, x, 1.45, w, 1.15, i === 3 ? "FCE7DB" : TINT);
  s.addText(a, { isTextBox: true, x, y: 1.5, w, h: 1.05, fontFace: FONT, fontSize: 14, bold: true, color: i === 3 ? ORANGE : NAVY, align: "center", valign: "middle", margin: 0.03 });
  T(s, b, { x, y: 2.68, w, h: 0.4, fontSize: 11, color: MUTED, align: "center" });
  if (i < steps.length - 1) s.addText("←", { isTextBox: true, x: x - gap - 0.02, y: 1.8, w: gap + 0.04, h: 0.4, fontFace: FONT, fontSize: 16, color: MUTED, align: "center", margin: 0 });
});
bullets(s, [
  "تنظیمات آموزش برای همه‌ی مدل‌ها کاملاً یکسان (دسته‌ی مؤثر ۳۲، نرخ یادگیری، ۵ epoch)",
  "بهترین epoch فقط با Macro-F1 اعتبارسنجی؛ مجموعه‌ی آزمون فقط یک بار",
  "هر پیکربندی با ۲ بذر؛ گزارش میانگین ± انحراف معیار",
  "معیار اصلی: Macro-F1 و بازیابی هر رده؛ دقت کلی فقط برای مقایسه",
], { x: 0.7, y: 3.25, w: 8.6, h: 1.8, fontSize: 14 });
s.addNotes("برخلاف مقاله‌ی مرجع، بین هر دو پیکربندی متوالی فقط یک جزء عوض شده و بقیه‌ی تنظیمات کاملاً یکسان است. پس می‌دانیم هر بهبود دقیقاً از کجا آمده.");

// ================================================================ 6. main table
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "نتایج اصلی روی مجموعه‌ی آزمون");
table(s, ["پیکربندی", "بذر", "Macro-F1", "بازیابی رده‌های بحرانی", "دقت کلی"],
  D.configs.map((c) => ({ hi: c.name === D.best_config, cells: [SHORT[c.name] || c.name, String(c.seeds), pm(c.macro_f1), pm(c.critical_recall), pm(c.accuracy)] })),
  { x: 0.5, y: 1.2, w: 9.0, colW: [2.9, 0.7, 1.8, 2.0, 1.6], fontSize: 12, rowH: 0.34 });
T(s, "رده‌های بحرانی: آتش‌سوزی، واژگونی، آب‌گرفتگی. ردیف رنگی: بهترین پیکربندی.", { x: 0.5, y: 4.85, w: 9.0, h: 0.3, fontSize: 11, color: MUTED });

// ================================================================ 7. chart
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "بیشترین بهبود از تابع زیان وزن‌دار آمد");
img(s, path.join(DATA_DIR, "fig_results.png"), 0.5, 1.1, 9.0);
T(s, "نقطه: میانگین ۲ بذر؛ خط: ± یک انحراف معیار", { x: 0.5, y: 4.95, w: 9.0, h: 0.3, fontSize: 11, color: MUTED, align: "center" });

// ================================================================ 8. RQ1
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "RQ1: attention در کل بهبود معناداری نداد");
card(s, 5.6, 1.2, 3.9, 2.0);
s.addText(sgn(d(REF, ATT)), { isTextBox: true, x: 5.6, y: 1.3, w: 3.9, h: 0.9, fontFace: FONT, fontSize: 40, bold: true, color: NAVY, align: "center", margin: 0 });
T(s, `تغییر Macro-F1 نسبت به BiLSTM\n(انحراف معیار بذرها حدود ${f3(Math.max(cfg(REF).macro_f1[1], cfg(ATT).macro_f1[1]))})`, { x: 5.7, y: 2.25, w: 3.7, h: 0.8, fontSize: 13, align: "center" });
T(s, "در هر دو بذر، روی رده‌های «برخورد» بهتر بود:", { x: 0.5, y: 1.2, w: 4.9, h: 0.45, fontSize: 14, bold: true });
table(s, ["رده", "BiLSTM", "+ Attention"], ["Contact", "Collision", "Grounding / Stranding"].map((c) => ({ cells: [`${c} (${D.support[c]})`, cfg(REF).found[c].toFixed(1).replace(".0", ""), cfg(ATT).found[c].toFixed(1).replace(".0", "")] })),
  { x: 0.5, y: 1.7, w: 4.9, colW: [2.5, 1.2, 1.2], fontSize: 12, rowH: 0.34 });
T(s, "تعداد گزارش درست‌تشخیص‌داده‌شده (میانگین بذرها)", { x: 0.5, y: 3.15, w: 4.9, h: 0.3, fontSize: 11, color: MUTED });
bullets(s, ["این رده‌ها معنای نزدیک دارند و تشخیصشان به یک واژه‌ی کلیدی وسط متن وابسته است: با فرضیه‌ی پیشنهاده سازگار",
  "ولی اندازه‌ی اثر کوچک است (حدود ۲ گزارش در Contact) و روی رده‌های بحرانی بهبودی دیده نشد"], { x: 0.5, y: 3.6, w: 9.0, h: 1.4, fontSize: 14 });

// ================================================================ 9. RQ2
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "RQ2: wce بهترین بود و focal اثری نداشت");
const losses = ["wce", "cb", "focal"].filter((l) => has(`bert_bilstm_att + ${l}`));
table(s, ["تابع زیان", "Macro-F1", "تغییر نسبت به ce", "بازیابی بحرانی", "دقت کلی"],
  losses.map((l) => { const n = `bert_bilstm_att + ${l}`; return { hi: l === "wce", cells: [l, pm(cfg(n).macro_f1), sgn(d(ATT, n)), f3(cfg(n).critical_recall[0]), f3(cfg(n).accuracy[0])] }; }),
  { x: 0.5, y: 1.2, w: 9.0, colW: [1.4, 2.0, 1.9, 1.9, 1.8], fontSize: 13, rowH: 0.36 });
bullets(s, [
  `wce: تنها بهبودی که به‌روشنی از نوسان بذرها بزرگ‌تر است؛ دقت کلی هم کم نشد`,
  "معامله‌ی wce: واژگونی حدود ۱ گزارش بیشتر از ۱۵، از دست رفتن کنترل ۳ گزارش بیشتر از ۱۲۴، در ازای حدود ۱٫۵ گزارش کمتر از ۲۵۰ در پرتکرارترین رده",
  "cb: بهبود کوچک‌تر و مرزی؛ focal: در Macro-F1 بی‌اثر",
], { x: 0.6, y: 2.75, w: 8.8, h: 2.3, fontSize: 14 });

// ================================================================ 10. oversampling
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Oversampling: دیدن بیشتر به‌جای جریمه‌ی بیشتر");
card(s, 5.1, 1.2, 4.4, 1.55); card(s, 0.5, 1.2, 4.4, 1.55);
T(s, "SVM", { x: 5.3, y: 1.3, w: 4.0, h: 0.4, fontSize: 16, bold: true, color: TEAL });
T(s, "گزارش‌های رده‌های کوچک تکرار می‌شوند تا همه‌ی رده‌ها هم‌اندازه‌ی بزرگ‌ترین شوند", { x: 5.3, y: 1.72, w: 4.0, h: 0.95, fontSize: 13 });
T(s, "BERT + BiLSTM + Attention", { x: 0.7, y: 1.3, w: 4.0, h: 0.4, fontSize: 16, bold: true, color: ORANGE });
T(s, "انتخاب هر گزارش با احتمال ۱ تقسیم بر اندازه‌ی رده؛ طول epoch ثابت؛ با ce (نه wce)", { x: 0.7, y: 1.72, w: 4.0, h: 0.95, fontSize: 13 });
if (has(OS_BERT) || has(OS_SVM)) {
  const rows = [];
  if (has(OS_SVM)) rows.push({ cells: ["TF-IDF + SVM", f3(cfg("tfidf_svm").macro_f1[0]), f3(cfg(OS_SVM).macro_f1[0]), sgn(d("tfidf_svm", OS_SVM)), "—"] });
  if (has(OS_BERT)) rows.push({ hi: true, cells: ["+ Attention", pm(cfg(ATT).macro_f1), pm(cfg(OS_BERT).macro_f1), sgn(d(ATT, OS_BERT)), sgn(d(WCE, OS_BERT))] });
  table(s, ["مدل", "بدون oversampling", "با oversampling", "تغییر", "در برابر wce"], rows, { x: 0.5, y: 2.95, w: 9.0, colW: [2.0, 2.0, 2.0, 1.5, 1.5], fontSize: 13, rowH: 0.36 });
  T(s, "<<OS_SLIDE_TEXT>>", { x: 0.5, y: 4.15, w: 9.0, h: 0.9, fontSize: 14 });
} else {
  T(s, "فقط داده‌ی آموزش بازنمونه‌گیری می‌شود؛ اعتبارسنجی و آزمون دست‌نخورده‌اند. نتایج پس از پایان اجرا اضافه می‌شود.", { x: 0.5, y: 3.1, w: 9.0, h: 0.8, fontSize: 15 });
}

// ================================================================ 11. rare classes
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "رده‌های نادر: ۱۵ نمونه برای قضاوت کم است");
const capN = ["tfidf_svm", "bert + ce", REF, ATT, WCE, "bert_bilstm_att + focal", "bert_bilstm_att + cb", OS_BERT].filter(has);
s.addChart(pres.charts.BAR, [{ name: "Capsizing / Listing", labels: capN.map((n) => (SHORT[n] || n).replace(" (مرجع)", "")), values: capN.map((n) => +cfg(n).found["Capsizing / Listing"].toFixed(1)) }], {
  x: 0.4, y: 1.15, w: 5.6, h: 3.9, barDir: "bar", chartColors: [TEAL], valAxisMinVal: 0, valAxisMaxVal: 15, valAxisMajorUnit: 5,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFontFace: FONT,
  catAxisLabelFontFace: FONT, catAxisLabelFontSize: 10, catAxisLabelColor: INK, valAxisLabelFontFace: FONT, valAxisLabelFontSize: 9, valAxisLabelColor: MUTED,
  valGridLine: { color: "E3E8EC", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, catAxisOrientation: "maxMin",
  showTitle: true, title: "واژگونی: گزارش‌های پیدا‌شده از ۱۵", titleFontFace: FONT, titleFontSize: 12, titleColor: NAVY,
});
bullets(s, ["هر گزارش ۶٫۷ واحد بازیابی را جابه‌جا می‌کند", "تفاوت ۱ تا ۲ گزارش بین مدل‌های BERT قابل اتکا نیست", "قطعی: همه‌ی مدل‌های BERT از SVM (۱۰ از ۱۵) بهترند", "برای ادعای قوی‌تر: بذر بیشتر و آزمون معناداری"],
  { x: 6.2, y: 1.4, w: 3.3, h: 3.6, fontSize: 14 });

// ================================================================ 12. attention words
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "Attention: مدل به واژه‌های درست نگاه می‌کند");
const W = { "Fire / Explosion": "آتش‌سوزی", "Flooding / Foundering": "آب‌گرفتگی", "Grounding / Stranding": "به گل نشستن", "Capsizing / Listing": "واژگونی", "Contact": "Contact", "Loss Of Control": "از دست رفتن کنترل" };
const STOP = new Set(["vessel", "resulting", "suffered", "experienced", "damage", "crew", "crewmember", "member", "did", "ship", "hit", "entered", "boat"]);
Object.entries(W).forEach(([c, fa], i) => {
  const col = i % 3, row = Math.floor(i / 3), w = 2.85, h = 1.05, x = 9.5 - (col + 1) * w - col * 0.2, y = 1.15 + row * (h + 0.15);
  card(s, x, y, w, h);
  T(s, fa, { x: x + 0.1, y: y + 0.08, w: w - 0.2, h: 0.35, fontSize: 14, bold: true, color: TEAL });
  s.addText((D.top_words[c] || []).filter((x) => !STOP.has(x)).slice(0, 3).join(" · "), { isTextBox: true, x: x + 0.1, y: y + 0.48, w: w - 0.2, h: 0.5, fontFace: FONT, fontSize: 13, color: INK, align: "center", margin: 0 });
});
card(s, 0.5, 3.6, 9.0, 1.45, "FCE7DB");
T(s, "یافته‌ی غیرمنتظره در رده‌ی «آسیب تجهیزات»", { x: 0.7, y: 3.7, w: 8.6, h: 0.4, fontSize: 15, bold: true, color: ORANGE });
T(s, "پروزن‌ترین واژه‌ها: solas، ladder، pilot، conform، regulations. بخش بزرگی از این رده گزارش‌های عدم انطباق پلکان راهنما با مقررات SOLAS است: زیرگروهی پنهان که attention آشکار کرد.", { x: 0.7, y: 4.12, w: 8.6, h: 0.85, fontSize: 13 });
s.addNotes("وزن attention نشان می‌دهد مدل کجا را نگاه کرده، نه لزوماً چرا تصمیم گرفته. این محدودیت شناخته‌شده است (Jain و Wallace، ۲۰۱۹).");

// ================================================================ 13. error analysis
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "خطاهای باقی‌مانده: ابهام برچسب، نه ضعف مدل");
[["علت ← پیامد", "«propulsion failure and grounded»", "برچسب: به گل نشستن؛ پیش‌بینی: از دست رفتن کنترل. در گزارش مشابه برچسب برعکس است."],
 ["چند رویداد در یک گزارش", "«hit the lock side … crewmember fell overboard»", "برچسب: Contact؛ پیش‌بینی: آسیب به اشخاص. هر دو منطقی است."],
 ["واژگان تخصصی دریایی", "«girting»، «allided»", "برای BERT عمومی ناآشنا؛ attention واژه‌ی درست را یافت ولی معنایش را نمی‌دانست."]].forEach(([h, ex, txt], i) => {
  const w = 2.85, x = 9.5 - (i + 1) * w - i * 0.225;
  card(s, x, 1.2, w, 2.75);
  T(s, h, { x: x + 0.15, y: 1.3, w: w - 0.3, h: 0.4, fontSize: 15, bold: true, color: NAVY });
  s.addText(ex, { isTextBox: true, x: x + 0.15, y: 1.75, w: w - 0.3, h: 0.6, fontFace: FONT, fontSize: 11, italic: true, color: TEAL, align: "center", valign: "middle", margin: 0 });
  T(s, txt, { x: x + 0.15, y: 2.4, w: w - 0.3, h: 1.45, fontSize: 13 });
});
T(s, "نتیجه: سقف حدود ۰٫۹۱ از برچسب تک‌کلاسه برای گزارش‌های چندرویدادی می‌آید؛ برای همین TF-IDF به BERT نزدیک است و لایه‌های اضافه کمک کمی می‌کنند.", { x: 0.5, y: 4.15, w: 9.0, h: 0.9, fontSize: 14, bold: true });

// ================================================================ 14. reference paper
s = pres.addSlide({ masterName: "CONTENT" });
title(s, "مقاله‌ی مرجع: سه نکته‌ی روش‌شناختی");
bullets(s, [
  "مدل‌ها با dropout، منظم‌سازی، آستانه‌ی گرادیان و تابع فعال‌سازی متفاوت مقایسه شده‌اند (جدول ۹)",
  "یک اجرا، بدون بذر یا آزمون معناداری؛ بدون خط مبنای کلاسیک",
  "در جدول ۳ خود مقاله، تغییر هایپرپارامترها دقت را ۲٫۳ واحد جابه‌جا می‌کند؛ بیشتر از بهبود ادعاشده‌ی BiLSTM (۱٫۱ واحد)",
], { x: 4.3, y: 1.25, w: 5.2, h: 3.0, fontSize: 14 });
card(s, 0.5, 1.25, 3.5, 2.6, "FCE7DB");
T(s, "در پژوهش حاضر", { x: 0.65, y: 1.35, w: 3.2, h: 0.4, fontSize: 15, bold: true, color: ORANGE, align: "center" });
s.addText(sgn(d("bert + ce", REF)), { isTextBox: true, x: 0.5, y: 1.85, w: 3.5, h: 0.8, fontFace: FONT, fontSize: 36, bold: true, color: NAVY, align: "center", margin: 0 });
T(s, "اثر BiLSTM بر Macro-F1 با تنظیمات یکسان و ۲ بذر: در حد نوسان", { x: 0.65, y: 2.7, w: 3.2, h: 0.9, fontSize: 13, align: "center" });
T(s, "توجه: وظیفه‌ی مقاله‌ی مرجع «علت حادثه» (۳۲ رده) است و وظیفه‌ی ما «نوع رویداد» (۹ رده)؛ اعداد مستقیماً قابل مقایسه نیستند.", { x: 0.5, y: 4.3, w: 9.0, h: 0.75, fontSize: 12, color: MUTED });

// ================================================================ 15. limits + next
s = pres.addSlide({ masterName: "DARK" });
T(s, "محدودیت‌ها و گام‌های بعدی", { x: 0.5, y: 0.35, w: 9.0, h: 0.7, fontSize: 26, bold: true, color: WHITE });
T(s, "محدودیت‌ها", { x: 5.1, y: 1.2, w: 4.4, h: 0.4, fontSize: 17, bold: true, color: "7FD1DC" });
bullets(s, ["فقط ۲ بذر؛ بدون آزمون معناداری", "۱۵ نمونه‌ی واژگونی در آزمون", `بهترین epoch در ${D.best_epoch_counts["5"] || 0} از ${D.per_run.length} اجرا همان epoch آخر بود`, "γ و β تنظیم نشدند"],
  { x: 5.1, y: 1.7, w: 4.4, h: 3.2, fontSize: 14, color: WHITE });
T(s, "گام‌های بعدی", { x: 0.5, y: 1.2, w: 4.4, h: 0.4, fontSize: 17, bold: true, color: "F4A261" });
bullets(s, ["بذر سوم و bootstrap جفت‌شده", "آموزش طولانی‌تر (۸ epoch)", "RoBERTa / DeBERTa و مدل دامنه‌ی دریایی", "مقایسه با LLM (zero-shot)", "دسته‌بندی چندبرچسبی برای علت و پیامد"],
  { x: 0.5, y: 1.7, w: 4.4, h: 3.2, fontSize: 14, color: WHITE });

pres.writeFile({ fileName: OUTFILE }).then((f) => console.log("wrote", f));
