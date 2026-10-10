// Persian RTL progress report built from data.json (made by build_data.py).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  HeadingLevel, WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Footer, PageNumber,
} = require("docx");

const [, , DATA_DIR, REPO, OUTFILE] = process.argv;
const D = JSON.parse(fs.readFileSync(path.join(DATA_DIR, "data.json"), "utf8"));
const FA = "B Nazanin", EN = "Times New Roman", BODY = 26;          // 13 pt
const CONTENT_W = 9638;                                              // A4 minus 2 cm margins, in DXA
const f3 = (x) => x.toFixed(3);
const cfg = (n) => D.configs.find((c) => c.name === n);
const has = (n) => Boolean(cfg(n));

// ---------------------------------------------------------------- text helpers
// "**bold**" inside a string becomes a bold run. Every run is right-to-left with the Persian font.
function runs(text, o = {}) {
  return String(text).split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((part) => {
    const bold = part.startsWith("**");
    return new TextRun({
      text: bold ? part.slice(2, -2) : part, rightToLeft: true,
      font: { ascii: EN, hAnsi: EN, cs: FA }, size: o.size || BODY, sizeComplexScript: o.size || BODY,
      bold: bold || o.bold, boldComplexScript: bold || o.bold, color: o.color,
    });
  });
}
const P = (text, o = {}) => new Paragraph({
  bidirectional: true, alignment: o.align || AlignmentType.BOTH,
  spacing: { after: o.after ?? 120, line: 300 }, children: runs(text, o),
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, bidirectional: true, children: runs(t, { size: 34, bold: true, color: "1F3864" }), spacing: { before: 360, after: 160 } });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, bidirectional: true, children: runs(t, { size: 29, bold: true, color: "2E5597" }), spacing: { before: 240, after: 120 } });
const BULLET = (t) => new Paragraph({ bidirectional: true, alignment: AlignmentType.BOTH, numbering: { reference: "bul", level: 0 }, spacing: { after: 80, line: 300 }, children: runs(t) });
const NUM = (t, ref = "num") => new Paragraph({ bidirectional: true, alignment: AlignmentType.BOTH, numbering: { reference: ref, level: 0 }, spacing: { after: 80, line: 300 }, children: runs(t) });
const CAPTION = (t) => P(t, { size: 21, align: AlignmentType.CENTER, color: "52514E", after: 240 });

// ---------------------------------------------------------------- tables
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders,
    shading: head ? { fill: "DCE6F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ bidirectional: true, alignment: AlignmentType.CENTER, children: runs(text, { size: 21, bold: head }) })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, visuallyRightToLeft: true,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
           ...rows.map((r) => new TableRow({ children: r.map((t, i) => cell(t, i, false)) }))],
  });
}
const SPACER = () => new Paragraph({ children: [], spacing: { after: 160 } });

// ---------------------------------------------------------------- images
function pngSize(file) { const b = fs.readFileSync(file); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function figure(file, caption, widthPx = 620) {
  if (!fs.existsSync(file)) return [P(`(شکل در دسترس نیست: ${path.basename(file)})`)];
  const [w, h] = pngSize(file);
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
            children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: widthPx, height: Math.round(widthPx * h / w) } })] }),
          CAPTION(caption)];
}

// ---------------------------------------------------------------- names
const FA_NAME = {
  "tfidf_svm": "TF-IDF + SVM", "tfidf_svm_oversample": "TF-IDF + SVM + oversampling",
  "bert + ce": "BERT", "bert_bilstm + ce": "BERT + BiLSTM (بازتولید مرجع)",
  "bert_bilstm_att + ce": "BERT + BiLSTM + Attention", "bert_bilstm_att + wce": "+ Attention + wce",
  "bert_bilstm_att + focal": "+ Attention + focal", "bert_bilstm_att + cb": "+ Attention + cb",
  "bert_bilstm_att + ce [oversample]": "+ Attention + oversampling",
};
const CLASS_FA = {
  "Accident to person(s)": "آسیب به اشخاص", "Damage / Loss Of Equipment": "آسیب یا از دست رفتن تجهیزات",
  "Loss Of Control": "از دست رفتن کنترل", "Grounding / Stranding": "به گل نشستن", "Contact": "برخورد با جسم ثابت (Contact)",
  "Collision": "تصادم دو شناور (Collision)", "Fire / Explosion": "آتش‌سوزی و انفجار", "Flooding / Foundering": "آب‌گرفتگی و غرق",
  "Capsizing / Listing": "واژگونی و کج شدن", "Hull Failure": "آسیب بدنه", "Non-accidental Event": "رویداد غیرتصادفی",
};
const pm = (ms) => (ms[1] > 0 ? `${f3(ms[0])} ± ${f3(ms[1])}` : f3(ms[0]));
const delta = (a, b, key = "macro_f1") => cfg(b)[key][0] - cfg(a)[key][0];
const sgn = (x) => (Math.abs(x) < 0.0005 ? "±0.000" : (x > 0 ? "+" : "−") + Math.abs(x).toFixed(3));

const S = D.summary, ref = "bert_bilstm + ce", att = "bert_bilstm_att + ce", wce = "bert_bilstm_att + wce";
const best = cfg(D.best_config);
const OS_BERT = "bert_bilstm_att + ce [oversample]", OS_SVM = "tfidf_svm_oversample";
const haveOS = has(OS_BERT) || has(OS_SVM);

// ---------------------------------------------------------------- oversampling narrative (chosen from the numbers)
const CAP = "Capsizing / Listing";
const FA_DIGITS = (x) => String(x).replace(/\d/g, (d) => "۰۱۲۳۴۵۶۷۸۹"[d]).replace(".", "٫");
const nFound = (n, c) => cfg(n).found[c].toFixed(1).replace(".0", "");
const noiseOf = (a, b) => Math.max(cfg(a).macro_f1[1], cfg(b).macro_f1[1]);
const beyond = (a, b) => Math.abs(delta(a, b)) > noiseOf(a, b);
const verdictFa = (a, b) => (cfg(a).seeds < 2 && cfg(b).seeds < 2 ? "تک‌اجرا، بدون برآورد نوسان"
  : beyond(a, b) ? "بزرگ‌تر از نوسان بذرها" : "در حد نوسان بذرها");
function osCase() {          // which of the five possible outcomes the BERT oversampling run produced
  const upCe = beyond(att, OS_BERT) && delta(att, OS_BERT) > 0;
  if (!beyond(wce, OS_BERT)) return upCe ? "equal_wce" : "between";
  if (delta(wce, OS_BERT) > 0) return "better_wce";
  return upCe ? "helped_less" : "no_help";
}
function osParagraphs() {
  const out = [];
  if (has(OS_SVM)) {
    out.push(`در SVM، oversampling مقدار Macro-F1 را از ${f3(cfg("tfidf_svm").macro_f1[0])} به ${f3(cfg(OS_SVM).macro_f1[0])} رساند (${sgn(delta("tfidf_svm", OS_SVM))}) و بازیابی رده‌های بحرانی ${sgn(delta("tfidf_svm", OS_SVM, "critical_recall"))} تغییر کرد. در رده‌ی واژگونی ${FA_DIGITS(nFound(OS_SVM, CAP))} از ${FA_DIGITS(D.support[CAP])} گزارش درست تشخیص داده شد (بدون oversampling: ${FA_DIGITS(nFound("tfidf_svm", CAP))}). SVM با یک بذر ثابت یک بار اجرا شد، پس برای این تفاوت برآورد نوسانی در دست نیست.`);
  }
  if (!has(OS_BERT)) {
    out.push("اجرای oversampling روی مدل پیشنهادی (BERT + BiLSTM + Attention) هنوز در جریان است و نتیجه‌ی آن پس از پایان اجرا اضافه می‌شود.");
    return out;
  }
  const oneSeed = cfg(OS_BERT).seeds < 2;
  const rep = FA_DIGITS((D.summary.split.train / D.labels.length / D.train_support[CAP]).toFixed(1));
  out.push(`در مدل پیشنهادی${oneSeed ? " (فقط ۱ بذر؛ نتیجه‌ی اولیه)" : ""}، oversampling مقدار Macro-F1 را به ${pm(cfg(OS_BERT).macro_f1)} رساند: ${sgn(delta(att, OS_BERT))} نسبت به ce بدون اقدام (${verdictFa(att, OS_BERT)}) و ${sgn(delta(wce, OS_BERT))} نسبت به wce (${verdictFa(wce, OS_BERT)}). در رده‌ی واژگونی به‌طور میانگین ${FA_DIGITS(nFound(OS_BERT, CAP))} از ${FA_DIGITS(D.support[CAP])} گزارش درست تشخیص داده شد (ce: ${FA_DIGITS(nFound(att, CAP))}، wce: ${FA_DIGITS(nFound(wce, CAP))}).`);
  const memo = `یک توضیح محتمل که هنوز آزموده نشده: هر یک از ${FA_DIGITS(D.train_support[CAP])} گزارش واژگونی در داده‌ی آموزش در هر epoch حدود ${rep} بار تکرار می‌شود و مدل ممکن است به‌جای یادگیری الگو، همین گزارش‌ها را حفظ کند. wce همین جبران را بدون تکرار داده انجام می‌دهد.`;
  // rare-class recall went up -> the loss is elsewhere (a trade-off), not memorisation of the repeated reports
  const why = () => delta(att, OS_BERT, "critical_recall") > 0
    ? `بازیابی رده‌های بحرانی بالا رفت (${sgn(delta(att, OS_BERT, "critical_recall"))})، ولی precision همین رده‌ها ${sgn(delta(att, OS_BERT, "critical_precision"))} تغییر کرد؛ یعنی مدل این رده‌ها را بیشتر پیش‌بینی می‌کند و بخشی از این پیش‌بینی‌ها نادرست است. oversampling مرز تصمیم را بیش از اندازه به سمت رده‌های کمیاب می‌برد.`
    : memo;
  let concl = {
    equal_wce: "نتیجه: oversampling نسبت به ce بدون اقدام کمک کرد و به همان سطح wce رسید. پس در این داده تکرار نمونه‌ها و وزن‌دادن به خطا دو راه هم‌ارز برای جبران نامتوازنی‌اند. wce ساده‌تر است، چون داده‌ی آموزش را تغییر نمی‌دهد.",
    between: "نتیجه: oversampling بین ce و wce قرار گرفت و از هیچ‌کدام به‌روشنی جدا نیست. با این تعداد بذر نمی‌توان گفت oversampling کمک کرده است؛ wce همچنان تنها روشی است که بهبودی بزرگ‌تر از نوسان بذرها داده است.",
    better_wce: "نتیجه: oversampling از wce هم بهتر بود و بهترین پیکربندی فعلی است.",
    helped_less: "نتیجه: oversampling نسبت به ce بدون اقدام کمک کرد، ولی کمتر از wce. " + why(),
    no_help: "نتیجه: oversampling نسبت به ce بدون اقدام کمکی نکرد و از wce ضعیف‌تر بود. " + why(),
  }[osCase()];
  if (oneSeed) concl += " چون فقط یک بذر اجرا شده، این نتیجه پس از اجرای بذر دوم قطعی می‌شود.";
  out.push(concl);
  return out;
}
const nRuns = D.per_run.length;
const at5 = D.best_epoch_counts["5"] || 0;

// ---------------------------------------------------------------- content
const C = [];
// title page
C.push(new Paragraph({ spacing: { before: 1800 }, children: [] }));
for (const [t, s, b, c] of [
  ["دانشگاه صنعتی شاهرود — دانشکده مهندسی کامپیوتر", 26, false, "52514E"],
  ["دسته‌بندی گزارش‌های حوادث دریایی با استفاده از مدل‌های زبانی", 40, true, "1F3864"],
  ["Classification of Marine Incident Reports Using Language Models", 24, false, "52514E"],
  ["گزارش پیشرفت: نتایج کامل آزمایش‌ها و بررسی oversampling", 30, true, "2E5597"],
]) C.push(P(t, { size: s, bold: b, color: c, align: AlignmentType.CENTER, after: 360 }));
C.push(new Paragraph({ spacing: { before: 900 }, children: [] }));
for (const t of ["دانشجو: محمد مهدی عرب", "استاد راهنما: دکتر هدی مشایخی", "استاد مشاور: دکتر مریم خدابخش", "مهر ۱۴۰۵"])
  C.push(P(t, { align: AlignmentType.CENTER, after: 120 }));
C.push(new Paragraph({ children: [new PageBreak()] }));

// 1. summary
C.push(H1("۱. خلاصه‌ی نتایج"));
C.push(P(`در این مرحله ${nRuns} اجرای آموزش مدل‌های مبتنی بر BERT (۶ پیکربندی × ۲ بذر تصادفی${has(OS_BERT) ? " به‌همراه اجراهای oversampling" : ""}) و خط مبنای TF-IDF + SVM روی دادگان عمومی MAIB انجام شد. یافته‌های اصلی:`));
C.push(BULLET(`همه‌ی مدل‌های مبتنی بر BERT از خط مبنای کلاسیک بهترند: Macro-F1 از **${f3(cfg("tfidf_svm").macro_f1[0])}** برای TF-IDF + SVM به **${f3(cfg("bert + ce").macro_f1[0])} تا ${f3(best.macro_f1[0])}** رسید.`));
C.push(BULLET(`با تنظیمات آموزش یکسان، افزودن BiLSTM (${sgn(delta("bert + ce", ref))}) و افزودن attention (${sgn(delta(ref, att))}) بهبودی بزرگ‌تر از نوسان بین بذرها ایجاد نکرد. ادعای مقاله‌ی مرجع درباره‌ی برتری BiLSTM روی این داده تأیید نشد.`));
C.push(BULLET(`آنتروپی متقاطع وزن‌دار (wce) بهترین نتیجه را داد: Macro-F1 برابر **${pm(cfg(wce).macro_f1)}** و دقت کلی **${f3(cfg(wce).accuracy[0])}**. روش پیشنهادی کامل (attention + wce) نسبت به بازتولید مقاله‌ی مرجع **${sgn(delta(ref, wce))}** Macro-F1 بهتر است و بیشتر این بهبود از تابع زیان می‌آید، نه از attention.`));
if (has(OS_BERT)) C.push(BULLET(`oversampling روی BERT${cfg(OS_BERT).seeds < 2 ? " (۱ بذر، نتیجه‌ی اولیه)" : ""} به Macro-F1 برابر **${pm(cfg(OS_BERT).macro_f1)}** رسید: ${sgn(delta(att, OS_BERT))} نسبت به همان مدل بدون هیچ اقدامی و ${sgn(delta(wce, OS_BERT))} نسبت به wce.`));
else if (has(OS_SVM)) C.push(BULLET(`oversampling روی SVM مقدار Macro-F1 را ${sgn(delta("tfidf_svm", OS_SVM))} تغییر داد. اجرای آن روی مدل پیشنهادی هنوز در جریان است (بخش ۵-۴).`));
else C.push(BULLET("آزمایش oversampling (درخواست جلسه‌ی قبل) پیاده‌سازی شده و در حال اجراست (بخش ۵-۴)."));
C.push(BULLET("وزن‌های attention روی واژه‌های کلیدی معنادار (مانند smoke، aground، ingress، capsized) متمرکزند و یک زیرگروه پنهان در کلاس «آسیب تجهیزات» را آشکار کردند: گزارش‌های عدم انطباق پلکان راهنما با مقررات SOLAS."));
C.push(BULLET("تحلیل خطا نشان می‌دهد بخش عمده‌ی خطاهای باقی‌مانده از گزارش‌های چندرویدادی و زنجیره‌ی «علت ← پیامد» است (مثلاً خرابی موتور که به به گل نشستن منجر شده). این ابهام ذاتی برچسب تک‌کلاسه، سقف حدود ۰٫۹۱ را توضیح می‌دهد."));

// 2. goals
C.push(H1("۲. هدف و پرسش‌های پژوهش"));
C.push(P("مسأله‌ی پژوهش، طبق پیشنهاده، دسته‌بندی خودکار گزارش‌های حوادث دریایی بر حسب نوع رویداد است، با تأکید بر رده‌های نادر و ایمنی‌بحرانی و بدون تولید داده‌ی مصنوعی. دو پرسش اصلی:"));
C.push(NUM("**RQ1:** آیا افزودن لایه‌ی توجه روی خروجی گام‌به‌گام BiLSTM، نسبت به معماری BERT + BiLSTM مقاله‌ی مرجع، کارایی را بهبود می‌دهد؟"));
C.push(NUM("**RQ2:** کدام راهکار مواجهه با نامتوازنی (آنتروپی متقاطع وزن‌دار، زیان کانونی، وزن‌دهی تعداد مؤثر) کارایی روی رده‌های نادر را بیشتر بهبود می‌دهد؟"));
C.push(P("علاوه بر این، طبق وظیفه‌ی جلسه‌ی قبل، **oversampling** داده‌ی آموزش به‌عنوان راهکار سوم مواجهه با نامتوازنی بررسی شد."));

// 3. data
C.push(H1("۳. دادگان"));
C.push(P(`دادگان عمومی MAIB Incident Type (Munaev, 2025؛ مجوز Apache-2.0) شامل ${S.raw_reports.toLocaleString("en")} گزارش کوتاه از سازمان بررسی سوانح دریایی بریتانیا است. برچسب‌ها از فیلد «Main Event L1» خود MAIB آمده‌اند و بازبرچسب‌گذاری دستی نشده‌اند.`));
C.push(H2("۳-۱. پاک‌سازی و انتخاب رده‌ها"));
C.push(BULLET(`گزارش تکراری حذف‌شده: ${S.exact_duplicates_removed}؛ گزارش تکراری با برچسب متناقض: ${S.conflicting_duplicates_removed}.`));
C.push(BULLET(`دو رده با تعداد بسیار کم حذف شدند: Hull Failure (${S.dropped_classes["Hull Failure"]} گزارش) و Non-accidental Event (${S.dropped_classes["Non-accidental Event"]} گزارش). با این تعداد، مجموعه‌ی آزمون ۰ یا ۱ نمونه از آن‌ها داشت و یک پیش‌بینی به‌تنهایی Macro-F1 را چند واحد جابه‌جا می‌کرد. آستانه: حداقل ۲۰ گزارش.`));
C.push(BULLET(`نتیجه: **${S.kept_classes} رده** و **${S.words_per_report.count.toLocaleString("en")} گزارش**؛ نسبت عدم توازن (بزرگ‌ترین به کوچک‌ترین رده) **${S.imbalance_ratio}**؛ میانه‌ی طول گزارش ${S.words_per_report["50%"]} واژه.`));
C.push(BULLET(`تقسیم طبقه‌بندی‌شده (stratified) با بذر ثابت ۴۲: آموزش ${S.split.train.toLocaleString("en")}، اعتبارسنجی ${S.split.val}، آزمون ${S.split.test}. همه‌ی مدل‌ها روی همین یک تقسیم ارزیابی شدند.`));
C.push(table(["رده", "کل گزارش‌ها", "در مجموعه‌ی آزمون", "وضعیت"],
  Object.entries(S.class_counts).map(([k, v]) => [`${CLASS_FA[k] || k}`, String(v), S.test_reports_per_class[k] != null ? String(S.test_reports_per_class[k]) : "—",
    S.dropped_classes[k] ? "حذف شد" : (D.critical.includes(k) ? "ایمنی‌بحرانی" : "")]),
  [3600, 1700, 2100, 2238]));
C.push(SPACER());
C.push(...figure(path.join(REPO, "reports", "class_distribution.png"), "شکل ۱. توزیع رده‌ها (قرمز: ایمنی‌بحرانی، خاکستری: حذف‌شده)", 520));

// 4. method
C.push(H1("۴. روش"));
C.push(H2("۴-۱. مدل‌ها"));
C.push(P("هر سه مدل عصبی رمزگذار یکسان BERT (bert-base-uncased) دارند و فقط در نحوه‌ی تبدیل بردارهای واژه‌ها به یک بردار برای طبقه‌بند تفاوت دارند:"));
C.push(table(["مدل", "شرح", "گام پیشنهاده"], [
  ["TF-IDF + SVM", "بردار وزن واژه‌ها و جفت‌واژه‌ها + ماشین بردار پشتیبان خطی", "۲-الف"],
  ["BERT", "بردار توکن [CLS] + لایه‌ی خطی", "۲-ب"],
  ["BERT + BiLSTM", "BiLSTM دوطرفه (۱۲۸ واحد در هر جهت) روی خروجی BERT؛ حالت پایانی دو جهت به طبقه‌بند می‌رود", "۲-ج (مرجع)"],
  ["BERT + BiLSTM + Attention", "توجه افزایشی روی همه‌ی حالت‌های BiLSTM؛ میانگین وزن‌دار همه‌ی واژه‌ها به طبقه‌بند می‌رود", "۳ (پیشنهادی)"],
], [2600, 5300, 1738]));
C.push(SPACER());
C.push(H2("۴-۲. توابع زیان"));
C.push(table(["نام", "روش", "گام پیشنهاده"], [
  ["ce", "آنتروپی متقاطع معمولی (بدون اقدام)", "مبنا"],
  ["wce", "آنتروپی متقاطع با وزن معکوس بسامد رده", "۴-الف"],
  ["focal", "زیان کانونی (Lin et al., 2017)، γ = 2", "۴-ب"],
  ["cb", "وزن‌دهی تعداد مؤثر نمونه (Cui et al., 2019)، β = 0.999", "۴-ج"],
], [1400, 6400, 1838]));
C.push(P("وزن‌های رده فقط از توزیع داده‌ی آموزش محاسبه می‌شوند.", { after: 160 }));
C.push(H2("۴-۳. Oversampling (درخواست جلسه‌ی قبل)"));
C.push(P("به‌جای تغییر تابع زیان، رده‌های کمیاب در آموزش بیشتر دیده می‌شوند. فقط داده‌ی **آموزش** بازنمونه‌گیری می‌شود و مجموعه‌های اعتبارسنجی و آزمون دست‌نخورده می‌مانند، پس نشت داده رخ نمی‌دهد:"));
C.push(BULLET("**TF-IDF + SVM:** گزارش‌های رده‌های کوچک‌تر به‌صورت تصادفی تکرار می‌شوند تا همه‌ی رده‌ها هم‌اندازه‌ی بزرگ‌ترین رده شوند."));
C.push(BULLET("**BERT + BiLSTM + Attention:** هر گزارش با احتمال متناسب با «۱ تقسیم بر اندازه‌ی رده‌اش» و با جایگذاری انتخاب می‌شود (WeightedRandomSampler). طول هر epoch ثابت می‌ماند، بنابراین تعداد به‌روزرسانی‌ها با سایر اجراها برابر است و مقایسه منصفانه است. هر گزارش رده‌ی واژگونی (حدود ۶۹ گزارش آموزشی) در هر epoch به‌طور میانگین حدود ۶ بار دیده می‌شود."));
C.push(BULLET("oversampling با ce ترکیب شد و نه با wce، زیرا ترکیب هر دو، نامتوازنی را دو بار جبران می‌کند."));
C.push(H2("۴-۴. تنظیمات آموزش و پروتکل ارزیابی"));
C.push(table(["تنظیم", "مقدار"], [
  ["حداکثر طول ورودی", "۱۹۲ توکن (حدود ۲٪ گزارش‌ها بریده می‌شوند)"],
  ["اندازه‌ی دسته", "۸ × ۴ گام انباشت گرادیان = دسته‌ی مؤثر ۳۲ (مطابق مقاله‌ی مرجع)"],
  ["بهینه‌ساز", "AdamW، weight decay = 0.01، warmup ۱۰٪ و کاهش خطی"],
  ["نرخ یادگیری", "BERT: 2e-5؛ لایه‌های جدید (BiLSTM، attention، طبقه‌بند): 1e-3"],
  ["تعداد epoch", "حداکثر ۵، توقف زودهنگام با صبر ۲ epoch روی Macro-F1 اعتبارسنجی"],
  ["دقت محاسبات", "دقت مختلط (fp16)؛ BiLSTM و attention در fp32"],
  ["سخت‌افزار", "GeForce GTX 1650 Ti (4GB)؛ حدود ۷۵ دقیقه برای هر اجرا"],
  ["بذرهای تصادفی", "۲ بذر (1 و 2) برای هر پیکربندی"],
], [2600, 7038]));
C.push(SPACER());
C.push(P("**پروتکل:** بهترین epoch فقط با Macro-F1 مجموعه‌ی اعتبارسنجی انتخاب می‌شود و مجموعه‌ی آزمون فقط یک بار، در پایان هر اجرا، استفاده می‌شود. همه‌ی مدل‌های BERT با تنظیمات کاملاً یکسان آموزش دیدند؛ تنها تفاوت هر دو پیکربندی متوالی یک جزء است (معماری یا تابع زیان). **معیار اصلی** Macro-F1 و بازیابی (recall) هر رده است؛ «بازیابی رده‌های بحرانی» میانگین بازیابی سه رده‌ی آتش‌سوزی، واژگونی و آب‌گرفتگی است. دقت کلی فقط برای مقایسه با ادبیات گزارش می‌شود."));

// 5. results
C.push(H1("۵. نتایج"));
C.push(H2("۵-۱. جدول اصلی"));
C.push(table(["پیکربندی", "بذر", "Macro-F1", "بازیابی رده‌های بحرانی", "دقت کلی"],
  D.configs.map((c) => [FA_NAME[c.name] || c.name, String(c.seeds), pm(c.macro_f1), pm(c.critical_recall), pm(c.accuracy)]),
  [3300, 700, 1900, 2000, 1738]));
C.push(CAPTION("جدول ۱. میانگین ± انحراف معیار روی بذرها، مجموعه‌ی آزمون (۸۶۵ گزارش)"));
C.push(...figure(path.join(DATA_DIR, "fig_results.png"), "شکل ۲. Macro-F1 و بازیابی رده‌های بحرانی (نقطه: میانگین، خط: ± یک انحراف معیار)", 640));

C.push(H2("۵-۲. RQ1: اثر BiLSTM و attention"));
C.push(P(`با تنظیمات یکسان، BERT + BiLSTM نسبت به BERT ساده ${sgn(delta("bert + ce", ref))} و attention نسبت به BiLSTM ${sgn(delta(ref, att))} در Macro-F1 تغییر داد. هر دو تفاوت در حد انحراف معیار بین بذرها (حدود ۰٫۰۰۴ تا ۰٫۰۰۷) است و نمی‌توان آن‌ها را معنادار دانست. با این حال، attention در هر دو بذر بهترین بازیابی را روی سه رده‌ی «برخورد» (Contact، Collision و به گل نشستن) داشت؛ رده‌هایی که از نظر معنایی نزدیک‌اند و تشخیصشان به یک واژه‌ی کلیدی در میانه‌ی متن وابسته است. این الگو با فرضیه‌ی پیشنهاده سازگار است، ولی اندازه‌ی آن کوچک است (حدود ۲ تا ۳ گزارش در رده‌ی Contact).`));

C.push(H2("۵-۳. RQ2: توابع زیان حساس به نامتوازنی"));
C.push(table(["تابع زیان", "تغییر Macro-F1 نسبت به ce", "تغییر بازیابی بحرانی", "حکم"],
  ["wce", "focal", "cb"].filter((l) => has(`bert_bilstm_att + ${l}`)).map((l) => {
    const n = `bert_bilstm_att + ${l}`, d = delta(att, n);
    const noise = Math.max(cfg(att).macro_f1[1], cfg(n).macro_f1[1]);
    return [l, sgn(d), sgn(delta(att, n, "critical_recall")), Math.abs(d) > noise ? "بزرگ‌تر از نوسان بذرها" : "در حد نوسان بذرها"];
  }), [1500, 2900, 2600, 2638]));
C.push(SPACER());
C.push(P(`آنتروپی متقاطع وزن‌دار (wce) بیشترین بهبود را داد (${sgn(delta(att, wce))}) که به‌روشنی از نوسان بین بذرها بزرگ‌تر است. وزن‌دهی تعداد مؤثر (cb) بهبود کوچک‌تری داشت (${sgn(delta(att, "bert_bilstm_att + cb"))}) که فقط اندکی از نوسان بذرها بیشتر است و با ۲ بذر قطعی نیست. با wce دقت کلی هم کاهش نیافت (${f3(cfg(att).accuracy[0])} ← ${f3(cfg(wce).accuracy[0])}). بهبود عمدتاً در رده‌های میانی (Collision و از دست رفتن کنترل) و در رده‌ی واژگونی (حدود ۱ گزارش از ۱۵) است، به قیمت کاهش اندک در پرتکرارترین رده (حدود ۱٫۵ گزارش از ۲۵۰). زیان کانونی در Macro-F1 اثری نداشت.`));

C.push(H2("۵-۴. Oversampling"));
if (haveOS) {
  const rows = [];
  if (has(OS_SVM)) rows.push(["TF-IDF + SVM", f3(cfg("tfidf_svm").macro_f1[0]), f3(cfg(OS_SVM).macro_f1[0]), sgn(delta("tfidf_svm", OS_SVM)), sgn(delta("tfidf_svm", OS_SVM, "critical_recall"))]);
  if (has(OS_BERT)) rows.push(["BERT + BiLSTM + Attention", pm(cfg(att).macro_f1), pm(cfg(OS_BERT).macro_f1), sgn(delta(att, OS_BERT)), sgn(delta(att, OS_BERT, "critical_recall"))]);
  C.push(table(["مدل", "بدون oversampling", "با oversampling", "تغییر Macro-F1", "تغییر بازیابی بحرانی"], rows, [2700, 1800, 1800, 1600, 1738]));
  C.push(SPACER());
  osParagraphs().forEach((t) => C.push(P(t)));
} else {
  C.push(P("این آزمایش پیاده‌سازی شده و در حال اجراست (۲ اجرا، حدود ۲٫۵ ساعت). جدول این بخش پس از پایان اجرا تکمیل می‌شود."));
}

C.push(H2("۵-۵. بازیابی هر رده"));
const showCfg = ["tfidf_svm", OS_SVM, "bert + ce", ref, att, wce, OS_BERT].filter(has);
const byRare = [...D.labels].sort((a, b) => D.support[a] - D.support[b]);
C.push(table(["رده (تعداد در آزمون)", ...showCfg.map((n) => FA_NAME[n].replace(" (بازتولید مرجع)", "").replace("BERT + BiLSTM + Attention", "+ Attention"))],
  byRare.map((c) => [`${CLASS_FA[c]} (${D.support[c]})`, ...showCfg.map((n) => cfg(n).found[c].toFixed(1).replace(".0", ""))]),
  [2838, ...showCfg.map(() => Math.floor(6800 / showCfg.length))]));
C.push(CAPTION("جدول ۲. تعداد گزارش‌های درست‌تشخیص‌داده‌شده‌ی هر رده (میانگین بذرها). نادرترین رده بالاست."));
C.push(P("با ۱۵ گزارش واژگونی در مجموعه‌ی آزمون، تفاوت ۱ تا ۲ گزارش بین مدل‌های BERT (۱۰٫۵ تا ۱۲) قابل اتکا نیست. آنچه قطعی است برتری همه‌ی مدل‌های BERT بر SVM در این رده است (۱۰ از ۱۵)."));
C.push(...figure(path.join(REPO, "reports", "per_class_recall.png"), "شکل ۳. بازیابی هر رده برای هر پیکربندی (نادرترین رده در چپ)", 640));

C.push(H2("۵-۶. روند آموزش"));
C.push(...figure(path.join(DATA_DIR, "fig_learning_curves.png"), "شکل ۴. Macro-F1 اعتبارسنجی در هر epoch (میانگین بذرها)", 520));
C.push(P(`در ${at5} اجرا از ${nRuns} اجرا بهترین epoch همان epoch پنجم (آخر) بود؛ یعنی احتمالاً آموزش طولانی‌تر بهبود اندکی می‌داد. این در محدودیت‌ها آمده است.`));

C.push(H2("۵-۷. رده‌هایی که با هم اشتباه گرفته می‌شوند"));
C.push(table(["رده‌ی واقعی", "پیش‌بینی مدل", "تعداد (جمع بذرها)"],
  D.confusion_pairs.slice(0, 6).map(([a, b, n]) => [CLASS_FA[a], CLASS_FA[b], String(n)]), [3700, 3700, 2238]));
C.push(CAPTION(`جدول ۳. پرتکرارترین خطاها برای بهترین پیکربندی (${FA_NAME[D.best_config]}، ${D.best_cm_seeds} بذر)`));
C.push(...figure(path.join(REPO, "reports", "confusion_matrix.png"), "شکل ۵. ماتریس درهم‌ریختگی بهترین پیکربندی (سطرها نرمال‌شده؛ قطر = بازیابی)", 470));

// 6. attention
C.push(H1("۶. تفسیرپذیری: وزن‌های attention"));
C.push(P("برای هر گزارش آزمون، وزن attention هر واژه ذخیره شد. جدول زیر واژه‌هایی را نشان می‌دهد که در گزارش‌های درست‌تشخیص‌داده‌شده‌ی هر رده بیشترین وزن را گرفته‌اند:"));
const STOP = new Set(["vessel", "resulting", "suffered", "experienced", "damage", "crew", "crewmember", "member", "did", "ship"]);
C.push(table(["رده", "واژه‌های پروزن"], D.labels.map((c) => [CLASS_FA[c], (D.top_words[c] || []).filter((w) => !STOP.has(w)).slice(0, 6).join("، ")]), [3400, 6238]));
C.push(CAPTION("جدول ۴. واژه‌های پروزن هر رده (واژه‌های عمومی مانند vessel حذف شده‌اند)"));
C.push(P("واژه‌ها از نظر معنایی مرتبط‌اند: smoke و extinguish برای آتش‌سوزی، ingress و water برای آب‌گرفتگی، aground و sandbank برای به گل نشستن. **یافته‌ی غیرمنتظره:** برای رده‌ی «آسیب یا از دست رفتن تجهیزات» پروزن‌ترین واژه‌ها solas، ladder، pilot، conform و regulations هستند. یعنی بخش بزرگی از این رده گزارش‌های «عدم انطباق پلکان راهنما با مقررات SOLAS» است؛ زیرگروهی همگن که مدل با چند واژه‌ی ثابت تشخیص می‌دهد. این یافته را تحلیل attention آشکار کرد."));
C.push(P("**احتیاط:** در برخی نمونه‌ها وزن بالایی روی واژه‌های دستوری (a، and) هم دیده می‌شود. وزن attention نشان می‌دهد مدل کجا را نگاه کرده، نه لزوماً چرا تصمیم گرفته است (Jain & Wallace, 2019)؛ تفسیرها با همین قید ارائه می‌شوند."));

// 7. error analysis
C.push(H1("۷. تحلیل خطا"));
C.push(P("بررسی نمونه‌های اشتباه بهترین مدل سه الگوی تکرارشونده را نشان داد:"));
C.push(NUM("**زنجیره‌ی علت ← پیامد:** «propulsion failure and grounded» با برچسب به گل نشستن، به‌عنوان از دست رفتن کنترل پیش‌بینی شد؛ برعکس، «dragged anchor … went aground» با برچسب از دست رفتن کنترل، به گل نشستن پیش‌بینی شد. برچسب MAIB گاهی علت و گاهی پیامد را انتخاب می‌کند.", "num2"));
C.push(NUM("**گزارش‌های چندرویدادی:** «hit the lock side … crewmember fell overboard» با برچسب Contact، آسیب به اشخاص پیش‌بینی شد؛ هر دو برداشت منطقی است.", "num2"));
C.push(NUM("**واژگان تخصصی دریایی:** «girting» (نوعی واژگونی یدک‌کش) و «allided» (برخورد با جسم ثابت) برای BERT عمومی ناآشنا هستند. در مورد دوم، attention واژه‌ی درست را پیدا کرد ولی مدل معنای آن را نمی‌دانست.", "num2"));
C.push(P("**نتیجه:** سقف حدود ۰٫۹۱ بیش از آنکه ناشی از ظرفیت مدل باشد، ناشی از ابهام ذاتی برچسب تک‌کلاسه برای گزارش‌های چندرویدادی است. این توضیح می‌دهد که چرا TF-IDF به BERT نزدیک است و چرا لایه‌های اضافه کمک کمی می‌کنند."));

// 8. reference paper
C.push(H1("۸. مقایسه با مقاله‌ی مرجع (Zhao et al., 2025)"));
C.push(table(["جنبه", "مقاله‌ی مرجع", "پژوهش حاضر"], [
  ["وظیفه", "دسته‌بندی علت حادثه (۳۲ رده)", "دسته‌بندی نوع رویداد (۹ رده)"],
  ["داده", "GISIS و گزارش‌های سواحل چین؛ غیرعمومی", "MAIB؛ عمومی"],
  ["نامتوازنی", "افزایش داده تا حدود ۸۰۰ نمونه در هر رده", "بدون داده‌ی مصنوعی؛ توابع زیان و oversampling"],
  ["تنظیمات مدل‌ها", "dropout، منظم‌سازی، آستانه‌ی گرادیان و تابع فعال‌سازی متفاوت (جدول ۹)", "کاملاً یکسان"],
  ["تکرار", "یک اجرا", "۲ بذر، میانگین ± انحراف معیار"],
  ["خط مبنای کلاسیک", "ندارد", "TF-IDF + SVM"],
  ["معیار اصلی", "دقت کلی", "Macro-F1 و بازیابی هر رده"],
], [2200, 3700, 3738]));
C.push(SPACER());
C.push(P("در جدول ۳ خود مقاله، تغییر هایپرپارامترها دقت مدل BERT + BiLSTM را بین ۰٫۸۷۵ و ۰٫۸۹۸ جابه‌جا می‌کند (۲٫۳ واحد)، که از بهبود ادعاشده‌ی BiLSTM نسبت به BERT (۰٫۸۸۷ ← ۰٫۸۹۸، ۱٫۱ واحد) بزرگ‌تر است. همچنین در شرح روش مشخص نیست که تقسیم داده پیش از افزایش داده انجام شده باشد. در پژوهش حاضر، با تنظیمات یکسان و چند بذر، بهبود BiLSTM مشاهده نشد. به دلیل تفاوت وظیفه و داده، اعداد دو پژوهش مستقیماً قابل مقایسه نیستند."));

// 9. limitations
C.push(H1("۹. محدودیت‌ها"));
for (const t of [
  "فقط ۲ بذر برای هر پیکربندی؛ برآورد انحراف معیار با ۲ نمونه دقیق نیست.",
  "تعداد کم نمونه‌ی آزمون برای رده‌های نادر (۱۵ گزارش واژگونی): هر گزارش ۶٫۷ واحد بازیابی را جابه‌جا می‌کند.",
  `تعداد epoch ثابت ۵؛ در ${at5} اجرا از ${nRuns} اجرا بهترین epoch همان آخری بود.`,
  "پارامترهای γ (focal) و β (cb) تنظیم نشدند و مقادیر پیشنهادی مقالات اصلی استفاده شد.",
  "آزمون معناداری آماری (مثلاً bootstrap جفت‌شده) هنوز انجام نشده است.",
  "دو رده‌ی بسیار کوچک حذف شدند؛ نتایج برای ۹ رده است.",
  "وزن attention توضیح علّی تصمیم مدل نیست.",
]) C.push(BULLET(t));

// 10. next steps
C.push(H1("۱۰. گام‌های بعدی پیشنهادی"));
for (const t of [
  "افزودن بذر سوم (و در صورت امکان ۵ بذر) و آزمون معناداری bootstrap جفت‌شده.",
  "آموزش طولانی‌تر (مثلاً ۸ epoch) برای بررسی اثر محدودیت ۵ epoch.",
  "خط مبنای Logistic Regression و مدل‌های قوی‌تر (RoBERTa، DeBERTa) و مدل تطبیق‌یافته با دامنه‌ی دریایی برای واژگان تخصصی.",
  "مقایسه با مدل‌های زبانی بزرگ به‌صورت zero-shot / few-shot.",
  "بررسی دسته‌بندی چندبرچسبی یا سلسله‌مراتبی برای جدا کردن علت و پیامد در گزارش‌های چندرویدادی.",
]) C.push(BULLET(t));

// appendix
C.push(new Paragraph({ children: [new PageBreak()] }));
C.push(H1("پیوست: جزئیات همه‌ی اجراها"));
C.push(table(["اجرا", "بهترین epoch", "زمان (دقیقه)", "Macro-F1 آزمون", "دقت آزمون"],
  D.per_run.map((r) => [r.run, String(r.best_epoch), String(r.minutes), f3(r.macro_f1), f3(r.accuracy)]),
  [4200, 1300, 1300, 1500, 1338]));
C.push(SPACER());
C.push(P("**تکرارپذیری:** کد کامل، بذرها و پیکربندی‌ها در مخزن پروژه (پوشه‌ی maib-thesis) موجود است. هر گام پیشنهاده یک فایل دارد (step1 تا step6) و همه‌ی تنظیمات در config.py قرار دارند. نتایج هر اجرا، شامل پیش‌بینی‌ها و وزن‌های attention، در پوشه‌ی results ذخیره شده است."));

// ---------------------------------------------------------------- document
const doc = new Document({
  styles: {
    default: { document: { run: { font: { ascii: EN, hAnsi: EN, cs: FA }, size: BODY, sizeComplexScript: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 34, bold: true }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 29, bold: true }, paragraph: { outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.START, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    ...["num", "num2"].map((r) => ({ reference: r, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.START, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] })),
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUTFILE, buf); console.log("wrote", OUTFILE); });
