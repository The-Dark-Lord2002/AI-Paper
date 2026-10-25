// Meeting deck for the supervisor (Persian, RTL). Every number comes from maib-thesis/results via
// docs/paper/tables/numbers.txt, results.tex, perclass.tex and reports/significance.md.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const lu = require("react-icons/lu");
const { applyTheme } = require("/root/.claude/skills/synced/73177168-6492-4975-b72c-25ca276d7775_2a78ec05-25b2-4fe9-98eb-0aab328cc50f/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || path.join(__dirname, "meeting_deck.pptx");
const CM_IMAGE = path.join(__dirname, "cm-1.png");

const THEME = {
  name: "Maritime Navy",
  headFontFace: "Vazirmatn",
  bodyFontFace: "Vazirmatn",
  colors: {
    dk1: "1E2633", lt1: "FFFFFF", dk2: "0E2A47", lt2: "EEF2F6",
    accent1: "1F5F8B", accent2: "B8913A", accent3: "9AA5B1",
    accent4: "B5413F", accent5: "2E8B83", accent6: "5B6B7A",
    hlink: "1F5F8B", folHlink: "5B6B7A",
  },
};
const HEX = THEME.colors;
const MUTED = "5B6B7A";
const FONT = "Vazirmatn";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                       // 13.333 x 7.5 in
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "گزارش پیشرفت پایان‌نامه";
pres.author = "Mohammad Mahdi Arab";
const C = pres.SchemeColor;
const W = 13.333, M = 0.6, CW = W - 2 * M;

// Persian (right-to-left) and English text options. fontFace is named on every run because PowerPoint
// takes Persian glyphs from the complex-script font, which the theme alone does not set.
const fa = (o = {}) => ({ fontFace: FONT, lang: "fa-IR", rtlMode: true, align: "right", ...o });
const en = (o = {}) => ({ fontFace: FONT, lang: "en-US", ...o });

async function icon(Component, color = "FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Component, { color: `#${color}`, size }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

async function waveBackground() {
  const waves = [0, 1, 2, 3, 4].map(i => {
    const y = 760 + i * 55, a = 34 + i * 6;
    return `<path d="M0 ${y} C 320 ${y - a}, 640 ${y + a}, 960 ${y} S 1600 ${y - a}, 1920 ${y}" fill="none" stroke="#FFFFFF" stroke-opacity="${0.05 + i * 0.015}" stroke-width="2"/>`;
  }).join("");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080"><rect width="1920" height="1080" fill="#${HEX.dk2}"/>${waves}</svg>`;
  return "image/png;base64," + (await sharp(Buffer.from(svg)).png().toBuffer()).toString("base64");
}

(async () => {
  const wave = await waveBackground();

  // ------------------------------------------------------------------ layouts
  pres.defineSlideMaster({
    title: "DARK",
    background: { data: wave },
    objects: [],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: "FFFFFF" },
    margin: [0.5, 0.6, 0.7, 0.6],
    objects: [
      { placeholder: { options: fa({ name: "kicker", type: "body", x: M, y: 0.32, w: CW, h: 0.36, fontSize: 13, bold: true, color: C.accent2, valign: "middle", margin: 0 }), text: "" } },
      { placeholder: { options: fa({ name: "title", type: "title", x: M, y: 0.68, w: CW, h: 0.8, fontSize: 30, bold: true, color: C.text2, valign: "middle", margin: 0 }), text: "" } },
      { text: { text: "طبقه‌بندی گزارش‌های حوادث دریایی · گزارش پیشرفت", options: fa({ x: 6.4, y: 6.98, w: 6.33, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 }) } },
    ],
    slideNumber: { x: M, y: 6.95, w: 0.6, h: 0.35, fontFace: FONT, fontSize: 10, color: MUTED },
  });

  const content = (section, kicker, title) => {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
    s.addText(kicker, fa({ placeholder: "kicker" }));
    s.addText(title, fa({ placeholder: "title" }));
    return s;
  };
  const card = (s, x, y, w, h, fill = C.background2, name) =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name });
  const circleIcon = async (s, Comp, x, y, d, fill = C.text2, name) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: name });
    s.addImage({ data: await icon(Comp), x: x + d * 0.24, y: y + d * 0.24, w: d * 0.52, h: d * 0.52, objectName: name + " icon" });
  };
  const chartText = { catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT, dataLabelFontFace: FONT, legendFontFace: FONT,
                      catAxisLabelColor: "1E2633", valAxisLabelColor: MUTED, dataLabelColor: "1E2633",
                      catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 12, legendFontSize: 12 };

  // ------------------------------------------------------------------ 1. title
  pres.addSection({ title: "شروع" });
  {
    const s = pres.addSlide({ masterName: "DARK", sectionTitle: "شروع" });
    s.addText("گزارش پیشرفت پایان‌نامه‌ی کارشناسی ارشد", fa({ x: M, y: 0.9, w: CW, h: 0.45, fontSize: 15, bold: true, color: C.accent2, margin: 0 }));
    s.addText("طبقه‌بندی گزارش‌های حوادث دریایی با مدل‌های زبانی مبتنی بر ترنسفورمر",
      fa({ x: 1.8, y: 1.45, w: W - M - 1.8, h: 1.7, fontSize: 36, bold: true, color: "FFFFFF", valign: "top", margin: 0 }));
    s.addText("Classification of Marine Incident Reports Using Transformer-Based Language Models",
      en({ x: 1.8, y: 3.15, w: W - M - 1.8, h: 0.5, fontSize: 16, color: "C9D6E3", align: "right", margin: 0 }));
    s.addText([
      { text: "دانشجو: محمدمهدی عرب", options: fa({ breakLine: true }) },
      { text: "استاد راهنما: دکتر هدی مشایخی", options: fa({ breakLine: true }) },
      { text: "استاد مشاور: دکتر مریم خدابخش", options: fa() },
    ], { x: 6.0, y: 4.05, w: W - M - 6.0, h: 1.35, fontSize: 16, color: "FFFFFF", paraSpaceAfter: 4, valign: "top", margin: 0 });
    s.addText("دانشکده‌ی مهندسی کامپیوتر · دانشگاه صنعتی شاهرود · آبان ۱۴۰۵",
      fa({ x: M, y: 6.55, w: CW, h: 0.4, fontSize: 12, color: "C9D6E3", margin: 0 }));
    s.addNotes("سلام و تشکر بابت وقت جلسه. در این جلسه سه چیز را نشان می‌دهم: نتایج نهایی آزمایش‌ها، اینکه این نتایج از کجا آمده‌اند و چطور بررسی شده‌اند، و اصلاحاتی که طبق نظر شما در مقاله انجام داده‌ام. در پایان چند سؤال دارم که برای نهایی کردن مقاله به نظر شما نیاز دارد.");
  }

  // ------------------------------------------------------------------ 2. summary
  pres.addSection({ title: "خلاصه" });
  {
    const s = content("خلاصه", "خلاصه", "نتیجه در یک نگاه");
    const stats = [
      { big: "0.915", color: C.accent2, label: "Macro-F1 مدل برتر", sub: "BERT + BiLSTM + Att + wce + LLM augmentation", subEn: true },
      { big: "+0.031", color: C.text2, label: "برتری معنادار بر TF-IDF + SVM", sub: "آزمون بوت‌استرپ جفتی: p = 0.005" },
      { big: "wce", color: C.accent5, label: "منبع اصلی بهبود", sub: "BiLSTM و attention نسبت به BERT ساده بهبود معنادار ندارند" },
    ];
    const cw = 3.85, gap = (CW - 3 * cw) / 2;
    stats.forEach((st, i) => {
      const x = W - M - cw - i * (cw + gap), y = 1.85;               // first card on the right
      card(s, x, y, cw, 3.55, C.background2, `stat ${i + 1}`);
      s.addText(st.big, en({ x, y: y + 0.35, w: cw, h: 1.2, fontSize: 54, bold: true, color: st.color, align: "center", margin: 0 }));
      s.addText(st.label, fa({ x: x + 0.3, y: y + 1.7, w: cw - 0.6, h: 0.55, fontSize: 17, bold: true, color: C.text1, align: "center", margin: 0 }));
      s.addText(st.sub, (st.subEn ? en : fa)({ x: x + 0.3, y: y + 2.3, w: cw - 0.6, h: 0.95, fontSize: 13, color: MUTED, align: "center", valign: "top", margin: 0 }));
    });
    s.addText("پیام اصلی: مدل‌های مبتنی بر BERT از روش کلاسیک بهترند، و برای کلاس‌های نادر، تابع زیان وزن‌دار از معماری پیچیده‌تر مهم‌تر است.",
      fa({ x: M, y: 5.75, w: CW, h: 0.8, fontSize: 16, color: C.text1, margin: 0 }));
    s.addNotes("خلاصه‌ی کل کار در یک اسلاید: بهترین مدل Macro-F1 برابر 0.915 دارد. برتری‌اش بر روش کلاسیک TF-IDF + SVM حدود 0.031 است و از نظر آماری معنادار است. اما صادقانه بگویم: بیشترِ این بهبود از تابع زیان وزن‌دار wce آمده، نه از لایه‌های BiLSTM و attention. این را در مقاله هم صریح نوشته‌ام.");
  }

  // ------------------------------------------------------------------ 3. problem
  pres.addSection({ title: "مسئله و روش" });
  {
    const s = content("مسئله و روش", "مسئله", "کلاس‌های حیاتی، نادرترین کلاس‌ها هستند");
    const points = [
      ["کدگذاری دستی", "کدگذاری گزارش‌ها هنوز دستی، کند و وابسته به کدگذار است."],
      ["عدم‌توازن شدید", "بزرگ‌ترین کلاس حدود 17 برابر کوچک‌ترین کلاس گزارش دارد."],
      ["ریسک کلاس‌های حیاتی", "آتش‌سوزی، آب‌گرفتگی و واژگونی نادرند؛ مدلی که فقط accuracy را بهینه کند آن‌ها را نادیده می‌گیرد."],
    ];
    const icons = [lu.LuFilePen, lu.LuScale, lu.LuTriangleAlert];
    for (let i = 0; i < points.length; i++) {
      const y = 1.85 + i * 1.5;
      await circleIcon(s, icons[i], W - M - 0.75, y, 0.75, i === 2 ? C.accent4 : C.text2, `point ${i + 1}`);
      s.addText(points[i][0], fa({ x: 7.2, y: y - 0.05, w: W - M - 0.95 - 7.2, h: 0.45, fontSize: 18, bold: true, color: C.text1, margin: 0 }));
      s.addText(points[i][1], fa({ x: 7.2, y: y + 0.42, w: W - M - 0.95 - 7.2, h: 0.85, fontSize: 15, color: MUTED, valign: "top", margin: 0 }));
    }
    const classes = ["Capsizing*", "Flooding*", "Fire*", "Collision", "Contact", "Grounding", "Loss of Control", "Damage", "Accident"];
    const counts = [99, 187, 324, 326, 406, 561, 823, 1374, 1662];
    s.addText("تعداد گزارش در هر کلاس (5,762 گزارش MAIB)", fa({ x: M, y: 1.65, w: 6.2, h: 0.4, fontSize: 14, bold: true, color: C.text1, margin: 0 }));
    s.addChart(pres.charts.BAR, [{ name: "Reports", labels: classes, values: counts }], {
      x: M, y: 2.05, w: 6.2, h: 4.55, barDir: "bar", barGapWidthPct: 45,
      chartColors: counts.map((_, i) => (i < 3 ? HEX.accent4 : HEX.accent3)),
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0",
      valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
      valAxisMinVal: 0, valAxisMaxVal: 1900, showLegend: false, ...chartText, objectName: "class distribution",
    });
    s.addText("* کلاس‌های حیاتی", fa({ x: M, y: 6.55, w: 6.2, h: 0.3, fontSize: 11, color: C.accent4, margin: 0 }));
    s.addNotes("مسئله: گزارش‌های حوادث هنوز دستی کدگذاری می‌شوند. داده‌ی ما 5,762 گزارش عمومی MAIB در ۹ کلاس است. کلاس Accident حدود 1,662 گزارش دارد و Capsizing فقط 99 گزارش، یعنی حدود ۱۷ برابر کمتر. سه کلاسی که با قرمز مشخص شده‌اند، یعنی Fire، Flooding و Capsizing، از نظر ایمنی حیاتی‌اند و در عین حال جزو نادرترین‌ها هستند. برای همین معیار اصلی ما Macro-F1 و Recall کلاس‌های حیاتی است، نه فقط accuracy.");
  }

  // ------------------------------------------------------------------ 4. research gap
  {
    const s = content("مسئله و روش", "شکاف پژوهش", "مطالعه‌ی مرجع چه چیزی را باز گذاشته بود");
    const colW = 5.4, xR = W - M - colW, xL = M;
    s.addText("Zhao et al. (2025) · BERT + BiLSTM", en({ x: xR, y: 1.6, w: colW, h: 0.45, fontSize: 16, bold: true, color: C.accent4, align: "right", margin: 0 }));
    s.addText("این پژوهش", fa({ x: xL, y: 1.6, w: colW, h: 0.45, fontSize: 16, bold: true, color: C.accent5, margin: 0 }));
    const rows = [
      ["مدل‌ها با تنظیمات آموزش متفاوت مقایسه شده‌اند", "یک دستورالعمل آموزش یکسان برای همه‌ی مدل‌ها"],
      ["اجرای تکراری گزارش نشده است", "دو seed برای هر پیکربندی و آزمون معناداری"],
      ["معیار اصلی فقط accuracy است", "Macro-F1 و Recall کلاس‌های حیاتی"],
      ["داده‌ها عمومی نیستند", "داده‌ی عمومی MAIB و نتایج قابل بازتولید"],
    ];
    const arrow = await icon(lu.LuArrowLeft, HEX.accent2);
    rows.forEach(([gap, ours], i) => {
      const y = 2.2 + i * 1.1, h = 0.88;
      card(s, xR, y, colW, h, C.background2, `gap ${i + 1}`);
      s.addText(gap, fa({ x: xR + 0.3, y, w: colW - 0.6, h, fontSize: 15, color: C.text1, valign: "middle", margin: 0 }));
      card(s, xL, y, colW, h, "E3F1EF", `answer ${i + 1}`);
      s.addText(ours, fa({ x: xL + 0.3, y, w: colW - 0.6, h, fontSize: 15, bold: true, color: C.text1, valign: "middle", margin: 0 }));
      s.addImage({ data: arrow, x: W / 2 - 0.25, y: y + h / 2 - 0.25, w: 0.5, h: 0.5, objectName: `arrow ${i + 1}` });
    });
    s.addNotes("مطالعه‌ی مرجع ما Zhao و همکاران در سال 2025 است که BERT + BiLSTM را پیشنهاد داده و گفته از BERT ساده بهتر است. ولی چهار مشکل داشت: مدل‌ها با تنظیمات متفاوت آموزش دیده بودند، پس معلوم نیست بهبود از معماری است یا از تنظیمات؛ اجرای تکراری نداشت؛ فقط accuracy گزارش شده بود؛ و داده‌ها عمومی نبودند. ما هر چهار مورد را با طراحی آزمایش کنترل‌شده پاسخ دادیم.");
  }

  // ------------------------------------------------------------------ 5. framework
  {
    const s = content("مسئله و روش", "روش", "چارچوب پیشنهادی");
    const steps = [
      ["Report", "گزارش متنی", "out"], ["BERT", "بازنمایی بافتاری", "core"], ["BiLSTM", "وابستگی ترتیبی", "core"],
      ["Attention", "وزن کلمات مهم", "core"], ["Softmax", "لایه‌ی طبقه‌بندی", "head"], ["9 classes", "نوع حادثه", "out"],
    ];
    const bw = 1.7, gap = (CW - 6 * bw) / 5, by = 2.25, bh = 1.25;
    const x0 = i => M + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x0(1) - 0.15, y: by - 0.6, w: 3 * bw + 2 * gap + 0.3, h: bh + 0.78, rectRadius: 0.12,
      fill: { color: "F6EFDF" }, line: { color: "F6EFDF" }, objectName: "model core" });
    s.addText("هسته‌ی مدل پیشنهادی", fa({ x: x0(1), y: by - 0.52, w: 3 * bw + 2 * gap, h: 0.36, fontSize: 13, bold: true, color: C.accent2, align: "center", margin: 0 }));
    steps.forEach(([name, fa_, kind], i) => {
      const fill = kind === "core" ? HEX.dk2 : kind === "head" ? HEX.accent1 : "FFFFFF";
      const txt = kind === "out" ? HEX.dk2 : "FFFFFF";
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x0(i), y: by, w: bw, h: bh, rectRadius: 0.1,
        fill: { color: fill }, line: { color: HEX.dk2, width: 1.25 }, objectName: `step ${name}` });
      s.addText([
        { text: name, options: en({ fontSize: 17, bold: true, color: txt, align: "center", breakLine: true }) },
        { text: fa_, options: fa({ fontSize: 12, color: txt, align: "center" }) },
      ], { x: x0(i), y: by, w: bw, h: bh, valign: "middle", margin: 0.04 });
      if (i < steps.length - 1)
        s.addShape(pres.shapes.LINE, { x: x0(i) + bw + 0.04, y: by + bh / 2, w: gap - 0.08, h: 0,
          line: { color: HEX.accent6, width: 1.5, endArrowType: "triangle" }, objectName: `flow ${i + 1}` });
    });
    // data-level and loss-level treatments
    const ty = 4.45, th = 1.55;
    card(s, M, ty, 5.6, th, C.background2, "data treatments");
    s.addText([
      { text: "در سطح داده (روی داده‌ی آموزش)", options: fa({ fontSize: 15, bold: true, color: C.text1, breakLine: true }) },
      { text: "Oversampling متوازن کلاس‌ها", options: fa({ fontSize: 13, color: MUTED, breakLine: true }) },
      { text: "افزایش داده با مدل زبانی Qwen2.5-1.5B و فیلتر سازگاری برچسب", options: fa({ fontSize: 13, color: MUTED }) },
    ], { x: M + 0.3, y: ty, w: 5.0, h: th, valign: "middle", margin: 0, paraSpaceAfter: 3 });
    s.addShape(pres.shapes.LINE, { x: x0(0) + bw / 2, y: by + bh + 0.05, w: 0, h: ty - by - bh - 0.05,
      line: { color: HEX.accent6, width: 1.5, dashType: "dash", beginArrowType: "triangle" }, objectName: "data arrow" });
    const lx = x0(4) + bw / 2 - 2.6;
    card(s, lx, ty, 5.2, th, C.background2, "loss functions");
    s.addText([
      { text: "در سطح تابع زیان", options: fa({ fontSize: 15, bold: true, color: C.text1, breakLine: true }) },
      { text: "ce · wce · focal · cb", options: en({ fontSize: 13, color: MUTED, align: "right", breakLine: true }) },
      { text: "wce: وزن‌دهی عکس فراوانی کلاس", options: fa({ fontSize: 13, color: MUTED }) },
    ], { x: lx + 0.3, y: ty, w: 4.6, h: th, valign: "middle", margin: 0, paraSpaceAfter: 3 });
    s.addShape(pres.shapes.LINE, { x: x0(4) + bw / 2, y: by + bh + 0.05, w: 0, h: ty - by - bh - 0.05,
      line: { color: HEX.accent6, width: 1.5, dashType: "dash", beginArrowType: "triangle" }, objectName: "loss arrow" });
    s.addText("مدل‌های پایه: روش کلاسیک TF-IDF + SVM، مدل BERT ساده، و مدل Zhao یعنی BERT + BiLSTM", fa({ x: M, y: 6.25, w: CW, h: 0.4, fontSize: 13, color: MUTED, margin: 0 }));
    s.addNotes("چارچوب: گزارش متنی وارد BERT می‌شود، خروجی BERT به BiLSTM می‌رود، لایه‌ی attention به کلمات مهم وزن می‌دهد و در نهایت softmax یکی از ۹ کلاس را انتخاب می‌کند. برای عدم‌توازن دو دسته روش داریم: در سطح تابع زیان، یعنی ce و wce و focal و cb؛ و در سطح داده، یعنی oversampling و افزایش داده با مدل زبانی Qwen2.5 که گزارش‌های کلاس‌های نادر را بازنویسی می‌کند و یک فیلتر بررسی می‌کند برچسب عوض نشده باشد. مدل‌های پایه هم SVM کلاسیک، BERT ساده و مدل Zhao هستند.");
  }

  // ------------------------------------------------------------------ 6. experiment design
  {
    const s = content("مسئله و روش", "روش", "طراحی آزمایش: مقایسه‌ی منصفانه و تکرارپذیر");
    const items = [
      [lu.LuDatabase, "داده", "تقسیم طبقه‌ای 70/15/15: 4,033 آموزش، 864 اعتبارسنجی، 865 آزمون"],
      [lu.LuRepeat, "تکرار", "12 پیکربندی عصبی × 2 seed و 3 اجرای SVM؛ مجموعاً 27 اجرا"],
      [lu.LuTarget, "معیارها", "Macro-F1 (معیار اصلی)، Macro-Precision و Recall، Accuracy، Recall کلاس‌های حیاتی"],
      [lu.LuSigma, "آزمون آماری", "McNemar دقیق برای هر seed و بوت‌استرپ جفتی روی میانگین دو seed"],
    ];
    const xText = 7.0, colRight = W - M;
    for (let i = 0; i < items.length; i++) {
      const y = 1.75 + i * 1.22;
      await circleIcon(s, items[i][0], colRight - 0.7, y, 0.7, C.text2, `design ${i + 1}`);
      s.addText(items[i][1], fa({ x: xText, y: y - 0.06, w: colRight - 0.9 - xText, h: 0.42, fontSize: 17, bold: true, color: C.text1, margin: 0 }));
      s.addText(items[i][2], fa({ x: xText, y: y + 0.36, w: colRight - 0.9 - xText, h: 0.7, fontSize: 14, color: MUTED, valign: "top", margin: 0 }));
    }
    const head = o => en({ bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, fontSize: 14, ...o });
    const cell = o => en({ fontSize: 13, color: HEX.dk1, ...o });
    const settings = [
      ["Encoder", "bert-base-uncased"], ["Max. length", "192 word-pieces"], ["Effective batch", "32 (8 × 4 accumulation)"],
      ["Optimizer", "AdamW, weight decay 0.01"], ["Learning rate", "2e-5 (BERT), 1e-3 (new layers)"],
      ["Schedule", "10% warm-up, linear decay"], ["Epochs", "max. 5, early stopping (patience 2)"],
      ["Hardware", "GTX 1650 Ti 4 GB, ≈75 min per run"],
    ];
    s.addText("تنظیمات مشترک همه‌ی مدل‌های مبتنی بر BERT", fa({ x: M, y: 1.65, w: 6.0, h: 0.4, fontSize: 14, bold: true, color: C.text1, margin: 0 }));
    s.addTable([
      [{ text: "Setting", options: head() }, { text: "Value", options: head() }],
      ...settings.map(([k, v], i) => [
        { text: k, options: cell({ bold: true, fill: { color: i % 2 ? "FFFFFF" : HEX.lt2 } }) },
        { text: v, options: cell({ fill: { color: i % 2 ? "FFFFFF" : HEX.lt2 } }) },
      ]),
    ], { x: M, y: 2.1, w: 6.0, colW: [1.9, 4.1], rowH: 0.47, border: { type: "none" }, margin: [0, 0.12, 0, 0.12], valign: "middle", objectName: "settings table" });
    s.addNotes("برای اینکه مقایسه منصفانه باشد، همه‌ی مدل‌های BERT با یک دستورالعمل یکسان آموزش دیده‌اند؛ پس هر تفاوتی فقط از معماری، تابع زیان یا داده می‌آید. هر پیکربندی با دو seed آموزش دیده و میانگین و انحراف معیار گزارش شده است. مجموعاً 27 اجرا انجام شده؛ هر اجرای BERT حدود 75 دقیقه روی GPU لپ‌تاپ خودم طول کشیده است. بهترین epoch با داده‌ی اعتبارسنجی انتخاب شده و داده‌ی آزمون فقط یک بار در پایان استفاده شده است.");
  }

  // ------------------------------------------------------------------ 7. main result
  pres.addSection({ title: "نتایج" });
  {
    const s = content("نتایج", "نتایج", "همه‌ی مدل‌های مبتنی بر BERT از baseline بهترند");
    const names = ["TF-IDF + SVM", "BERT", "BERT + BiLSTM", "BERT + BiLSTM + Att", "Att + wce", "Att + wce + LLM augm."];
    const f1 = [0.885, 0.898, 0.901, 0.905, 0.914, 0.915];
    s.addText("Macro-F1 روی 865 گزارش آزمون (میانگین دو seed)", fa({ x: M, y: 1.65, w: 7.3, h: 0.4, fontSize: 14, bold: true, color: C.text1, margin: 0 }));
    s.addChart(pres.charts.BAR, [{ name: "Macro-F1", labels: names, values: f1 }], {
      x: M, y: 2.05, w: 7.3, h: 4.6, barDir: "bar", barGapWidthPct: 40, catAxisOrientation: "maxMin",
      chartColors: [HEX.accent3, HEX.accent1, HEX.accent1, HEX.accent1, HEX.accent1, HEX.accent2],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
      valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1.08, valGridLine: { style: "none" }, catGridLine: { style: "none" },
      showLegend: false, ...chartText, objectName: "macro-F1 chart",
    });
    const notes = [
      ["بهترین میانگین", "Att + wce + LLM augmentation با 0.915 ± 0.002 و accuracy برابر 93.5٪ (SVM: 91.4٪)"],
      ["هم‌تراز با wce", "Att + wce با 0.914 ± 0.006؛ اختلاف 0.001 کمتر از نوسان بین seedهاست"],
      ["سهم معماری", "افزودن BiLSTM و سپس attention هر کدام حدود 0.004؛ در حد نوسان seed"],
    ];
    notes.forEach(([h, t], i) => {
      const y = 1.75 + i * 1.6;
      card(s, 8.3, y, W - M - 8.3, 1.38, i === 0 ? "F6EFDF" : C.background2, `note ${i + 1}`);
      s.addText([
        { text: h, options: fa({ fontSize: 16, bold: true, color: C.text1, breakLine: true }) },
        { text: t, options: fa({ fontSize: 13, color: MUTED }) },
      ], { x: 8.5, y, w: W - M - 8.7, h: 1.38, valign: "middle", margin: 0, paraSpaceAfter: 4 });
    });
    s.addNotes("نتیجه‌ی اصلی: همه‌ی مدل‌های BERT از SVM بهترند. بهترین میانگین متعلق به BERT + BiLSTM + Attention با wce و افزایش داده با LLM است: 0.915. ولی Att + wce بدون افزایش داده 0.914 گرفته؛ این اختلاف از نوسان بین دو seed کمتر است، پس این دو عملاً هم‌ترازند. افزودن BiLSTM و attention هر کدام فقط حدود 0.004 اضافه کرده که در حد نوسان seed است.");
  }

  // ------------------------------------------------------------------ 8. imbalance treatments
  {
    const s = content("نتایج", "نتایج", "تابع زیان وزن‌دار (wce) مؤثرترین روش بود");
    const head = o => fa({ bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, fontSize: 14, align: "center", ...o });
    const rows = [
      ["ce (پایه)", "0.905", "0.845", "—"],
      ["focal", "0.905", "0.858", "0.000"],
      ["cb", "0.910", "0.857", "+0.005"],
      ["oversampling", "0.909", "0.851", "+0.003"],
      ["LLM augmentation", "0.908", "0.862", "+0.003"],
      ["wce", "0.914", "0.868", "+0.009"],
      ["wce + LLM augmentation", "0.915", "0.868", "+0.010"],
    ];
    const table = [[
      { text: "اختلاف با ce", options: head() }, { text: "Recall حیاتی", options: head() },
      { text: "Macro-F1", options: head() }, { text: "روش", options: head() },
    ]];
    rows.forEach((r, i) => {
      const best = i === 6, strong = i >= 5;
      const fill = { color: best ? "F6EFDF" : i % 2 ? "FFFFFF" : HEX.lt2 };
      const num = t => ({ text: "\u200E" + t, options: en({ fontSize: 14, align: "center", bold: strong, color: HEX.dk1, fill }) });
      table.push([num(r[3]), num(r[2]), num(r[1]), { text: r[0], options: fa({ fontSize: 14, bold: strong, color: HEX.dk1, fill }) }]);
    });
    s.addTable(table, { x: 4.35, y: 1.75, w: W - M - 4.35, colW: [2.3, 1.6, 1.5, 2.98], rowH: 0.52, border: { type: "none" },
      margin: [0, 0.15, 0, 0.15], valign: "middle", objectName: "imbalance table" });
    const insights = [
      ["wce", "تغییر یک‌خطی تابع زیان: Macro-F1 +0.009 و Recall حیاتی +0.022"],
      ["oversampling و focal", "تغییر در حد نوسان seed"],
      ["افزایش داده با LLM", "برای SVM مفیدتر (+0.005)؛ روی wce فقط +0.001، ولی پراکندگی seedها را کم کرد (0.002 در برابر 0.006)"],
    ];
    insights.forEach(([h, t], i) => {
      const y = 1.75 + i * 1.55;
      card(s, M, y, 3.45, 1.35, i === 0 ? "E3F1EF" : C.background2, `insight ${i + 1}`);
      s.addText([
        { text: h, options: fa({ fontSize: 15, bold: true, color: C.text1, breakLine: true }) },
        { text: t, options: fa({ fontSize: 12, color: MUTED }) },
      ], { x: M + 0.2, y, w: 3.05, h: 1.35, valign: "middle", margin: 0, paraSpaceAfter: 3 });
    });
    s.addText("همه‌ی روش‌ها روی مدل BERT + BiLSTM + Att. Recall حیاتی: میانگین Recall سه کلاس Fire، Flooding و Capsizing", fa({ x: 4.35, y: 6.1, w: W - M - 4.35, h: 0.35, fontSize: 11, color: MUTED, margin: 0 }));
    s.addNotes("از بین روش‌های مقابله با عدم‌توازن، wce بیشترین اثر را داشت: Macro-F1 را 0.009 و Recall کلاس‌های حیاتی را 0.022 بالا برد، فقط با یک تغییر یک‌خطی در تابع زیان. focal و oversampling در حد نوسان seed بودند. افزایش داده با LLM برای SVM مفیدتر بود، چون SVM از واژگان جدید سود می‌برد، ولی BERT از قبل با پیش‌آموزش بازنویسی‌ها را می‌فهمد. با این حال روی wce پراکندگی بین seedها را کم کرد.");
  }

  // ------------------------------------------------------------------ 9. critical classes
  {
    const s = content("نتایج", "نتایج", "عملکرد روی کلاس‌های حیاتی");
    const cats = ["Capsizing (15)", "Flooding (28)", "Fire (48)"];
    s.addText("F1 هر کلاس حیاتی روی داده‌ی آزمون (در پرانتز: تعداد گزارش آزمون)", fa({ x: M, y: 1.65, w: 7.4, h: 0.4, fontSize: 14, bold: true, color: C.text1, margin: 0 }));
    s.addChart(pres.charts.BAR, [
      { name: "TF-IDF + SVM", labels: cats, values: [0.800, 0.828, 0.948] },
      { name: "Att (ce)", labels: cats, values: [0.823, 0.865, 0.969] },
      { name: "Att + wce", labels: cats, values: [0.868, 0.881, 0.969] },
      { name: "Att + wce + LLM", labels: cats, values: [0.868, 0.874, 0.969] },
    ], {
      x: M, y: 2.05, w: 7.4, h: 4.6, barDir: "col", barGapWidthPct: 55, barGrouping: "clustered",
      chartColors: [HEX.accent3, HEX.accent1, HEX.accent5, HEX.accent2],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 10,
      valAxisMinVal: 0, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00",
      valGridLine: { color: "E3E7EC", size: 0.5 }, catGridLine: { style: "none" },
      showLegend: true, legendPos: "b", ...chartText, dataLabelFontSize: 10, objectName: "critical classes chart",
    });
    const stats = [["0.827 → 0.868", "Recall میانگین کلاس‌های حیاتی: SVM ← مدل برتر"], ["0.823 → 0.868", "F1 کلاس Capsizing با افزودن wce"]];
    stats.forEach(([big, label], i) => {
      const y = 1.75 + i * 1.6;
      s.addText(big, en({ x: 8.4, y, w: W - M - 8.4, h: 0.75, fontSize: 32, bold: true, color: C.text2, align: "right", margin: 0 }));
      s.addText(label, fa({ x: 8.4, y: y + 0.75, w: W - M - 8.4, h: 0.5, fontSize: 14, color: MUTED, margin: 0 }));
    });
    card(s, 8.4, 5.0, W - M - 8.4, 1.6, "FBEDEC", "honest note");
    s.addText([
      { text: "نکته‌ی صادقانه", options: fa({ fontSize: 14, bold: true, color: C.accent4, breakLine: true }) },
      { text: "BERT ساده بالاترین Recall حیاتی را دارد (0.873). تفاوت‌ها در حد نوسان seed است و فقط 15 گزارش Capsizing در آزمون هست.", options: fa({ fontSize: 12, color: C.text1 }) },
    ], { x: 8.6, y: 5.0, w: W - M - 8.8, h: 1.6, valign: "middle", margin: 0, paraSpaceAfter: 3 });
    s.addNotes("روی کلاس‌های حیاتی: wce مهم‌ترین اثر را داشت. F1 کلاس Capsizing از 0.823 به 0.868 رسید و میانگین Recall سه کلاس حیاتی نسبت به SVM از 0.827 به 0.868 رسید. یک نکته را هم صادقانه بگویم: BERT ساده بالاترین Recall حیاتی را دارد، یعنی 0.873، ولی این تفاوت‌ها در حد نوسان seed است؛ کلاس Capsizing فقط 15 گزارش آزمون دارد و یک گزارش درست یا غلط عدد را حدود 0.07 جابه‌جا می‌کند.");
  }

  // ------------------------------------------------------------------ 10. significance
  {
    const s = content("نتایج", "نتایج", "کدام تفاوت‌ها از نظر آماری معنادارند؟");
    const head = o => fa({ bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, fontSize: 14, align: "center", ...o });
    const rows = [
      ["مدل برتر در برابر SVM", "+0.031", "[0.009, 0.056]", "0.005", true],
      ["Att + wce در برابر SVM", "+0.030", "[0.007, 0.059]", "0.005", true],
      ["BERT ساده در برابر SVM", "+0.013", "[−0.011, 0.039]", "0.272", false],
      ["Att + wce در برابر Att (ce)", "+0.009", "[−0.003, 0.022]", "0.142", false],
      ["BERT + BiLSTM در برابر BERT", "+0.004", "[−0.011, 0.018]", "0.575", false],
      ["مدل برتر در برابر Att + wce", "+0.001", "[−0.007, 0.008]", "0.839", false],
    ];
    const table = [[
      { text: "نتیجه", options: head() }, { text: "p", options: head() }, { text: "بازه‌ی اطمینان 95٪", options: head() },
      { text: "اختلاف Macro-F1", options: head() }, { text: "مقایسه", options: head() },
    ]];
    rows.forEach(([cmp, d, ci, p, sig], i) => {
      const fill = { color: i % 2 ? "FFFFFF" : HEX.lt2 };
      const num = t => ({ text: "\u200E" + t, options: en({ fontSize: 14, align: "center", color: HEX.dk1, fill }) });
      table.push([
        { text: sig ? "معنادار" : "معنادار نیست", options: fa({ fontSize: 14, bold: true, align: "center", color: sig ? HEX.accent5 : HEX.accent6, fill }) },
        num(p), num(ci), num(d),
        { text: cmp, options: fa({ fontSize: 14, color: HEX.dk1, fill, bold: sig }) },
      ]);
    });
    s.addTable(table, { x: M, y: 1.75, w: CW, colW: [2.2, 1.3, 2.6, 2.2, 3.83], rowH: 0.55, border: { type: "none" },
      margin: [0, 0.15, 0, 0.15], valign: "middle", objectName: "significance table" });
    s.addText("بوت‌استرپ جفتی روی میانگین دو seed (2000 بازنمونه‌گیری). آزمون McNemar برای مدل برتر در برابر SVM: p = 0.005 و p = 0.016 در دو seed.",
      fa({ x: M, y: 5.75, w: CW, h: 0.4, fontSize: 12, color: MUTED, margin: 0 }));
    s.addText("جمع‌بندی: فقط برتری مدل‌های دارای wce بر روش کلاسیک معنادار است؛ تفاوت‌های درون خانواده‌ی BERT معنادار نیستند.",
      fa({ x: M, y: 6.2, w: CW, h: 0.45, fontSize: 15, bold: true, color: C.text1, margin: 0 }));
    s.addNotes("این اسلاید مهم‌ترین اسلاید از نظر علمی است. برتری مدل برتر بر SVM معنادار است: اختلاف 0.031 با بازه‌ی اطمینان 0.009 تا 0.056 و p برابر 0.005؛ آزمون McNemar هم در هر دو seed معنادار است. اما BERT ساده به‌تنهایی برتری معناداری بر SVM ندارد (p = 0.27) و هیچ‌کدام از تفاوت‌های درون خانواده‌ی BERT معنادار نیستند. یعنی نمی‌توانیم بگوییم BiLSTM یا attention لازم است؛ چیزی که فرق ایجاد کرده wce است. این را صریح در مقاله نوشته‌ام.");
  }

  // ------------------------------------------------------------------ 11. error analysis
  pres.addSection({ title: "تحلیل" });
  {
    const s = content("تحلیل", "تحلیل خطا", "خطاها بیشتر بین علت و پیامد هستند");
    s.addImage({ path: CM_IMAGE, x: M, y: 1.6, w: 5.25, h: 4.95, objectName: "confusion matrix" });
    const head = o => en({ bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, fontSize: 13, align: "center", ...o });
    const pairs = [["Damage", "Loss of Control", "17"], ["Damage", "Contact", "8"], ["Contact", "Collision", "8"],
                   ["Loss of Control", "Damage", "7"], ["Grounding", "Loss of Control", "6"]];
    const x = 6.4, w = W - M - x;
    s.addText("پرتکرارترین خطاها (مدل برتر، دو seed، 112 خطا از 1,730 پیش‌بینی)", fa({ x, y: 1.6, w, h: 0.4, fontSize: 14, bold: true, color: C.text1, margin: 0 }));
    s.addTable([
      [{ text: "True class", options: head() }, { text: "Predicted as", options: head() }, { text: "Count", options: head() }],
      ...pairs.map((r, i) => r.map(t => ({ text: t, options: en({ fontSize: 13, align: "center", color: HEX.dk1, fill: { color: i % 2 ? "FFFFFF" : HEX.lt2 } }) }))),
    ], { x, y: 2.05, w, colW: [2.4, 2.4, w - 4.8], rowH: 0.42, border: { type: "none" }, margin: [0, 0.1, 0, 0.1], valign: "middle", objectName: "confusion pairs" });
    card(s, x, 4.8, w, 1.75, "F6EFDF", "error insight");
    s.addText([
      { text: "علت یا پیامد؟", options: fa({ fontSize: 15, bold: true, color: C.text1, breakLine: true }) },
      { text: "نقص پیشرانش که به گیر کردن کشتی منجر می‌شود با هر دو برچسب قابل توصیف است، ولی هر گزارش فقط یک برچسب دارد. پس سقف حدود 0.91 احتمالاً از برچسب‌گذاری تک‌برچسبی می‌آید، نه از ظرفیت مدل.", options: fa({ fontSize: 13, color: C.text1 }) },
    ], { x: x + 0.25, y: 4.8, w: w - 0.5, h: 1.75, valign: "middle", margin: 0, paraSpaceAfter: 3 });
    s.addNotes("ماتریس درهم‌ریختگی مدل برتر را می‌بینید. بیشترین خطا بین Damage و Loss of Control است: 17 گزارش Damage به‌عنوان Loss of Control پیش‌بینی شده‌اند. بیشتر این خطاها جفت علت و پیامدند؛ مثلاً نقص پیشرانش که بعد به گیر کردن کشتی منجر می‌شود. چون هر گزارش فقط یک برچسب دارد، مدل نمی‌تواند هر دو را بگوید. Contact و Collision هم ذاتاً نزدیک‌اند، چون هر دو برخورد کشتی با چیزی هستند. نتیجه اینکه برای بهبود بیشتر، برچسب‌گذاری چندبرچسبی احتمالاً مؤثرتر از معماری پیچیده‌تر است.");
  }

  // ------------------------------------------------------------------ 12. attention
  {
    const s = content("تحلیل", "تحلیل توجه", "لایه‌ی attention یک زیرگروه پنهان را نشان داد");
    const head = o => en({ bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, fontSize: 13, ...o });
    const words = [
      ["Accident", "passenger, fell, ship, cruise, sustained"], ["Capsizing", "capsized, boat, FRC, rescue, water"],
      ["Collision", "pilot, manoeuvring, collided, tanker, moored"], ["Contact", "contact, quayside, manoeuvring, cargo, lock"],
      ["Damage", "SOLAS, ladder, pilot, regulations, conform"], ["Fire", "engine, extinguish, caught, extinguished, smoke"],
      ["Flooding", "water, ingress, taking, started, motor"], ["Grounding", "grounded, aground, ran, water, rescue"],
      ["Loss of Control", "failure, engine, fishing, towed, fouled"],
    ];
    const tw = 7.3, tx = W - M - tw;
    s.addTable([
      [{ text: "Class", options: head() }, { text: "Top attention words (best model, seed 1)", options: head() }],
      ...words.map(([c, w], i) => {
        const hl = c === "Damage";
        const fill = { color: hl ? "F6EFDF" : i % 2 ? "FFFFFF" : HEX.lt2 };
        return [{ text: c, options: en({ fontSize: 13, bold: true, color: HEX.dk1, fill }) },
                { text: w, options: en({ fontSize: 13, bold: hl, color: hl ? HEX.accent2 : HEX.dk1, fill }) }];
      }),
    ], { x: tx, y: 1.7, w: tw, colW: [1.9, tw - 1.9], rowH: 0.47, border: { type: "none" }, margin: [0, 0.12, 0, 0.12], valign: "middle", objectName: "attention table" });
    const iw = tx - M - 0.4;
    card(s, M, 1.7, iw, 2.65, "F6EFDF", "attention insight");
    s.addText([
      { text: "یافته", options: fa({ fontSize: 15, bold: true, color: C.accent2, breakLine: true }) },
      { text: "در کلاس Damage کلمات SOLAS، ladder و pilot بالاترین وزن را دارند: زیرگروهی تکرارشونده از گزارش‌های «نردبان راهنمای نامطابق با SOLAS» درون یک کلاس کلی.", options: fa({ fontSize: 13, color: C.text1 }) },
    ], { x: M + 0.25, y: 1.7, w: iw - 0.5, h: 2.65, valign: "middle", margin: 0, paraSpaceAfter: 3 });
    card(s, M, 4.6, iw, 1.6, C.background2, "attention caution");
    s.addText([
      { text: "احتیاط", options: fa({ fontSize: 14, bold: true, color: C.text1, breakLine: true }) },
      { text: "وزن توجه نشان می‌دهد مدل به کجا نگاه کرده، نه اینکه چرا تصمیم گرفته.", options: fa({ fontSize: 13, color: MUTED }) },
    ], { x: M + 0.25, y: 4.6, w: iw - 0.5, h: 1.6, valign: "middle", margin: 0, paraSpaceAfter: 3 });
    s.addText("کلمات عمومی (vessel، crew، damage و …) حذف شده‌اند. FRC: قایق نجات سریع.", fa({ x: tx, y: 6.45, w: tw, h: 0.3, fontSize: 11, color: MUTED, margin: 0 }));
    s.addNotes("این جدول از وزن‌های attention مدل برتر ساخته شده و نشان می‌دهد مدل برای هر کلاس بیشتر به چه کلماتی توجه کرده. بیشترشان همان کلمات مورد انتظارند، مثل capsized برای واژگونی یا ingress برای آب‌گرفتگی. اما در کلاس Damage، کلمات SOLAS، ladder و pilot بالای لیست‌اند؛ یعنی درون این کلاس کلی، یک زیرگروه تکرارشونده از گزارش‌های نردبان راهنمای غیراستاندارد وجود دارد. این برای تحلیلگران ایمنی کاربرد عملی دارد. البته با احتیاط: وزن توجه نشان می‌دهد مدل کجا را نگاه کرده، نه دلیل تصمیمش را.");
  }

  // ------------------------------------------------------------------ 13. paper revisions
  pres.addSection({ title: "مقاله" });
  {
    const s = content("مقاله", "مقاله", "اصلاحات انجام‌شده طبق نظر شما");
    const items = [
      "عنوان جدید مقاله", "چکیده به سبک مقالات مرجع، بدون نتایج عددی", "حذف سؤال‌های پژوهش (RQ)",
      "مقدمه با سه چالش اصلی و مرجع", "گسترش کارهای مرتبط در دو زیربخش", "توضیح ۳ تا ۴ خطی برای هر فرمول",
      "شکل ۱: یک pipeline با جعبه‌ی جدای تابع زیان", "بخش Evaluation: داده، تنظیمات، baseline، معیارها، بحث",
      "Precision و Recall کلی و جدول F1 هر کلاس", "بهترین پررنگ، دومی زیرخط‌دار؛ کپشن بدون ایتالیک",
      "بازبینی همه‌ی اعداد و اصلاح مدل برتر در کل متن", "۶ صفحه، بدون مرجع کنفرانسی، برچسب Keywords",
    ];
    const check = await icon(lu.LuCheck, "FFFFFF");
    const colW = (CW - 0.5) / 2;
    items.forEach((t, i) => {
      const col = i < 6 ? 0 : 1, row = i % 6;
      const xCol = W - M - colW - col * (colW + 0.5), y = 1.75 + row * 0.8;
      s.addShape(pres.shapes.OVAL, { x: xCol + colW - 0.48, y: y + 0.06, w: 0.48, h: 0.48, fill: { color: HEX.accent5 }, line: { color: HEX.accent5 }, objectName: `check ${i + 1}` });
      s.addImage({ data: check, x: xCol + colW - 0.48 + 0.11, y: y + 0.17, w: 0.26, h: 0.26, objectName: `check icon ${i + 1}` });
      s.addText(t, fa({ x: xCol, y, w: colW - 0.65, h: 0.6, fontSize: 15, color: C.text1, valign: "middle", margin: 0 }));
    });
    s.addText("نسخه‌ی کامل در PR شماره‌ی 18 مخزن GitHub پروژه", fa({ x: M, y: 6.6, w: CW, h: 0.3, fontSize: 11, color: MUTED, margin: 0 }));
    s.addNotes("همه‌ی نکاتی که روی نسخه‌ی قبلی نوشته بودید اعمال شده است: عنوان، چکیده، حذف سؤال‌های پژوهش، مقدمه و کارهای مرتبط، توضیح فرمول‌ها، شکل یک، بخش Evaluation، Precision و Recall کلی، جدول هر کلاس و قالب جدول‌ها. مهم‌تر از همه، همه‌ی اعداد را دوباره بررسی کردم و اشتباه قبلی متن را که مدل دیگری را بهترین معرفی کرده بود اصلاح کردم. مقاله ۶ صفحه است و مرجع کنفرانسی ندارد.");
  }

  // ------------------------------------------------------------------ 14. validity of the numbers
  {
    const s = content("مقاله", "اعتبار نتایج", "همه‌ی اعداد از اجرای واقعی آزمایش‌ها آمده‌اند");
    const stats = [
      ["27", "اجرای مستقل، هر کدام با فایل نتیجه‌ی جداگانه"],
      ["865", "پیش‌بینی ذخیره‌شده برای هر اجرا، گزارش‌به‌گزارش"],
      ["31 h", "زمان کل آموزش روی GTX 1650 Ti (حدود 75 دقیقه هر اجرا)"],
      ["27 / 27", "فایل با بازمحاسبه‌ی Macro-F1 و accuracy از ماتریس درهم‌ریختگی سازگار است"],
    ];
    const cw = (CW - 3 * 0.35) / 4;
    stats.forEach(([big, label], i) => {
      const x = W - M - cw - i * (cw + 0.35), y = 1.75;
      card(s, x, y, cw, 2.3, C.background2, `evidence ${i + 1}`);
      s.addText(big, en({ x, y: y + 0.25, w: cw, h: 0.9, fontSize: 38, bold: true, color: C.text2, align: "center", margin: 0 }));
      s.addText(label, fa({ x: x + 0.2, y: y + 1.2, w: cw - 0.4, h: 0.95, fontSize: 13, color: MUTED, align: "center", valign: "top", margin: 0 }));
    });
    const lines = [
      [lu.LuFileCode, "جدول‌های مقاله مستقیماً با کد (make_tables.py) از همین فایل‌ها ساخته می‌شوند؛ هیچ عددی دستی تایپ نشده است."],
      [lu.LuGitCommitHorizontal, "فایل‌های نتیجه بین ۱۹ تا ۲۲ مهر با حساب GitHub خودم ثبت شده‌اند، همراه با loss هر epoch و زمان آموزش."],
      [lu.LuPlay, "امکان اجرای زنده در جلسه: baseline (TF-IDF + SVM) در حدود یک دقیقه دوباره اجرا می‌شود."],
    ];
    for (let i = 0; i < lines.length; i++) {
      const y = 4.4 + i * 0.75;
      await circleIcon(s, lines[i][0], W - M - 0.55, y, 0.55, C.accent1, `proof ${i + 1}`);
      s.addText(lines[i][1], fa({ x: M, y: y - 0.02, w: CW - 0.75, h: 0.6, fontSize: 14, color: C.text1, valign: "middle", margin: 0 }));
    }
    s.addNotes("درباره‌ی اعتبار اعداد: همه‌ی 27 اجرا را خودم روی لپ‌تاپ اجرا کرده‌ام؛ حدود 31 ساعت آموزش. برای هر اجرا یک فایل نتیجه هست که 865 پیش‌بینی، loss هر epoch و زمان آموزش را دارد. Macro-F1 و accuracy همه‌ی فایل‌ها را از ماتریس درهم‌ریختگی دوباره حساب کرده‌ام و همه سازگارند. جدول‌های مقاله با یک اسکریپت مستقیماً از همین فایل‌ها ساخته می‌شوند. اگر بخواهید، همین الان baseline را جلوی شما اجرا می‌کنم. (قبل از جلسه یک بار step1 و step2 را اجرا کن تا مطمئن شوی کار می‌کند.)");
  }

  // ------------------------------------------------------------------ 15. decisions
  pres.addSection({ title: "جمع‌بندی" });
  {
    const s = pres.addSlide({ masterName: "DARK", sectionTitle: "جمع‌بندی" });
    s.addText("پرسش‌ها", fa({ x: M, y: 0.6, w: CW, h: 0.4, fontSize: 14, bold: true, color: C.accent2, margin: 0 }));
    s.addText("تصمیم‌هایی که به نظر شما نیاز دارد", fa({ x: M, y: 1.0, w: CW, h: 0.8, fontSize: 32, bold: true, color: "FFFFFF", margin: 0 }));
    const qs = [
      "آیا ساختار جدید چکیده مورد تأیید است؟",
      "ارجاع به مقاله‌ی اصلی BERT (کنفرانسی) و preprintهای arXiv مجاز است؟",
      "ترتیب نویسندگان و محل انتشار هدف چه باشد؟",
      "کد در GitHub منتشر و لینک آن در مقاله درج شود؟",
    ];
    qs.forEach((q, i) => {
      const y = 2.15 + i * 0.95;
      s.addShape(pres.shapes.OVAL, { x: W - M - 0.62, y, w: 0.62, h: 0.62, fill: { color: HEX.accent2 }, line: { color: HEX.accent2 }, objectName: `question ${i + 1}` });
      s.addText(String(i + 1), en({ x: W - M - 0.62, y, w: 0.62, h: 0.62, fontSize: 18, bold: true, color: HEX.dk2, align: "center", valign: "middle", margin: 0 }));
      s.addText(q, fa({ x: 4.9, y, w: W - M - 0.85 - 4.9, h: 0.62, fontSize: 18, color: "FFFFFF", valign: "middle", margin: 0 }));
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 2.15, w: 3.9, h: 3.5, rectRadius: 0.12, fill: { color: "FFFFFF", transparency: 90 }, line: { color: "C9D6E3", width: 0.75 }, objectName: "next steps" });
    s.addText([
      { text: "گام‌های بعدی", options: fa({ fontSize: 16, bold: true, color: "FFFFFF", breakLine: true }) },
      { text: "نهایی‌سازی مقاله با نظر شما", options: fa({ fontSize: 14, color: "C9D6E3", bullet: true, breakLine: true }) },
      { text: "در صورت صلاحدید: seed سوم برای تقویت آزمون‌ها", options: fa({ fontSize: 14, color: "C9D6E3", bullet: true, breakLine: true }) },
      { text: "نگارش فصل‌های پایان‌نامه", options: fa({ fontSize: 14, color: "C9D6E3", bullet: true }) },
    ], { x: M + 0.25, y: 2.15, w: 3.4, h: 3.5, valign: "middle", margin: 0, paraSpaceAfter: 8 });
    s.addText("سپاس از راهنمایی‌های شما", fa({ x: M, y: 6.45, w: CW, h: 0.45, fontSize: 16, color: "C9D6E3", margin: 0 }));
    s.addNotes("برای نهایی کردن مقاله به نظر شما در چهار مورد نیاز دارم: تأیید ساختار چکیده؛ اینکه به مقاله‌ی اصلی BERT که کنفرانسی است و به چند preprint از arXiv می‌توانم ارجاع بدهم یا نه؛ ترتیب نویسندگان و محل انتشار؛ و اینکه کد را در GitHub منتشر کنیم. اگر صلاح بدانید، می‌توانم یک seed سوم هم اجرا کنم تا آزمون‌های آماری قوی‌تر شوند. ممنون از وقتتان.");
  }

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch(e => { console.error(e); process.exit(1); });
