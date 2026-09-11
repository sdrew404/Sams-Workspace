#!/usr/bin/env python3
"""Generate dashboard.html — a self-contained interactive results dashboard.

Reads results/sampled_reviews.csv and embeds all data + computed metrics
into a single HTML file with zero external dependencies.
"""
import csv
import json
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "results" / "sampled_reviews.csv"
OUT_PATH = BASE / "dashboard.html"

CLASSES = ["positive", "neutral", "negative"]
CLASS_COLORS = {"positive": "#16a34a", "neutral": "#d97706", "negative": "#dc2626"}


def compute_metrics(rows):
    n = len(rows)
    correct = sum(r["true_label"] == r["pred_label"] for r in rows)
    accuracy = correct / n if n else 0.0
    per_class = {}
    for cls in CLASSES:
        tp = sum(r["true_label"] == cls and r["pred_label"] == cls for r in rows)
        fp = sum(r["true_label"] != cls and r["pred_label"] == cls for r in rows)
        fn = sum(r["true_label"] == cls and r["pred_label"] != cls for r in rows)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[cls] = {"n": sum(r["true_label"] == cls for r in rows),
                          "precision": prec, "recall": rec, "f1": f1}
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
    confusion = [[sum(r["true_label"] == a and r["pred_label"] == b for r in rows)
                  for b in CLASSES] for a in CLASSES]
    return accuracy, macro_f1, per_class, confusion


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        raw_rows = list(csv.DictReader(fh))

    reviews = [{
        "idx": int(r["idx"]),
        "user_id": r["user_id"],
        "asin": r["asin"],
        "rating": float(r["rating"]),
        "title": r["title"],
        "text": r["text"],
        "true_label": r["true_label"],
        "pred_label": r["predicted_label"],
        "raw": r["raw_response"],
    } for r in raw_rows]

    accuracy, macro_f1, per_class, confusion = compute_metrics(reviews)
    true_dist = dict(Counter(r["true_label"] for r in reviews))
    pred_dist = dict(Counter(r["pred_label"] for r in reviews))
    n_wrong = sum(r["true_label"] != r["pred_label"] for r in reviews)

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "meta": {
            "endpoint": "http://dobolyi.com:9000/v1",
            "model": "DeepSeek-V4-Flash-0731",
            "sample": len(reviews),
            "seed": 42,
            "source": "Amazon Reviews 2023 — Gift Cards (152,410 reviews total)",
        },
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "n_wrong": n_wrong,
            "per_class": per_class,
            "confusion": confusion,
            "true_dist": true_dist,
            "pred_dist": pred_dist,
        },
        "reviews": reviews,
    }

    html = TEMPLATE.replace("@@DATA_JSON@@", json.dumps(payload, ensure_ascii=False))
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB, "
          f"{len(reviews)} reviews embedded)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentiment Classifier — Results Dashboard</title>
<style>
  :root{
    --bg:#f6f7f9; --card:#ffffff; --border:#e3e6ea; --text:#111827; --muted:#6b7280;
    --accent:#4f46e5; --good:#16a34a; --warn:#d97706; --bad:#dc2626; --chip:#eef1f5;
  }
  *{box-sizing:border-box; margin:0; padding:0}
  body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       background:var(--bg); color:var(--text); padding:28px 20px 80px}
  .wrap{max-width:1080px; margin:0 auto}
  header h1{font-size:22px; font-weight:700}
  header .meta{color:var(--muted); margin-top:4px; font-size:13px}
  header .meta b{color:var(--text); font-weight:600}

  .kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
        gap:12px; margin:20px 0}
  .kpi{background:var(--card); border:1px solid var(--border); border-radius:12px;
       padding:14px 16px}
  .kpi .v{font-size:26px; font-weight:700}
  .kpi .l{color:var(--muted); font-size:12px; margin-top:2px}
  .kpi .s{font-size:12px; margin-top:4px}

  .cols{display:grid; grid-template-columns: 5fr 7fr; gap:16px; margin-bottom:20px}
  @media(max-width:860px){ .cols{grid-template-columns:1fr} }
  .panel{background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px}
  .panel h2{font-size:13px; text-transform:uppercase; letter-spacing:.05em;
            color:var(--muted); margin-bottom:12px}

  table.cm{border-collapse:collapse; width:100%}
  table.cm th, table.cm td{border:1px solid var(--border); padding:8px 10px;
        text-align:center; font-size:13px}
  table.cm th{color:var(--muted); font-weight:600; background:#fafbfc}
  table.cm td.num{font-weight:700}
  table.cm th.rowlabel{text-align:left; text-transform:capitalize}
  table.cm .note{font-size:11px; color:var(--muted); margin-top:10px}

  table.pm{width:100%; border-collapse:collapse}
  table.pm th, table.pm td{padding:7px 10px; text-align:right; font-size:13px;
        border-bottom:1px solid var(--border)}
  table.pm th:first-child, table.pm td:first-child{text-align:left}
  table.pm th{color:var(--muted); font-weight:600}
  table.pm .dot{display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px}

  .bar{height:8px; background:var(--chip); border-radius:4px; overflow:hidden; margin-top:3px}
  .bar i{display:block; height:100%; border-radius:4px}

  .filters{position:sticky; top:10px; z-index:5; background:var(--card);
        border:1px solid var(--border); border-radius:12px; padding:12px 14px;
        display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:16px;
        box-shadow:0 2px 8px rgba(0,0,0,.04)}
  .filters input[type=search]{flex:1 1 220px; min-width:160px; padding:7px 10px;
        border:1px solid var(--border); border-radius:8px; font:inherit; background:#fff}
  .filters select{padding:7px 8px; border:1px solid var(--border); border-radius:8px;
        font:inherit; background:#fff}
  .filters .count{margin-left:auto; color:var(--muted); font-size:13px; white-space:nowrap}
  .filters button{background:none; border:1px solid var(--border); border-radius:8px;
        padding:7px 10px; font:inherit; cursor:pointer; color:var(--muted)}
  .filters button:hover{color:var(--text)}

  .rv{background:var(--card); border:1px solid var(--border); border-radius:12px;
      padding:14px 16px; margin-bottom:10px}
  .rv .rh{display:flex; align-items:baseline; gap:10px; flex-wrap:wrap}
  .rv .idx{color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums}
  .rv .stars{color:#f59e0b; letter-spacing:1px; font-size:13px}
  .rv .title{font-weight:600; font-size:15px}
  .badge{display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px;
         font-weight:600; color:#fff; text-transform:capitalize}
  .b-pos{background:var(--good)} .b-neu{background:var(--warn)} .b-neg{background:var(--bad)}
  .b-unparsed{background:#9ca3af}
  .rv .labels{display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:8px 0}
  .rv .lbl{font-size:12px; color:var(--muted)}
  .rv .text{color:#374151; white-space:pre-wrap; overflow:hidden;
            display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical}
  .rv.expanded .text{-webkit-line-clamp:unset; display:block}
  .rv .raw{font-size:12px; color:var(--muted); margin-top:6px}
  .rv .raw code{background:var(--chip); padding:1px 6px; border-radius:4px}
  .rv .more{background:none; border:none; color:var(--accent); cursor:pointer;
            font:inherit; padding:2px 0; margin-top:6px}
  .rv .foot{display:flex; gap:8px; align-items:center; margin-top:10px;
            padding-top:10px; border-top:1px dashed var(--border); flex-wrap:wrap}
  .vbtn{border:1px solid var(--border); border-radius:8px; padding:5px 12px;
        font:inherit; font-size:13px; cursor:pointer; background:#fff; color:var(--muted)}
  .vbtn.on-agree{border-color:var(--good); color:var(--good); background:#f0fdf4; font-weight:600}
  .vbtn.on-disagree{border-color:var(--bad); color:var(--bad); background:#fef2f2; font-weight:600}
  .rv .note{flex:1 1 200px; min-width:160px; border:1px solid var(--border);
        border-radius:8px; padding:5px 8px; font:inherit; font-size:13px}
  .verdict-line{font-size:13px; color:var(--muted); margin-left:auto; white-space:nowrap}
  .verdict-line b{color:var(--text)}
  footer{margin-top:24px; color:var(--muted); font-size:12px; text-align:center}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Sentiment Classifier — Results Dashboard</h1>
    <div class="meta" id="meta"></div>
  </header>

  <div class="kpis">
    <div class="kpi"><div class="v" id="k-acc"></div><div class="l">Accuracy</div></div>
    <div class="kpi"><div class="v" id="k-f1"></div><div class="l">Macro F1</div></div>
    <div class="kpi"><div class="v" id="k-n"></div><div class="l">Reviews</div></div>
    <div class="kpi"><div class="v" id="k-wrong"></div><div class="l">Misclassified</div></div>
    <div class="kpi"><div class="v" id="k-judged"></div><div class="l">Judged by you</div>
      <div class="s" id="k-judged-s"></div></div>
  </div>

  <div class="cols">
    <div class="panel">
      <h2>Confusion matrix (true → predicted)</h2>
      <table class="cm" id="cm"></table>
      <div class="note">Rows = star-rating ground truth · Columns = model prediction</div>
    </div>
    <div class="panel">
      <h2>Per-class metrics (vs star-rating ground truth)</h2>
      <table class="pm" id="pm"></table>
    </div>
  </div>

  <div class="filters">
    <input type="search" id="q" placeholder="Search title or text…">
    <select id="f-true"><option value="">true: all</option></select>
    <select id="f-pred"><option value="">pred: all</option></select>
    <select id="f-agree"><option value="">agreement: all</option>
      <option value="correct">model correct</option>
      <option value="wrong">model wrong (vs stars)</option></select>
    <select id="f-rating"><option value="">rating: all</option></select>
    <button id="reset">Reset</button>
    <span class="count" id="count"></span>
  </div>

  <div id="list"></div>

  <footer>Generated from <code>results/sampled_reviews.csv</code> · your verdicts &amp; notes live in this browser · stars are the rating the reviewer gave</footer>
</div>

<script>
const DATA = @@DATA_JSON@@;
const CLS = ["positive","neutral","negative"];
const COLORS = {positive:"#16a34a", neutral:"#d97706", negative:"#dc2626"};
const LS_KEY = "sw-verdicts-v1";

/* ---- safe localStorage (file:// origins can throw) ---- */
const store = (() => {
  try { const t = localStorage.getItem(LS_KEY); return {ok:true, data: t?JSON.parse(t):{}}; }
  catch(e){ return {ok:false, data:{}}; }
})();
const saveStore = () => { if(store.ok) localStorage.setItem(LS_KEY, JSON.stringify(store.data)); };

/* ---- header + KPIs ---- */
const m = DATA.meta, met = DATA.metrics;
document.getElementById("meta").innerHTML =
  `Model: <b>${m.model}</b> · Endpoint: ${m.endpoint} · ` +
  `Sample: <b>${m.sample} random</b> (seed ${m.seed}) · ` +
  `Source: ${m.source} · Generated ${DATA.generated}`;
document.getElementById("k-acc").textContent = (met.accuracy*100).toFixed(1)+"%";
document.getElementById("k-f1").textContent = met.macro_f1.toFixed(3);
document.getElementById("k-n").textContent = m.sample;
document.getElementById("k-wrong").textContent = met.n_wrong;
const kJudged = document.getElementById("k-judged"),
      kJudgedS = document.getElementById("k-judged-s");
function refreshJudged(){
  const entries = Object.entries(store.data);
  const agree = entries.filter(([,v])=>v.v==="agree").length;
  const dis = entries.filter(([,v])=>v.v==="disagree").length;
  kJudged.textContent = entries.length;
  kJudgedS.textContent = (entries.length? `agree ${agree} · disagree ${dis}`:"click Agree/Disagree on a review");
}
refreshJudged();

/* ---- confusion matrix ---- */
const cmEl = document.getElementById("cm");
let html = `<tr><th></th>` + CLS.map(c=>`<th>pred ${c}</th>`).join("") + `</tr>`;
CLS.forEach((r,i)=>{
  html += `<tr><th class="rowlabel">${r}</th>` +
    met.confusion[i].map((v,j)=>{
      const bg = v? COLORS[CLS[j]]+"22" : "";
      const fg = i===j? COLORS[CLS[j]] : "";
      return `<td class="num" style="background:${bg};color:${fg}">${v}</td>`;
    }).join("") + `</tr>`;
});
cmEl.innerHTML = html;

/* ---- per-class table ---- */
const pmEl = document.getElementById("pm");
html = `<tr><th>class (n)</th><th>precision</th><th>recall</th><th>F1</th></tr>`;
CLS.forEach(c=>{
  const p = met.per_class[c];
  html += `<tr><td><span class="dot" style="background:${COLORS[c]}"></span>${c} <span style="color:var(--muted)">(n=${p.n})</span></td>
    <td>${p.precision.toFixed(3)}</td><td>${p.recall.toFixed(3)}</td>
    <td><b>${p.f1.toFixed(3)}</b><div class="bar"><i style="width:${(p.f1*100).toFixed(0)}%;background:${COLORS[c]}"></i></div></td></tr>`;
});
pmEl.innerHTML = html;

/* ---- filter selects ---- */
const fTrue = document.getElementById("f-true"), fPred = document.getElementById("f-pred"),
      fRating = document.getElementById("f-rating");
CLS.forEach(c=>{
  fTrue.insertAdjacentHTML("beforeend", `<option value="${c}">true: ${c}</option>`);
  fPred.insertAdjacentHTML("beforeend", `<option value="${c}">pred: ${c}</option>`);
});
[...new Set(DATA.reviews.map(r=>r.rating))].sort().forEach(r=>{
  fRating.insertAdjacentHTML("beforeend", `<option value="${r}">${r}★</option>`);
});
function esc(s){ return String(s??"").replace(/[&<>"]/g, ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[ch])); }
function stars(r){ return "★".repeat(Math.round(r)) + "☆".repeat(5-Math.round(r)); }

/* ---- render ---- */
const qEl = document.getElementById("q"), fAgree = document.getElementById("f-agree"),
      list = document.getElementById("list"), countEl = document.getElementById("count");
function render(){
  const q = qEl.value.trim().toLowerCase();
  const tv = fTrue.value, pv = fPred.value, av = fAgree.value, rv = fRating.value;
  const rows = DATA.reviews.filter(r=>{
    if(tv && r.true_label!==tv) return false;
    if(pv && r.pred_label!==pv) return false;
    if(av==="correct" && r.true_label!==r.pred_label) return false;
    if(av==="wrong" && r.true_label===r.pred_label) return false;
    if(rv && String(r.rating)!==rv) return false;
    if(q && !(r.title+" "+r.text).toLowerCase().includes(q)) return false;
    return true;
  });
  countEl.textContent = `${rows.length} / ${DATA.reviews.length} reviews`;
  list.innerHTML = rows.map(r=>{
    const v = store.data[r.idx]||{};
    const agreeCls = v.v==="agree"?" on-agree":(v.v==="disagree"?" on-disagree":"");
    const badge = c=>`<span class="badge b-${c[0]?c.slice(0,3):"unparsed"}">${esc(c)}</span>`;
    return `<div class="rv" data-idx="${r.idx}">
      <div class="rh">
        <span class="idx">#${r.idx+1}</span>
        <span class="stars" title="rating ${r.rating}">${stars(r.rating)}</span>
        <span class="title">${esc(r.title||"(no title)")}</span>
      </div>
      <div class="labels">
        <span class="lbl">stars→</span>${badge(r.true_label)}
        <span class="lbl">model→</span>${badge(r.pred_label)}
        ${r.true_label!==r.pred_label?'<span class="lbl" style="color:var(--bad)">✗ mismatch</span>':""}
      </div>
      <div class="text">${esc(r.text||"...")}</div>
      <button class="more">show more / less</button>
      <div class="raw" hidden>raw model answer: <code>${esc(r.raw)}</code> ·
        asin: <code>${esc(r.asin||"?")}</code> · user: <code>${esc(r.user_id||"?")}</code></div>
      <div class="foot">
        <button class="vbtn agree${agreeCls}" data-v="agree">✓ Agree</button>
        <button class="vbtn disagree${agreeCls}" data-v="disagree">✗ Disagree</button>
        <input class="note" placeholder="your note…" value="${esc(v.n||"")}">
        <span class="verdict-line" data-vline></span>
      </div>
    </div>`;
  }).join("");
  attachHandlers();
}
function attachHandlers(){
  list.querySelectorAll(".rv").forEach(card=>{
    const idx = +card.dataset.idx;
    const more = card.querySelector(".more");
    more.addEventListener("click", ()=>card.classList.toggle("expanded"));
    card.querySelectorAll(".vbtn").forEach(btn=>{
      btn.addEventListener("click", ()=>{
        const verdict = btn.dataset.v;
        store.data[idx] = {...(store.data[idx]||{}), v: verdict};
        saveStore();
        render(); refreshJudged();
      });
    });
    const note = card.querySelector(".note");
    note.addEventListener("change", ()=>{
      store.data[idx] = {...(store.data[idx]||{}), n: note.value};
      saveStore();
    });
  });
  list.querySelectorAll(".rv").forEach(card=>{  // verdict line per card
    const idx = +card.dataset.idx, v = store.data[idx];
    const el = card.querySelector("[data-vline]");
    if(v && v.v) el.textContent = v.v==="agree"?"you: agree ✓":"you: disagree ✗";
  });
}
document.getElementById("reset").addEventListener("click", ()=>{
  qEl.value=""; fTrue.value=""; fPred.value=""; fAgree.value=""; fRating.value=""; render();
});
[qEl, fTrue, fPred, fAgree, fRating].forEach(el=>el.addEventListener("input", render));
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
