"""Portable offline experiment dashboard. No web server or JS dependencies."""
from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path

from .experiment import _all_candidates, baselines, load_run
from .genetics import pareto_fronts


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _plot_progress(records: list[dict]) -> str:
    groups = defaultdict(list)
    for c in records:
        if "score" in c:
            groups[c["generation"]].append(c["score"]["fitness"])
    if not groups:
        return '<div class="empty">No scored generations yet</div>'
    curve = [(g, max(values)) for g, values in sorted(groups.items())]
    minscore = min(score for _, score in curve)
    maxscore = max(score for _, score in curve)
    lo, hi = min(0.0, minscore - .05), max(1.0, maxscore + .05)
    span = max(.001, hi - lo)
    xs = [54 + (x - curve[0][0]) * 540 / max(1, curve[-1][0] - curve[0][0]) for x, _ in curve]
    ys = [225 - (s - lo) * 175 / span for _, s in curve]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    circles = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#72e2b5"><title>Gen {g}: {score:.4f}</title></circle>'
                      for x, y, (g, score) in zip(xs, ys, curve))
    ticks = "".join(
        f'<text x="{x:.1f}" y="250" fill="#8796a8" text-anchor="middle" font-size="11">G{g}</text>'
        for x, (g, _) in zip(xs, curve)
    )
    guides = "".join(
        f'<line x1="54" y1="{y}" x2="594" y2="{y}" stroke="#233447" stroke-dasharray="5,5"/>'
        f'<text x="45" y="{y+4}" fill="#718197" text-anchor="end" font-size="10">{v:.2f}</text>'
        for v, y in [(hi, 50), ((hi+lo)/2, 137), (lo, 225)]
    )
    return (f'<svg viewBox="0 0 620 270" role="img" aria-label="Best fitness by generation" class="plot">'
            f'{guides}<polyline points="{pts}" fill="none" stroke="#72e2b5" stroke-width="3"/>'
            f'{circles}{ticks}</svg>')


def _plot_pareto(records: list[dict], tasks: list[str]) -> str:
    if len(tasks) < 2:
        return '<div class="empty">Select at least two tasks for a Pareto scatter plot.</div>'
    scored = [r for r in records if "score" in r and all(k in r["score"].get("metrics", {}) for k in tasks[:2])]
    if not scored:
        return '<div class="empty">Score candidates to populate the scatter plot.</div>'
    fronts = pareto_fronts(scored, tasks)
    first = {x["id"] for x in fronts[0]}
    dots = []
    for c in scored:
        x = 52 + 520 * c["score"]["metrics"][tasks[0]]
        y = 224 - 182 * c["score"]["metrics"][tasks[1]]
        color = "#72e2b5" if c["id"] in first else "#7e94b2"
        opacity = ".98" if c["id"] in first else ".54"
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" opacity="{opacity}">'
                    f'<title>{_e(c["id"])} — {_e(tasks[0])}: {c["score"]["metrics"][tasks[0]]:.3f}, '
                    f'{_e(tasks[1])}: {c["score"]["metrics"][tasks[1]]:.3f}</title></circle>')
    return (f'<svg class="plot" viewBox="0 0 620 270" role="img" aria-label="Pareto task scatter plot">'
            f'<path d="M52 40 V224 H576" fill="none" stroke="#53637a" stroke-width="1.5"/>'
            f'<text x="314" y="258" fill="#b4c4d9" text-anchor="middle" font-size="12">{_e(tasks[0])} →</text>'
            f'<text x="18" y="140" fill="#b4c4d9" text-anchor="middle" font-size="12" '
            f'transform="rotate(-90 18 140)">{_e(tasks[1])} →</text>'
            f'<text x="52" y="239" fill="#718197" font-size="10">0</text>'
            f'<text x="565" y="239" fill="#718197" font-size="10">1</text>'
            f'<text x="32" y="45" fill="#718197" font-size="10">1</text>'
            f'{"".join(dots)}</svg>')


def create_report(run: Path, out: Path) -> Path:
    spec, state = load_run(run)
    records = _all_candidates(run)
    scored = [r for r in records if "score" in r]
    ranked = sorted(scored, key=lambda e: e["score"]["fitness"], reverse=True)
    fake_count = sum(c["score"]["source"] == "SIMULATED_TOY" for c in scored)
    source_label = ("SIMULATED DATA — NOT MODEL PERFORMANCE" if fake_count == len(scored) and fake_count else
                    "MIXED SYNTHETIC AND REAL/UNVERIFIED — DO NOT COMPARE" if fake_count else
                    "Measured / manually imported" if scored else "No final evaluations")
    tasks = [t.name for t in spec.evaluation.tasks]
    fronts = pareto_fronts(scored, tasks) if len(tasks) >= 2 and scored else []
    pareto_ids = {c["id"] for c in fronts[0]} if fronts else set()
    baseline_scores = baselines(run)
    best = ranked[0] if ranked else None
    best_text = f'{best["score"]["fitness"]:.4f}' if best else '—'
    gen_count = 1 + state["generation"]
    best_baseline = max(baseline_scores.values(), key=lambda d: d["fitness"], default=None)
    comparable = (best and best_baseline is not None and best["score"].get("source") in ("lm_eval", "lerp_eval")
                  and best_baseline.get("source") in ("lm_eval", "lerp_eval") and
                  best["score"].get("evaluation_settings") == best_baseline.get("evaluation_settings"))
    baseline_diff = f'{best["score"]["fitness"]-best_baseline["fitness"]:+.4f}' if comparable else '—'
    rows = []
    for item in sorted(records, key=lambda x: (x["generation"], x["candidate"])):
        score = item.get("score")
        source = score.get("source", "pending") if score else "pending"
        screen = item.get("screen_score")
        screening_txt = f'{screen["fitness"]:.4f} (preliminary)' if screen else '—'
        metrics = score.get("metrics", {}) if score else {}
        metrics_txt = " · ".join(f"{_e(k)}: {v:.3f}" for k, v in metrics.items()) or "—"
        rank = "Front 1" if item["id"] in pareto_ids else "—"
        fitness = f'{score["fitness"]:.4f}' if score else "—"
        rows.append(f'<tr data-gen="{item["generation"]}"><td><span class="mono">{_e(item["id"])}</span></td>'
                    f'<td>G{item["generation"]}</td><td class="num">{fitness}</td>'
                    f'<td>{_e(item.get("method", spec.method))}</td>'
                    f'<td>{_e(source)}</td><td>{_e(score.get("status", "—")) if score else "—"}</td>'
                    f'<td>{screening_txt}</td><td>{metrics_txt}</td><td>{rank}</td></tr>')
    baselines_html = "".join(
        f'<div class="baseline-row"><span class="mono">{_e(name)}</span><strong>{item["fitness"]:.4f}</strong></div>'
        for name, item in baseline_scores.items()
    ) or '<div class="empty">No baselines. Run <code>lerp baseline --name all</code>.</div>'
    summary_status = "SIMULATED" if fake_count else ("SMOKE TEST" if spec.evaluation.limit else "EXPERIMENT")
    template = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Lerp | {title}</title>
<style>
:root{{--bg:#08111d;--panel:#101c2b;--panel2:#152337;--line:#25354a;--fg:#e8f0f9;--muted:#91a4bb;--accent:#72e2b5;--blue:#8aaaf5}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 Inter,system-ui,-apple-system,Segoe UI,sans-serif}}
.wrap{{max-width:1300px;margin:0 auto;padding:26px 30px 80px}}
header{{display:flex;align-items:center;justify-content:space-between;gap:16px;border-bottom:1px solid var(--line);padding-bottom:24px}}
.brand{{font-size:13px;letter-spacing:.19em;font-weight:900;color:var(--accent)}}h1{{font-size:clamp(26px,3vw,40px);letter-spacing:-.035em;margin:8px 0 4px;line-height:1.2}}h2{{font-size:17px;margin:0 0 16px}}
p{{margin:6px 0;color:var(--muted)}}.tag{{border:1px solid #32674f;background:#15382e;color:#9ff0cb;border-radius:99px;padding:7px 14px;font-size:11px;font-weight:800;letter-spacing:.08em}}
.warning{{margin:22px 0;padding:12px 16px;border:1px solid #886d3a;background:#332b1a;color:#ffd595;border-radius:12px;font-weight:700}}
.kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:25px 0}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;min-width:0}}
.metric-label{{color:var(--muted);font-size:12px;font-weight:700;letter-spacing:.03em}}
.value{{font-size:30px;letter-spacing:-.04em;font-weight:800;margin:4px 0 0;font-variant-numeric:tabular-nums}}.value span{{font-size:12px;color:var(--muted)}}
.layout{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}}.plot{{display:block;width:100%;height:auto;min-height:180px}}
.small{{font-size:12px;color:var(--muted)}}.baseline-row{{display:flex;justify-content:space-between;gap:10px;padding:10px 0;border-top:1px solid var(--line)}}
.mono,code{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}code{{color:var(--blue)}}
section.full{{margin:18px 0}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:800px}}th,td{{padding:12px;border-top:1px solid var(--line);text-align:left;white-space:nowrap}}th{{color:var(--muted);font-size:11px;letter-spacing:.04em;text-transform:uppercase}}tbody tr:hover{{background:#18283b}}.num{{font-variant-numeric:tabular-nums;font-weight:800;color:var(--accent)}}
select{{background:var(--panel2);border:1px solid var(--line);color:var(--fg);border-radius:8px;padding:9px}}.table-head{{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:16px}}
.empty{{color:var(--muted);font-size:13px;padding:18px 0}}footer{{margin-top:26px;font-size:12px;color:var(--muted)}}
@media(max-width:800px){{.wrap{{padding:18px}}.kpis{{grid-template-columns:repeat(2,1fr)}}.layout{{grid-template-columns:1fr}}header{{align-items:flex-start;flex-direction:column}}}}
</style></head><body><div class="wrap">
<header><div><div class="brand">◈ LERP / RESEARCH LAB</div><h1>{title}</h1><p>{method} · {mode} · {selection} selection · {parent_count} parents · seed {seed}</p></div><div class="tag">{status}</div></header>
{warning}
<div class="kpis"><div class="card"><div class="metric-label">Best fitness</div><div class="value">{best}</div></div>
<div class="card"><div class="metric-label">Candidates scored</div><div class="value">{scored}<span> / {total}</span></div></div>
<div class="card"><div class="metric-label">Generations</div><div class="value">{generations}</div></div>
<div class="card"><div class="metric-label">Δ vs. best baseline</div><div class="value">{delta}</div></div></div>
<div class="layout"><section class="card"><h2>Fitness progression</h2><div class="small">Maximum recorded weighted fitness by generation</div>{progress}</section>
<section class="card"><h2>Objective trade-off</h2><div class="small">Green = non-dominated Pareto front. Higher is better.</div>{pareto}</section></div>
<div class="layout"><section class="card"><h2>Parent checkpoints</h2><p>Measurements from the same benchmark configuration.</p>{baselines}</section>
<section class="card"><h2>Experiment metadata</h2><div class="baseline-row"><span>Run schema</span><strong>{schema}</strong></div>
<div class="baseline-row"><span>Layer interpolation points</span><strong>{genes}</strong></div>
<div class="baseline-row"><span>Fitness tasks</span><strong>{task_count}</strong></div>
<div class="baseline-row"><span>Evaluation limit</span><strong>{limit}</strong></div>
<div class="baseline-row"><span>Data classification</span><strong>{source_label}</strong></div></section></div>
<section class="card full"><div class="table-head"><h2>Candidate registry</h2><select id="genFilter" aria-label="Filter generation"><option value="all">All generations</option>{options}</select></div>
<div class="scroll"><table><thead><tr><th>Candidate</th><th>Generation</th><th>Fitness</th><th>Merge method</th><th>Score source</th><th>Validity</th><th>Screen-only fitness</th><th>Task metrics</th><th>Pareto</th></tr></thead>
<tbody>{rows}</tbody></table></div></section>
<footer>Generated locally by Lerp v0.3. Research tool only. Merged checkpoint performance is not guaranteed. Imported manual scores are not independently verified.<br>
Always compare to parent baselines and hold-out datasets. License obligations follow all upstream weights.</footer>
</div><script>const select=document.getElementById('genFilter');select.addEventListener('change',()=>{{for(const tr of document.querySelectorAll('tbody tr')){{tr.hidden=select.value!=='all'&&tr.dataset.gen!==select.value;}}}});</script></body></html>'''
    options = ''.join(f'<option value="{g}">Generation {g}</option>' for g in range(gen_count))
    warning = (f'<div class="warning">{_e(source_label)}. Toy scores are produced by a synthetic mathematical function, not by running any LLM.</div>'
               if fake_count else ('<div class="warning">Evaluation LIMIT is set; these are preliminary smoke-test measurements, not publication-grade results.</div>'
                                    if spec.evaluation.limit is not None else ''))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template.format(
        title=_e(spec.name), method=_e(spec.method), mode=_e(spec.mode), selection=_e(spec.selection),
        parent_count=len(spec.parents), seed=spec.seed, status=summary_status,
        warning=warning, best=best_text, scored=len(scored), total=len(records),
        generations=gen_count, delta=baseline_diff, progress=_plot_progress(records),
        pareto=_plot_pareto(records, tasks), baselines=baselines_html,
        schema=state["schema_version"], genes=spec.genes, task_count=len(tasks),
        limit=spec.evaluation.limit if spec.evaluation.limit is not None else 'None',
        source_label=_e(source_label), options=options, rows=''.join(rows),
    ), encoding="utf-8")
    return out
