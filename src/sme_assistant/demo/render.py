"""HTML for the dashboard.

Everything is inlined. The Raspberry Pi that runs this may have no network, and
a demonstrator that needs a CDN to look right is a demonstrator that fails in
the room. No external stylesheet, no web font, no script from anywhere.

Arm colours are the Okabe-Ito assignment used by the dissertation figures, so
an arm looks the same on screen as it does in Chapter 4.
"""

from __future__ import annotations

from html import escape
from typing import Any, Iterable, Mapping

ARM_COLOUR = {"A": "#009E73", "B": "#0072B2", "C": "#E69F00", "D": "#D55E00"}
ARM_ROLE = {
    "A": "Naive RAG. No status metadata.",
    "B": "Status metadata shown.",
    "C": "Superseded documents filtered out.",
    "D": "Verification over the same evidence as B.",
}

STYLE = """
*{box-sizing:border-box}
body{margin:0;font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,
     Helvetica,Arial,sans-serif;color:#1a1a1a;background:#f6f7f9}
a{color:#0060a8}
.wrap{max-width:1500px;margin:0 auto;padding:0 20px 56px}
.banner{padding:13px 20px;font-weight:700;letter-spacing:.02em;color:#fff;
        text-align:center}
.banner.live{background:#8a3a00}
.banner.replay{background:#004b7c}
.banner .sub{display:block;font-weight:400;font-size:13px;opacity:.92;
             letter-spacing:0}
header{padding:22px 0 8px}
h1{font-size:21px;margin:0 0 4px}
.muted{color:#5b6167;font-size:13.5px}
nav{margin:14px 0 22px;display:flex;gap:10px;flex-wrap:wrap}
nav a{display:inline-block;padding:8px 15px;border:1px solid #c8cdd3;
      border-radius:7px;background:#fff;text-decoration:none;color:#1a1a1a;
      font-size:14px}
nav a.on{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
      gap:16px;align-items:start}
.card{background:#fff;border:1px solid #dfe3e8;border-radius:10px;
      overflow:hidden}
.card h2{margin:0;padding:11px 15px;font-size:15px;color:#fff}
.card .body{padding:14px 15px}
.role{font-size:12.5px;color:#5b6167;margin:-4px 0 12px}
.answer{background:#f7f9fb;border-left:3px solid #c8cdd3;padding:11px 13px;
        border-radius:0 6px 6px 0;white-space:pre-wrap;font-size:14px}
.flag{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11.5px;
      font-weight:700;margin:0 6px 6px 0}
.bad{background:#fbe2e2;color:#8a1d1d}
.ok{background:#e3f4ec;color:#155e3f}
.info{background:#e7eef7;color:#14456f}
.warn{background:#fcefdb;color:#7a4b06}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid #eceff2;
      vertical-align:top}
th{color:#5b6167;font-weight:600;white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
code,.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
           font-size:12.5px}
details{margin-top:11px}
summary{cursor:pointer;font-size:13px;color:#0060a8}
.evidence{font-size:12.5px;margin-top:8px}
.evidence li{margin-bottom:5px}
.sup{color:#8a1d1d;font-weight:700}
form{background:#fff;border:1px solid #dfe3e8;border-radius:10px;padding:16px;
     margin-bottom:20px}
input[type=text],select{width:100%;padding:10px 12px;font-size:15px;
      border:1px solid #c8cdd3;border-radius:7px;font-family:inherit}
button{margin-top:11px;padding:10px 20px;font-size:15px;border:0;
       border-radius:7px;background:#1a1a1a;color:#fff;cursor:pointer}
button:disabled{background:#9aa1a8;cursor:not-allowed}
.note{background:#fff8e6;border:1px solid #f0dfae;border-radius:8px;
      padding:12px 14px;font-size:13.5px;margin-bottom:18px}
.prov{font-size:12px;color:#5b6167;margin-top:26px;border-top:1px solid #dfe3e8;
      padding-top:14px}
.choice{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
        gap:18px;margin-top:22px}
.choice a{display:block;padding:22px;border:1px solid #dfe3e8;border-radius:12px;
          background:#fff;text-decoration:none;color:#1a1a1a}
.choice h3{margin:0 0 8px;font-size:17px}
.choice p{margin:0;font-size:13.5px;color:#5b6167}
"""


def page(title: str, banner: str, banner_class: str, banner_sub: str,
         body: str) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{escape(title)}</title><style>{STYLE}</style></head><body>"
        f"<div class='banner {banner_class}'>{escape(banner)}"
        f"<span class='sub'>{escape(banner_sub)}</span></div>"
        f"<div class='wrap'>{body}</div></body></html>"
    )


def landing(replay_ready: bool, replay_error: str | None,
            live_ready: bool, live_error: str | None) -> str:
    def tile(href: str, heading: str, text: str, ready: bool,
             error: str | None) -> str:
        if ready:
            return (f"<a href='{href}'><h3>{escape(heading)}</h3>"
                    f"<p>{escape(text)}</p></a>")
        return ("<div class='choice-off' style='padding:22px;border:1px dashed "
                "#c8cdd3;border-radius:12px;background:#fff'>"
                f"<h3 style='margin:0 0 8px;font-size:17px;color:#5b6167'>"
                f"{escape(heading)}</h3><p style='margin:0;font-size:13.5px;"
                f"color:#5b6167'>{escape(text)}</p>"
                f"<p style='margin:10px 0 0;font-size:13px;color:#8a1d1d'>"
                f"Unavailable: {escape(error or 'unknown reason')}</p></div>")

    body = (
        "<header><h1>On-device RAG assistant with claim-level verification</h1>"
        "<p class='muted'>MSc dissertation artefact, WMG, University of "
        "Warwick. Choose a mode. The two are kept separate on purpose and their "
        "outputs are never shown together.</p></header>"
        "<div class='choice'>"
        + tile("/replay", "Frozen Study Replay",
               "The 68 evaluated questions, with all four experimental arms "
               "side by side, read from the committed run records. No model is "
               "invoked and no network or Ollama installation is needed.",
               replay_ready, replay_error)
        + tile("/live", "Live Assistant",
               "Ask a new question and watch Arm D answer it: retrieval, "
               "grounded draft, then verification. This is a demonstration, "
               "not part of the reported evaluation.",
               live_ready, live_error)
        + "</div>"
    )
    return page("Dashboard", "Post-evaluation demonstrator",
                "replay",
                "Built after the experiment. Contributes no evidence to any "
                "hypothesis.", body)


def _flags(answer) -> str:
    out = []
    if answer.cited_superseded:
        out.append("<span class='flag bad'>cited withdrawn policy</span>")
    if answer.hallucinated_citations:
        out.append("<span class='flag bad'>invented citation</span>")
    if answer.has_valid_citation_ids is False:
        out.append("<span class='flag warn'>citation id not retrieved</span>")
    if answer.abstained:
        out.append("<span class='flag info'>declined to answer</span>")
    if answer.revised:
        out.append("<span class='flag warn'>answer revised by verifier</span>")
    if answer.verification and answer.verification.get("conflict_detected"):
        relationship = answer.verification.get("relationship") or "conflict"
        out.append(f"<span class='flag info'>{escape(str(relationship))}</span>")
    if answer.confidence:
        out.append(f"<span class='flag info'>confidence: "
                   f"{escape(str(answer.confidence))} (rule-based)</span>")
    if not out:
        out.append("<span class='flag ok'>no flags raised</span>")
    return "".join(out)


def _timings(answer) -> str:
    generation = answer.generation or {}
    verifier = getattr(answer, "verification_generation", None) or {}
    rows = [("Retrieval, embed", answer.retrieval.get("embed_seconds")),
            ("Retrieval, search", answer.retrieval.get("search_seconds")),
            ("Draft, prompt", generation.get("prompt_seconds")),
            ("Draft, generation", generation.get("eval_seconds")),
            ("Draft, total", generation.get("wall_seconds")),
            ("Verifier, prompt", verifier.get("prompt_seconds")),
            ("Verifier, generation", verifier.get("eval_seconds")),
            ("Verification, total", answer.verification_seconds),
            ("End to end", answer.wall_seconds)]
    cells = "".join(
        f"<tr><th>{escape(label)}</th><td class='num'>{value:.3f} s</td></tr>"
        for label, value in rows if isinstance(value, (int, float)))
    # Both stages are shown separately. The verifier processes roughly twice
    # the prompt and emits roughly four times the output of the draft, which is
    # the whole of the latency finding and is invisible in a single total.
    for label, stage in (("Draft", generation), ("Verifier", verifier)):
        tokens_in, tokens_out = stage.get("prompt_tokens"), stage.get("eval_tokens")
        if tokens_in or tokens_out:
            cells += (f"<tr><th>{escape(label)} tokens</th><td class='num'>"
                      f"{tokens_in or 0} in, {tokens_out or 0} out</td></tr>")
        rate = stage.get("eval_tokens_per_second")
        if isinstance(rate, (int, float)):
            cells += (f"<tr><th>{escape(label)} decode</th><td class='num'>"
                      f"{rate:.2f} tok/s</td></tr>")
    # Temperature and throttling by stage, not merged. The previous version
    # took the draft's reading or fell back to the verifier's and printed one
    # row called "CPU temperature", which on the Pi 5 record hid the finding:
    # the draft ran at 84.8 degrees and the verifier, arriving second onto an
    # already hot core, at 88.1. Throttling was reported once, so a stage that
    # was throttled and a stage that was not looked like one machine.
    reported_any = False
    for label, stage in (("Draft", generation), ("Verifier", verifier)):
        if not stage:
            continue
        temperature = stage.get("cpu_temp_c")
        if isinstance(temperature, (int, float)):
            reported_any = True
            cells += (f"<tr><th>{escape(label)} CPU temperature</th>"
                      f"<td class='num'>{temperature:.1f} &deg;C</td></tr>")
        throttled = stage.get("throttled")
        if throttled is not None:
            reported_any = True
            cells += (f"<tr><th>{escape(label)} throttled</th>"
                      f"<td class='num'>{'yes' if throttled else 'no'}"
                      f"</td></tr>")
    if not reported_any:
        # Said once, and said as absence rather than as "no". The laptop runs
        # report neither field, and a row reading "no" would be a measurement
        # that was never taken.
        cells += ("<tr><th>Thermal telemetry</th><td class='num'>"
                  "not reported on this host</td></tr>")
    return f"<table>{cells}</table>"


def _evidence(answer) -> str:
    results = answer.retrieval.get("results") or []
    if not results:
        return "<p class='muted'>No evidence recorded.</p>"
    items = []
    for result in results:
        status = str(result.get("status", ""))
        marker = (f" <span class='sup'>[{escape(status.upper())}]</span>"
                  if status and status != "current" else "")
        items.append(
            f"<li><code>{escape(str(result.get('chunk_id','')))}</code>"
            f"{marker} &middot; score {float(result.get('score', 0)):.3f}<br>"
            f"<span class='muted'>{escape(str(result.get('citation','')))}"
            f"</span></li>")
    return f"<ul class='evidence'>{''.join(items)}</ul>"


def _verdicts(answer, *, expanded: bool) -> str:
    """The claim audit.

    Expanded in the single-card live view, where there is width for a
    three-column table, and collapsed in the four-up replay, where it made the
    Arm D card twice the height of the others and wrapped the claim column to
    two words. The audit is the most interesting thing the verifier produces,
    so it stays one click away rather than being dropped.
    """
    verification = answer.verification or {}
    verdicts = verification.get("verdicts") or []
    if not verdicts:
        return ""
    rows = "".join(
        f"<tr><td>{escape(str(v.get('claim','')))}</td>"
        f"<td>{escape(str(v.get('verdict','')))}</td>"
        f"<td><code>{escape(', '.join(v.get('supporting') or []) or '-')}</code>"
        f"</td>"
        f"<td><code>{escape(', '.join(v.get('contradicting') or []) or '-')}</code>"
        f"</td></tr>" for v in verdicts)
    rationale = verification.get("rationale")
    tail = (f"<p class='muted' style='margin-top:8px'>{escape(str(rationale))}</p>"
            if rationale else "")
    contradicted = sum(1 for v in verdicts
                       if str(v.get("verdict", "")).upper() == "CONTRADICTED")
    label = f"Claim audit ({len(verdicts)} claims"
    label += f", {contradicted} contradicted)" if contradicted else ")"
    # The verdicts below are what the verifier model returned. They are not a
    # key, and they are sometimes wrong: in the frozen run the layer marked a
    # correct claim contradicted and endorsed a withdrawn document's claim on
    # the same question. Presenting them without that caveat would invite a
    # viewer to read a model's opinion as an adjudication.
    caveat = ("<p class='muted' style='margin:8px 0 0'><strong>Recorded "
              "verifier output, not ground truth.</strong> These verdicts are "
              "what the verification model returned. They were not checked "
              "against the answer key and are sometimes incorrect.</p>")
    return (f"<details{' open' if expanded else ''}><summary>{escape(label)}"
            "</summary>"
            "<table><tr><th>Claim</th><th>Verdict</th><th>Supported by</th>"
            f"<th>Contradicted by</th></tr>{rows}</table>{caveat}{tail}</details>")


def arm_card(answer, *, show_draft: bool = True, expanded: bool = False) -> str:
    colour = ARM_COLOUR.get(answer.arm, "#5b6167")
    draft = ""
    if (show_draft and answer.draft_answer
            and answer.draft_answer.strip() != answer.answer.strip()):
        draft = ("<details><summary>Draft before verification</summary>"
                 f"<div class='answer'>{escape(answer.draft_answer)}</div>"
                 "</details>")
    elif show_draft and answer.has_verification:
        draft = ("<p class='muted' style='margin-top:9px'>The verifier returned "
                 "the draft unchanged.</p>")
    citations = ", ".join(answer.citations) or "none"
    return (
        f"<div class='card'><h2 style='background:{colour}'>Arm {escape(answer.arm)}"
        f"</h2><div class='body'>"
        f"<p class='role'>{escape(ARM_ROLE.get(answer.arm, ''))}</p>"
        f"<div class='answer'>{escape(answer.answer) or '<em>empty</em>'}</div>"
        f"<p style='margin:11px 0 4px'>{_flags(answer)}</p>"
        f"<p class='muted'>Cites <code>{escape(citations)}</code></p>"
        f"{draft}{_verdicts(answer, expanded=expanded)}"
        "<details><summary>Retrieved evidence</summary>"
        f"{_evidence(answer)}</details>"
        "<details><summary>Timings and device state</summary>"
        f"{_timings(answer)}</details>"
        "</div></div>")


def replay_page(library, selected, question_ids: Iterable[str]) -> str:
    options = "".join(
        f"<option value='{escape(qid)}'"
        f"{' selected' if selected and qid == selected.question_id else ''}>"
        f"{escape(qid)}</option>" for qid in question_ids)
    cards = "".join(arm_card(selected.by_arm[arm], expanded=False)
                    for arm in ("A", "B", "C", "D") if arm in selected.by_arm)
    superseded = selected.any_cited_superseded
    note = ""
    if superseded:
        note = ("<div class='note'><strong>At least one arm cited a withdrawn "
                "document on this question.</strong> The withdrawn identifiers "
                f"are <code>{escape(', '.join(superseded))}</code>. Compare the "
                "arms below to see which configurations avoided it.</div>")
    runs = library.provenance["runs"]
    prov = "".join(
        f"<tr><th>Arm {escape(arm)}</th><td><code>{escape(str(info['directory']))}"
        f"</code></td><td class='muted'>{escape(str(info['description'] or ''))}"
        f"</td></tr>" for arm, info in sorted(runs.items()))
    corpus = next(iter(runs.values()))["corpus_sha256"][:12]
    body = (
        "<header><h1>Frozen Study Replay</h1>"
        "<p class='muted'>The 68 held-out test questions, exactly as the four "
        "arms answered them during the frozen experimental run. Nothing on this "
        "page is generated now: every word was produced on 14 August 2026 and "
        "has not changed since.</p></header>"
        "<nav><a href='/'>Mode</a><a class='on' href='/replay'>Frozen replay</a>"
        "<a href='/live'>Live assistant</a></nav>"
        "<form method='get' action='/replay'>"
        "<label for='q'><strong>Question</strong></label>"
        f"<select id='q' name='q' onchange='this.form.submit()'>{options}</select>"
        "<noscript><button type='submit'>Show</button></noscript></form>"
        f"<h2 style='font-size:17px;margin:0 0 4px'>{escape(selected.question)}</h2>"
        f"<p class='muted'><code>{escape(selected.question_id)}</code> &middot; "
        f"{escape(selected.category)}"
        + (f" &middot; family <code>{escape(selected.family_id)}</code>"
           if selected.family_id else "") + "</p>"
        f"{note}<div class='grid'>{cards}</div>"
        f"<div class='prov'><strong>Provenance.</strong> Corpus "
        f"<code>{escape(corpus)}</code>, {library.provenance['question_count']} "
        "questions, four arms. These are the four frozen quality runs named in "
        "<code>FROZEN_QUALITY_RUNS</code>; no other run can be shown here."
        f"<table style='margin-top:8px'>{prov}</table></div>")
    return page("Frozen Study Replay", "Frozen experimental replay", "replay",
                "Committed records from the evaluated run. No model is invoked "
                "and nothing is generated now.", body)


def live_page(question: str | None, answer, error: str | None,
              model_status: Mapping[str, Any]) -> str:
    ready = bool(model_status.get("ready"))
    status_line = (
        f"Models available: <code>{escape(str(model_status.get('generation')))}"
        f"</code> and <code>{escape(str(model_status.get('verification')))}"
        "</code>." if ready else
        f"<span style='color:#8a1d1d'>{escape(str(model_status.get('detail')))}"
        "</span>")
    # Amendment 1.31.3. Whether this pipeline is configured as the frozen Arm D
    # run was, field by field. Shown rather than only tested: a demonstration
    # that quietly drifted from the experiment it demonstrates is exactly the
    # confusion the two-mode design exists to prevent.
    agreement = model_status.get("frozen_agreement") or {}
    agreement_line = ""
    if agreement:
        if agreement.get("matches"):
            agreement_line = (
                "<p class='muted'>Configuration matches the frozen Arm D run on "
                f"all {len(agreement.get('fields') or {})} compared fields, "
                "including the configuration fingerprint, sampling options, "
                "retrieval parameters and index hash.</p>")
        else:
            differs = ", ".join(escape(str(d)) for d in agreement.get("differs") or [])
            agreement_line = (
                "<p style='color:#8a1d1d'><strong>This pipeline differs from "
                f"the frozen Arm D run on: {differs}.</strong> Answers below "
                "are produced by a different configuration from the one the "
                "reported results came from.</p>")
    result = ""
    if error:
        result = f"<div class='note'><strong>Could not answer.</strong> {escape(error)}</div>"
    elif answer is not None:
        result = (f"<h2 style='font-size:17px;margin:18px 0 10px'>"
                  f"{escape(question or '')}</h2>"
                  f"<div class='grid'>{arm_card(answer, expanded=True)}</div>")
    body = (
        "<header><h1>Live Assistant</h1>"
        "<p class='muted'>Arm D, the verified configuration, answering a new "
        "question on this device. Retrieval, grounded generation, then a "
        "verification pass over the same evidence. Nothing here has been scored "
        "and nothing here is evidence about the system's quality.</p></header>"
        "<nav><a href='/'>Mode</a><a href='/replay'>Frozen replay</a>"
        "<a class='on' href='/live'>Live assistant</a></nav>"
        "<div class='note'>Only Arm D runs live. The four-arm comparison is "
        "available under <a href='/replay'>Frozen replay</a>, where it comes "
        "from the committed experimental records rather than from a fresh "
        f"execution.<br><span class='muted'>{status_line}</span>"
        f"{agreement_line}</div>"
        "<form method='post' action='/live'>"
        "<label for='q'><strong>Ask a question about the knowledge base"
        "</strong></label>"
        f"<input id='q' type='text' name='q' value='{escape(question or '')}' "
        "placeholder='When does Statutory Sick Pay start being paid?' "
        f"{'' if ready else 'disabled'}>"
        f"<button type='submit' {'' if ready else 'disabled'}>"
        f"{'Ask' if ready else 'Models unavailable'}</button></form>"
        f"{result}")
    return page("Live Assistant", "Live demonstration, not part of the reported "
                "evaluation", "live",
                "Arm D only. Output is unscored and contributes to no result.",
                body)
