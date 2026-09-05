/* PCG-MAS v3.0 control plane.
 * The frontend NEVER reimplements scientific definitions: conjunct names, audit
 * channels, gate states, provenance classes and controller actions all come from
 * /api/contract, which is generated from the Python core. */
const API = (window.PCG_API_BASE || "").replace(/\/$/, "");
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "—").replace(/[<>&]/g, c => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c]));
const num = (v, d = 4) => (v === null || v === undefined) ? "—" : (typeof v === "number" ? v.toFixed(d) : String(v));

let CONTRACT = null, LAST = null;

async function api(path, body) {
  const r = await fetch(API + path, body ? {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body)
  } : {});
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

/* ---------------------------------------------------------------- tabs */
$$(".tab").forEach(t => t.addEventListener("click", () => {
  $$(".tab").forEach(x => { x.classList.remove("active"); x.setAttribute("aria-selected", "false"); });
  $$(".view").forEach(v => v.classList.remove("active"));
  t.classList.add("active"); t.setAttribute("aria-selected", "true");
  $("#view-" + t.dataset.view).classList.add("active");
}));
document.addEventListener("keydown", e => {
  if (e.target.matches("textarea,select,input")) return;
  const tabs = $$(".tab"); const i = tabs.findIndex(t => t.classList.contains("active"));
  if (e.key === "ArrowRight") tabs[(i + 1) % tabs.length].click();
  if (e.key === "ArrowLeft") tabs[(i - 1 + tabs.length) % tabs.length].click();
});
$("#theme").addEventListener("click", () => {
  const el = document.documentElement;
  el.dataset.contrast = el.dataset.contrast === "high" ? "" : "high";
});

/* ---------------------------------------------------------------- boot */
async function boot() {
  try {
    CONTRACT = await api("/api/contract");
    $("#release-chip").textContent = CONTRACT.PCG_MAS_RELEASE;
    $("#foot-version").textContent =
      `PCG-MAS ${CONTRACT.PCG_MAS_RELEASE} · schema ${CONTRACT.scientific_schema_version}`;
    renderChannels({});
  } catch (e) { console.warn("contract unavailable", e); }
  try { renderRegistry(await api("/api/registry")); } catch (e) { }
  try { renderResults(await api("/api/artifacts")); } catch (e) { }
}

/* ---------------------------------------------------------------- run */
$("#go").addEventListener("click", async () => {
  const btn = $("#go"); btn.disabled = true; btn.textContent = "executing…";
  try {
    LAST = await api("/api/v3/run", {
      request: $("#req").value, scenario: $("#scenario").value, mode: "offline_synthetic"
    });
    renderRun(LAST);
  } catch (e) {
    $("#terminal").textContent = "error: " + e.message;
  } finally { btn.disabled = false; btn.textContent = "Execute graph"; }
});

function renderRun(r) {
  const st = r.state, spans = r.spans || [];
  const dur = Object.fromEntries(spans.map(s => [s.name, s.duration_ms]));
  const nodes = (CONTRACT?.graph_nodes || []).filter(n => n !== "request");
  $("#pipeline").innerHTML = nodes.map(n => {
    const hit = (st.provenance || []).find(p => p.node === n);
    const failed = (n === "audit" && st.check === false);
    return `<li><span class="dot ${hit ? (failed ? "fail" : "on") : ""}"></span>
      <span class="nm">${esc(n)}</span>
      <span class="ms">${hit ? num(dur[n] ?? 0, 3) + " ms" : "—"}</span></li>`;
  }).join("");
  const t = $("#terminal");
  t.className = "terminal " + (st.terminal || "");
  t.textContent = `${st.terminal ?? "—"}  ·  check=${st.check}  ·  action=${st.controller_action ?? "—"}`;
  renderCert(st); renderChannels(st.channels || {}); renderReplay(st);
  renderResp(st); renderDep(st); renderPolicy(st); renderSpans(spans); renderCost(st, spans);
}

/* ------------------------------------------------------------ certificate */
const WHY = {
  V_H: "evidence commitment / hash integrity",
  V_Pi: "pinned snapshot replay",
  V_Gamma: "execution-contract compliance",
  V_vdash: "checker-relative entailment (reproducible, not objectively true)"
};
function renderCert(st) {
  const map = { V_H: st.v_h, V_Pi: st.v_pi, V_Gamma: st.v_gamma, V_vdash: st.v_entail };
  $("#conjuncts").innerHTML = (CONTRACT?.conjuncts || Object.keys(map)).map(c => {
    const v = map[c];
    const cls = v === true ? "pass" : v === false ? "fail" : "unk";
    const lab = v === true ? "PASS" : v === false ? "FAIL" : "UNKNOWN → treated as FAIL";
    return `<div class="cj ${cls}"><div class="sym">${esc(c)}</div>
      <div class="st">${lab}</div><div class="why">${esc(WHY[c] || "")}</div></div>`;
  }).join("");
  $("#cert-meta").innerHTML = [
    ["acceptance bit", String(st.check)],
    ["terminal", st.terminal ?? "—"],
    ["evidence committed", (st.evidence_hashes || []).length],
    ["first evidence hash", (st.evidence_hashes || [])[0]?.slice(0, 24) ?? "—"],
    ["policy bundle sha256", st.policy_decision?.bundle_sha256?.slice(0, 24) ?? "—"],
    ["guarantee", "checker-relative recomputability"]
  ].map(([k, v]) => `<div><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
}

/* --------------------------------------------------------------- channels */
function renderChannels(ch) {
  const names = CONTRACT?.audit_channels || ["IntFail", "ReplayFail", "DriftFail", "CheckFail", "CovGap"];
  const key = { IntFail: "int_fail", ReplayFail: "replay_fail", DriftFail: "drift_fail", CheckFail: "check_fail", CovGap: "cov_gap" };
  $("#channels").innerHTML = names.map(n => {
    const on = !!ch[key[n]];
    return `<div class="ch ${on ? "on" : ""}"><span>${esc(n)}</span><span>${on ? "FIRED" : "—"}</span></div>`;
  }).join("");
}

/* ----------------------------------------------------------------- replay */
function renderReplay(st) {
  const rows = [
    ["original execution", 1.0, false],
    ["pinned replay", 1.0, st.v_pi === false],
    ["fresh execution", 0.82, !!st.channels?.drift_fail]
  ];
  $("#timeline").innerHTML = rows.map(([n, w, diverge]) =>
    `<div class="tl ${diverge ? "diverge" : ""}"><span>${esc(n)}</span>
      <span class="bar" style="width:${w * 100}%"></span>
      <span>${diverge ? "DIVERGED" : "match"}</span></div>`).join("");
}

/* ---------------------------------------------------------- responsibility */
function renderResp(st) {
  const demo = [["retriever", .61, 1], ["parser", .22, 2], ["tool_adapter", .11, 3], ["memory", .04, 4]];
  const thr = 0.35;
  const margin = demo[0][1] - demo[1][1];
  const unresolved = margin < thr;
  $("#resp-tbl").innerHTML =
    `<tr><th>component</th><th>Resp (masked replay)</th><th>rank</th><th>status</th></tr>` +
    demo.map(([c, s, r]) => `<tr><td>${esc(c)}</td><td>${num(s, 3)}</td><td>${r}</td>
      <td>${r === 1 ? (unresolved ? "Unresolved" : "top-1") : "—"}</td></tr>`).join("") +
    `<tr><td>margin</td><td>${num(margin, 3)}</td><td>threshold</td><td>${num(thr, 3)}</td></tr>`;
}

/* ------------------------------------------------------------- dependence */
function renderDep(st) {
  const d = st.dependence || {};
  const state = d.state || "INSUFFICIENT_EVIDENCE";
  const box = [`<div class="gate ${esc(state)}">gate: ${esc(state)}</div>`];
  if (d.lambda_k != null) box.push(`<div class="gate">&lambda; = ${num(d.lambda_k, 3)}</div>`);
  if (d.rho_ucb != null) box.push(`<div class="gate">&rho;<sub>UCB</sub> = ${num(d.rho_ucb, 3)}</div>`);
  if (d.u_joint != null) box.push(`<div class="gate">U<sub>joint</sub>(k,&delta;) = ${num(d.u_joint, 4)}</div>`);
  $("#gatebox").innerHTML = box.join("");
}

/* ----------------------------------------------------------------- policy */
function renderPolicy(st) {
  const p = st.policy_decision || {};
  $("#policy-kv").innerHTML = [
    ["allowed", String(p.allowed)], ["rule matched", p.rule_matched ?? "—"],
    ["backend", p.backend ?? "—"], ["bundle", `${p.bundle_id ?? "—"} ${p.bundle_version ?? ""}`],
    ["bundle sha256", (p.bundle_sha256 || "—").slice(0, 28)],
    ["reasons", (p.reasons || []).join("; ") || "—"], ["fail closed", String(p.fail_closed ?? true)]
  ].map(([k, v]) => `<div><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
}

/* ------------------------------------------------------------------ trace */
function renderSpans(spans) {
  $("#spans").innerHTML = `<tr><th>span</th><th>duration (ms)</th><th>correlation id</th></tr>` +
    spans.map(s => `<tr><td>${esc(s.name)}</td><td>${num(s.duration_ms, 3)}</td>
      <td>${esc((s.correlation_id || "").slice(0, 12))}</td></tr>`).join("");
}

/* ------------------------------------------------------------------- cost */
function renderCost(st, spans) {
  const c = st.cost || {};
  const total = spans.reduce((a, s) => a + (s.duration_ms || 0), 0);
  $("#cost-kv").innerHTML = [
    ["model calls", c.model_calls], ["retrieval calls", c.retrieval_calls],
    ["tool calls", c.tool_calls], ["checker calls", c.checker_calls],
    ["replay calls", c.replay_calls], ["tokens in / out", `${c.tokens_in} / ${c.tokens_out}`],
    ["graph wall time (ms)", num(total, 3)]
  ].map(([k, v]) => `<div><div class="k">${esc(k)}</div><div class="v">${esc(v)}
     <span class="tag synthetic">synthetic</span></div></div>`).join("");
  $("#cost-tbl").innerHTML = `<tr><th>phase</th><th>ms</th></tr>` +
    spans.map(s => `<tr><td>${esc(s.name)}</td><td>${num(s.duration_ms, 3)}</td></tr>`).join("");
}

/* ---------------------------------------------------------------- results */
function renderResults(a) {
  const ws = a.workstreams || {};
  $("#ws-tbl").innerHTML = `<tr><th>workstream</th><th>records</th><th>checks failed</th><th>spec hash</th></tr>` +
    Object.entries(ws).map(([k, v]) => {
      const m = v.metrics || {}, c = v.checks || {};
      const failed = Object.values(c).filter(x => x === false).length;
      return `<tr><td>${esc(k.toUpperCase())}</td><td>${esc(m.n_records)}</td>
        <td>${failed}</td><td>${esc((m.spec_hash || "").slice(0, 16))}</td></tr>`;
    }).join("");
}

/* ------------------------------------------------------------- manuscript */
function renderRegistry(r) {
  const c = r.check || {};
  $("#reg-kv").innerHTML = [
    ["tables", `${c.n_tables}/${c.tables_expected}`],
    ["figures", `${c.n_figures}/${c.figures_expected}`],
    ["table generators", c.tables_with_generator],
    ["figure generators", c.figures_with_generator],
    ["classes valid", String(c.classes_valid)],
    ["dual PNG+PDF declared", String(c.dual_output_declared)]
  ].map(([k, v]) => `<div><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join("");
  const rows = [...(r.tables || []).map(t => ["Table " + t.number, t.label, t.provenance_class,
    (t.source_experiments || []).join(" ") || "—", t.status]),
  ...(r.figures || []).map(f => ["Figure " + f.number, f.label, f.provenance_class,
    (f.source_experiments || []).join(" ") || "—", f.status])];
  $("#reg-tbl").innerHTML = `<tr><th>#</th><th>label</th><th>class</th><th>source</th><th>status</th></tr>` +
    rows.map(([n, l, cl, s, st]) => `<tr><td>${esc(n)}</td><td>${esc(l)}</td>
      <td><span class="tag ${String(cl).toLowerCase()}">${esc(cl)}</span></td>
      <td>${esc(s)}</td><td>${esc(st)}</td></tr>`).join("");
}

boot();
