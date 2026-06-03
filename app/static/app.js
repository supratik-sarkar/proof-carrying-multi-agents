/* =================================================================
   PCG-MAS · static frontend
   - Single source of truth: window.__pcgState
   - Live Run is the writer; every other tab is a pure reader
   - Renderers are idempotent and pure: re-running them is always safe
   ================================================================= */

(function () {
  'use strict';

  // -----------------------------------------------------------------
  // Canonical frontend state (Phase 6 contract)
  // -----------------------------------------------------------------
  window.__pcgState = {
    run_id: null,
    cert_id: null,
    question: "",
    input_mode: "text",            // text | url | file
    backend_label: "",
    started_at: null,
    last_completed_at: null,
    evidence: [],
    tools: [],
    claims: [],
    claim_certificates: [],
    responsibility: null,
    audit_envelopes: null,
    risk: null,
    certificate: null,
    events: [],                    // append-only SSE event trace
    errors: [],
    status: "idle",                // idle | running | complete | error
  };

  const CHANNELS = ["V_I", "V_R", "V_D", "V_Ch", "V_Cov"];
  const CHANNEL_LABELS = {
    V_I:   "Integrity",
    V_R:   "Replay",
    V_D:   "Drift",
    V_Ch:  "Checker",
    V_Cov: "Coverage",
  };

  const RISK_TAG_CLASS = {
    Answer:   "tag-answer",
    Verify:   "tag-verify",
    Escalate: "tag-escalate",
    Refuse:   "tag-refuse",
  };

  // =================================================================
  // PCG comparison-card components (Results tab)
  // Pure functions of payload → HTML. Idempotent. Reusable.
  // =================================================================

  // 1. Certificate header badge — sha256 + cert ID, always visible if cert exists.
  function pcgCardHeader(pcg) {
    if (!pcg) return "";
    const action = pcg.risk_action || "—";
    const accepted = pcg.accepted === true;
    const certId = pcg.cert_id || "";
    const hash = (pcg.integrity_hash || "").slice(0, 16);
    return '' +
      '<div class="pcg-card-header">' +
      '  <div class="pcg-card-header-row">' +
      '    <span class="risk-tag ' + (RISK_TAG_CLASS[action] || "") + '">' + escapeHtml(action) + '</span>' +
      '    <span class="pcg-cert-id" title="Certificate ID + integrity hash">' +
      '      <code>' + escapeHtml(certId) + '</code>' +
      (hash ? ' · <code>sha256:' + escapeHtml(hash) + '…</code>' : '') +
      '    </span>' +
      '  </div>' +
      '</div>';
  }

  // 2. Inline 5-channel pass strip — aggregate across all claims.
  //    Shows immediately whether each channel held over the run.
  function pcgChannelStrip(pcg) {
    if (!pcg || !pcg.attribution) return "";
    // Aggregate per channel across all claims
    const agg = {};
    for (const ch of CHANNELS) agg[ch] = "idle";
    if (pcg.attribution.length === 0) return "";
    for (const ch of CHANNELS) {
      let allPass = true, anyFail = false, anySkip = false;
      for (const a of pcg.attribution) {
        const s = a.channels && a.channels[ch] && a.channels[ch].state;
        if (!s) { allPass = false; continue; }
        if (s === "fail") anyFail = true;
        else if (s === "skip") anySkip = true;
        if (s !== "pass") allPass = false;
      }
      agg[ch] = anyFail ? "fail" : allPass ? "pass" : anySkip ? "skip" : "pending";
    }
    const pills = CHANNELS.map(ch =>
      '<span class="chan-pill pcg-mini" data-state="' + agg[ch] + '">' +
      ch + ' ' + stateGlyph(agg[ch]) +
      '</span>'
    ).join("");
    return '<div class="pcg-chan-strip" aria-label="five-channel verdict">' + pills + '</div>';
  }

  // 3. Per-claim attribution list — "Claim 1 ✓ supported by e1 (source)".
  //    Renders every claim with its accept/reject state and supporting evidence.
  function pcgAttributionList(pcg) {
    if (!pcg || !pcg.attribution || pcg.attribution.length === 0) {
      // Even when there are no claims, communicate that explicitly
      return '<div class="pcg-attribution-empty muted small">No atomic claims were extracted from this run.</div>';
    }
    const rows = pcg.attribution.map((a, i) => {
      const idx = String(i + 1).padStart(2, "0");
      const verdict = a.accepted
        ? '<span class="pcg-attr-mark pass">✓</span>'
        : '<span class="pcg-attr-mark fail">✕</span>';
      const supports = (a.supports && a.supports.length)
        ? a.supports.map(s =>
            '<span class="pcg-support-chip" title="' + escapeHtml(s.snippet || "") + '">' +
            '<code>' + escapeHtml(s.id) + '</code> · ' + escapeHtml(s.source) +
            '</span>'
          ).join("")
        : '<span class="muted small">no cited evidence</span>';
      return '' +
        '<li class="pcg-attr-item">' +
        '  <div class="pcg-attr-head">' + verdict +
        '    <span class="pcg-attr-id">Claim ' + idx + '</span>' +
        '  </div>' +
        '  <div class="pcg-attr-text">' + escapeHtml(a.claim_text || "") + '</div>' +
        '  <div class="pcg-attr-supports">' + supports + '</div>' +
        '</li>';
    }).join("");
    return '<ul class="pcg-attribution-list">' + rows + '</ul>';
  }

  // 4. "Why this matters" auto-generated narrative — reads payload, describes
  //    in human terms what PCG-MAS verified vs what the raw LLM can/can't show.
  //    Programmatic, not Fermat-specific. Branches by:
  //      - whether raw answer text matches PCG answer text (or seems to)
  //      - which channels passed/failed
  //      - risk action chosen
  function pcgWhyThisMatters(pcg, raw) {
    if (!pcg) return "";
    const rawText = (raw && raw.answer) || "";
    const pcgText = pcg.answer || "";
    const sameish = textsAreSimilar(rawText, pcgText);
    const action = pcg.risk_action || "";
    const accepted = pcg.accepted === true;
    const channels = aggregateAttributionChannels(pcg.attribution);
    const passedChannels = CHANNELS.filter(c => channels[c] === "pass");
    const failedChannels = CHANNELS.filter(c => channels[c] === "fail");
    const nClaims = pcg.n_claims || 0;
    const nAccepted = pcg.n_accepted || 0;

    let headline = "";
    let body = "";

    if (accepted && sameish) {
      headline = "Same answer — but PCG-MAS proves it is grounded.";
      const passedList = passedChannels.length
        ? passedChannels.map(c => '<code>' + c + '</code>').join(", ")
        : "none";
      body =
        "Both the raw LLM and PCG-MAS produced this answer. " +
        "The raw side has no machinery to attest correctness; " +
        "PCG-MAS additionally certifies " + passedChannels.length + " of 5 channels (" + passedList + "), " +
        "cites the supporting evidence per claim, and emits a tamper-evident SHA-256 over the certificate.";
    } else if (accepted && !sameish) {
      headline = "Both answered — but with different text.";
      body =
        "PCG-MAS accepted the answer through the 5-channel checker. " +
        "Its version is the one carrying the certificate. Inspect both texts to see where they diverge.";
    } else if (!accepted && action === "Verify") {
      headline = "PCG-MAS asks for a verification pass; raw LLM does not.";
      body =
        "Raw LLM committed to an answer with no verification machinery. " +
        "PCG-MAS detected enough channel risk to require a second-pass check " +
        (failedChannels.length ? "(failing: " + failedChannels.map(c => '<code>' + c + '</code>').join(", ") + ") " : "") +
        "before committing the same answer.";
    } else if (!accepted && action === "Escalate") {
      headline = "PCG-MAS escalates; raw LLM would have answered anyway.";
      body =
        "Channel verdicts indicate the answer cannot be self-certified. PCG-MAS recommends human or stronger-model review. " +
        (failedChannels.length ? "Failed channels: " + failedChannels.map(c => '<code>' + c + '</code>').join(", ") + ". " : "") +
        "Raw LLM, lacking these audits, has no warning to surface.";
    } else if (!accepted && action === "Refuse") {
      headline = "PCG-MAS refuses; raw LLM has no choice but to guess.";
      const why = failedChannels.length
        ? failedChannels.map(c => '<code>' + c + '</code>').join(", ")
        : "no channel reached acceptance";
      body =
        "The 5-channel checker found the support insufficient to attest the answer " +
        "(failed: " + why + "). PCG-MAS withholds the answer rather than emitting an unverifiable one. " +
        "The raw LLM has no equivalent mechanism — it answers regardless.";
    } else {
      headline = "PCG-MAS verifies; raw LLM answers blindly.";
      body =
        "PCG-MAS evaluated " + nClaims + " atomic claim(s), accepted " + nAccepted + ", " +
        "and reached action " + escapeHtml(action) + ". Raw LLM produces text only.";
    }

    return '' +
      '<div class="pcg-why">' +
      '  <p class="pcg-why-headline">' + headline + '</p>' +
      '  <p class="pcg-why-body">' + body + '</p>' +
      '</div>';
  }

  // Helper: aggregate channel pass/fail across attribution entries
  function aggregateAttributionChannels(attribution) {
    const agg = {};
    for (const ch of CHANNELS) agg[ch] = "idle";
    if (!attribution || attribution.length === 0) return agg;
    for (const ch of CHANNELS) {
      let allPass = true, anyFail = false, anySkip = false;
      for (const a of attribution) {
        const s = a.channels && a.channels[ch] && a.channels[ch].state;
        if (!s) { allPass = false; continue; }
        if (s === "fail") anyFail = true;
        else if (s === "skip") anySkip = true;
        if (s !== "pass") allPass = false;
      }
      agg[ch] = anyFail ? "fail" : allPass ? "pass" : anySkip ? "skip" : "pending";
    }
    return agg;
  }

  // Helper: cheap text similarity for the "same answer" branch.
  // Tokenize lowercase alphanumeric, drop stopwords, compute Jaccard ≥ 0.55.
  function textsAreSimilar(a, b) {
    if (!a || !b) return false;
    const STOP = new Set([
      "a","an","the","is","was","were","be","by","of","in","on","at","to",
      "and","or","for","with","as","its","it","this","that","these","those",
      "have","has","had","not","no","yes",
    ]);
    const tok = s => new Set(
      String(s).toLowerCase().match(/[a-z0-9]+/g)?.filter(t => !STOP.has(t) && t.length > 1) || []
    );
    const A = tok(a), B = tok(b);
    if (A.size === 0 || B.size === 0) return false;
    let inter = 0;
    for (const t of A) if (B.has(t)) inter++;
    const union = A.size + B.size - inter;
    return (inter / union) >= 0.55;
  }

  // Top-level assembler — composes the four components in order.
  function renderPcgComparisonCard(pcg, raw, tokenInfo) {
    if (!pcg) return '<p class="muted small">No PCG-MAS payload.</p>';
    const ans = pcg.answer || "(no answer)";
    return '' +
      pcgCardHeader(pcg) +
      pcgChannelStrip(pcg) +
      '<div class="comparison-answer pcg-answer">' + escapeHtml(ans).replace(/\n/g, "<br>") + '</div>' +
      pcgAttributionList(pcg) +
      pcgWhyThisMatters(pcg, raw) +
      '<div class="comparison-meta pcg-meta">' +
        '<span class="token-badge token-overhead">~' + tokenInfo.pcgTokens + ' tokens (+' + tokenInfo.overhead + '% vs raw)</span>' +
        '<span class="muted small">' + (pcg.n_claims || 0) + ' claim(s) · ' + (pcg.n_accepted || 0) + ' accepted · ' + (pcg.n_evidence || 0) + ' evidence item(s)</span>' +
      '</div>';
  }

  // ===================================================================
  // STATE MUTATIONS — only the SSE handler calls these
  // ===================================================================
  function resetStateForRun(meta) {
    window.__pcgState = {
      ...window.__pcgState,
      run_id: meta.run_id,
      cert_id: null,
      question: meta.question,
      input_mode: meta.input_mode,
      backend_label: meta.backend_label,
      started_at: new Date().toISOString(),
      last_completed_at: null,
      evidence: [],
      tools: [],
      claims: [],
      claim_certificates: [],
      responsibility: null,
      audit_envelopes: null,
      risk: null,
      certificate: null,
      events: [],
      errors: [],
      status: "running",
    };
  }

  function applyEvent(ev) {
    const S = window.__pcgState;
    S.events.push(ev);
    const t = ev.event;
    const d = ev.data || {};

    if (t === "start") {
      S.cert_id = d.cert_id || null;
    } else if (t === "evidence") {
      S.evidence = d.items || [];
    } else if (t === "claim") {
      S.claims.push(d);
    } else if (t === "claim_cert") {
      // Replace existing entry with same claim_id (idempotent under replay)
      const cid = d.claim && d.claim.claim_id;
      const i = S.claim_certificates.findIndex(c => c.claim && c.claim.claim_id === cid);
      if (i >= 0) S.claim_certificates[i] = d;
      else S.claim_certificates.push(d);
    } else if (t === "responsibility") {
      S.responsibility = S.responsibility || [];
      const cid = d.claim_id;
      const i = S.responsibility.findIndex(r => r.claim_id === cid);
      if (i >= 0) S.responsibility[i] = d;
      else S.responsibility.push(d);
    } else if (t === "audit_envelope") {
      S.audit_envelopes = S.audit_envelopes || [];
      const ch = d.channel;
      const i = S.audit_envelopes.findIndex(e => e.channel === ch);
      if (i >= 0) S.audit_envelopes[i] = d;
      else S.audit_envelopes.push(d);
    } else if (t === "risk") {
      S.risk = d;
    } else if (t === "certificate") {
      S.certificate = d;
    } else if (t === "done") {
      S.status = "complete";
      S.last_completed_at = new Date().toISOString();
    } else if (t === "error") {
      S.status = "error";
      S.errors.push(d);
    }
    // 'channel' events are captured in S.events but the aggregate state for
    // them is derived from claim_certificates in renderers (single source of truth)
  }

  // ===================================================================
  // RENDERERS — pure functions of state. Idempotent. Run any time.
  // ===================================================================

  // -------- Live Run: aggregate pipeline chips & flow nodes --------
  function aggregateChannelState(state) {
    // Pass if ALL claims have that channel == pass.
    // Fail if ANY claim has that channel == fail.
    // Pending if any are pending and none failed.
    // Idle if no claim_certificates yet.
    const agg = {};
    for (const ch of CHANNELS) agg[ch] = "idle";
    if (!state.claim_certificates.length) {
      // Use latest channel events from S.events to show pending state
      const last = {};
      for (const ev of state.events) {
        if (ev.event === "channel" && ev.data) {
          last[ev.data.channel] = ev.data.state;
        }
      }
      for (const ch of CHANNELS) {
        if (last[ch]) agg[ch] = last[ch];
      }
      return agg;
    }
    for (const ch of CHANNELS) {
      let allPass = true;
      let anyFail = false;
      let anySkip = false;
      for (const cc of state.claim_certificates) {
        const v = cc.channels && cc.channels[ch];
        if (!v) { allPass = false; continue; }
        if (v.state === "fail") anyFail = true;
        if (v.state === "skip") anySkip = true;
        if (v.state !== "pass") allPass = false;
      }
      if (anyFail) agg[ch] = "fail";
      else if (allPass) agg[ch] = "pass";
      else if (anySkip) agg[ch] = "skip";   // upstream failed; not pending
      else agg[ch] = "pending";
    }
    return agg;
  }

  function renderAggregatePipeline(state) {
    const agg = aggregateChannelState(state);
    for (const ch of CHANNELS) {
      const chip = document.querySelector(`.status-chip[data-channel="${ch}"]`);
      if (chip) chip.setAttribute("data-state", agg[ch]);
      const node = document.querySelector(`.node[data-node="${ch}"]`);
      if (node) node.setAttribute("data-state", agg[ch]);
    }
    // Edges 0..4 light up as their target channel transitions; edge 5 is to risk
    const edgeForChannel = { V_I: 0, V_R: 1, V_D: 2, V_Ch: 3, V_Cov: 4 };
    for (const ch of CHANNELS) {
      const idx = edgeForChannel[ch];
      const edge = document.querySelector(`.edge[data-edge="${idx}"]`);
      if (edge) edge.setAttribute("data-state", agg[ch] === "idle" ? "idle" : agg[ch]);
    }
    // Final edge & risk node
    const riskEdge = document.querySelector('.edge[data-edge="5"]');
    const riskNode = document.querySelector('.node[data-node="risk"]');
    const riskLabel = document.getElementById("risk-action-label");
    const certHash = document.getElementById("cert-hash");
    if (state.risk) {
      const action = state.risk.action || "Risk";
      const passLike = action === "Answer";
      if (riskEdge) riskEdge.setAttribute("data-state", passLike ? "pass" : "fail");
      if (riskNode) riskNode.setAttribute("data-state", passLike ? "pass" : "fail");
      if (riskLabel) riskLabel.textContent = action;
    } else {
      if (riskEdge) riskEdge.setAttribute("data-state", "idle");
      if (riskNode) riskNode.setAttribute("data-state", "idle");
      if (riskLabel) riskLabel.textContent = "Risk";
    }
    if (certHash) {
      const h = (state.certificate && state.certificate.integrity_hash) || "";
      certHash.textContent = h ? "#" + h.slice(0, 8) : "SHA-256";
    }
  }

  // -------- Live Run: summary strip --------
  function renderSummaryStrip(state) {
    const strip = document.getElementById("summary-strip");
    const N = state.claims.length;
    if (!N) {
      if (strip) strip.setAttribute("hidden", "");
      return;
    }
    if (strip) strip.removeAttribute("hidden");
    const accepted = state.claim_certificates.filter(c => c.accepted).length;
    const action = (state.risk && state.risk.action) || "—";
    const r = (state.risk && typeof state.risk.posterior_risk === "number")
              ? state.risk.posterior_risk.toFixed(3) : "—";
    const tokens = (state.certificate && state.certificate.meta &&
                    state.certificate.meta.tokens_total) || 0;
    setText("sum-claims", String(N));
    setText("sum-accepted", String(accepted));
    setText("sum-action", action);
    setText("sum-risk", r);
    setText("sum-tokens", String(tokens));
    const sumAction = document.getElementById("sum-action");
    if (sumAction) {
      sumAction.className = "summary-value " + (RISK_TAG_CLASS[action] || "");
    }
  }

  // -------- Live Run: per-claim cards --------
  function renderClaimsList(state) {
    const wrap = document.getElementById("claims-list");
    if (!wrap) return;
    if (!state.claims.length) {
      wrap.innerHTML = '<p class="muted small">Run PCG-MAS to extract atomic claims and audit them along the 5 channels.</p>';
      return;
    }
    const certs = Object.fromEntries(
      state.claim_certificates.map(c => [c.claim && c.claim.claim_id, c])
    );
    wrap.innerHTML = state.claims.map((c, i) => {
      const cc = certs[c.claim_id] || null;
      return renderClaimCard(c, cc, i + 1);
    }).join("");
  }

  function renderClaimCard(claim, cert, idx) {
    const chRow = CHANNELS.map(ch => {
      const v = cert && cert.channels && cert.channels[ch];
      const state = v ? v.state : "pending";
      return `<span class="chan-pill" data-state="${state}">${ch} ${stateGlyph(state)}</span>`;
    }).join("");

    const verdict = !cert ? "pending"
                  : cert.accepted ? "Accepted" : "Rejected";
    const verdictClass = !cert ? "pending"
                      : cert.accepted ? "pass" : "fail";

    const supports = (claim.support_ids || []).join(", ") ||
                     (claim.tool_output_ids || []).join(", ") ||
                     "(no cited evidence)";
    const hashShort = cert && cert.integrity_hash
                     ? cert.integrity_hash.slice(0, 16) + "…" : "—";

    // Detail block: collect failed channel reasons if any
    const failDetails = !cert ? "" :
      CHANNELS.map(ch => {
        const v = cert.channels && cert.channels[ch];
        if (!v) return "";
        if (v.state === "fail" || v.state === "pass") {
          return `<li><code>${ch}</code> · ${v.state} · ${escapeHtml(v.detail || "")}</li>`;
        }
        return `<li><code>${ch}</code> · ${v.state}</li>`;
      }).join("");

    return `
      <div class="claim-card" data-claim-id="${escapeHtml(claim.claim_id)}">
        <div class="claim-card-head">
          <span class="claim-id">Claim ${String(idx).padStart(2, "0")}</span>
          <span class="claim-verdict ${verdictClass}">${verdict}</span>
        </div>
        <p class="claim-text">${escapeHtml(claim.claim_text)}</p>
        <div class="chan-row">${chRow}</div>
        <div class="claim-meta">
          <span><strong>Evidence:</strong> ${escapeHtml(supports)}</span>
          <span class="claim-hash"><strong>Hash:</strong> <code>${escapeHtml(hashShort)}</code></span>
        </div>
        <details class="claim-details">
          <summary>Channel details</summary>
          <ul class="claim-details-list">${failDetails || '<li class="muted">No channel data yet.</li>'}</ul>
        </details>
      </div>
    `;
  }

  function stateGlyph(s) {
    switch (s) {
      case "pass": return "✓";
      case "fail": return "✕";
      case "skip": return "—";
      case "pending": return "…";
      default: return "·";
    }
  }

  // -------- Live Run: result banner --------
  function renderResultBanner(state) {
    const banner = document.getElementById("result-banner");
    const badge  = document.getElementById("result-badge");
    const ratEl  = document.getElementById("result-rationale");
    const ansEl  = document.getElementById("result-answer");
    const metaEl = document.getElementById("result-meta");
    if (!state.risk && !state.certificate) {
      if (banner) banner.setAttribute("hidden", "");
      return;
    }
    if (banner) banner.removeAttribute("hidden");
    const action = (state.risk && state.risk.action) || "—";
    const isAnswer = action === "Answer";
    if (banner) banner.setAttribute("data-decision", isAnswer ? "accept" : "reject");
    if (badge) {
      badge.textContent = action;
      badge.className = "badge " + (isAnswer ? "pass" : "fail");
    }
    const summary = (state.risk && state.risk.summary) || "";
    if (ratEl) ratEl.textContent = summary;
    const ans = state.certificate && (
      isAnswer ? (state.certificate.answer_final || state.certificate.answer_draft) :
                 (state.certificate.answer_final || "(answer withheld)")
    );
    if (ansEl) ansEl.textContent = ans || "";
    if (metaEl) {
      const cid = state.cert_id || "";
      const ih  = state.certificate && state.certificate.integrity_hash
                 ? state.certificate.integrity_hash.slice(0, 24) : "";
      metaEl.textContent = (cid ? cid + " · " : "") + (ih ? "sha256:" + ih + "…" : "");
    }
  }

  // -------- Certificate Inspector --------
  function renderCertificate(state) {
    const stale = document.getElementById("cert-stale-banner");
    const gridWrap = document.getElementById("cert-grid-wrap");
    const jsonWrap = document.getElementById("cert-json-wrap");
    if (!state.certificate) {
      if (stale) {
        stale.textContent = "No certificate yet. Run PCG-MAS first.";
        stale.removeAttribute("hidden");
      }
      if (gridWrap) gridWrap.setAttribute("hidden", "");
      if (jsonWrap) jsonWrap.setAttribute("hidden", "");
      return;
    }
    if (stale) stale.setAttribute("hidden", "");
    if (gridWrap) gridWrap.removeAttribute("hidden");
    if (jsonWrap) jsonWrap.removeAttribute("hidden");

    // Dense per-claim grid
    const body = document.getElementById("cert-grid-body");
    if (body) {
      body.innerHTML = state.claim_certificates.map(cc => {
        const ch = cc.channels || {};
        const sym = s => stateGlyph(s);
        const action = (state.risk && state.risk.action) || "—";
        const hash = (cc.integrity_hash || "").slice(0, 10) + "…";
        return `
          <tr>
            <td><code>${escapeHtml(cc.claim && cc.claim.claim_id || "")}</code></td>
            <td class="state-${(ch.V_I  || {}).state || "idle"}">${sym((ch.V_I  || {}).state || "idle")}</td>
            <td class="state-${(ch.V_R  || {}).state || "idle"}">${sym((ch.V_R  || {}).state || "idle")}</td>
            <td class="state-${(ch.V_D  || {}).state || "idle"}">${sym((ch.V_D  || {}).state || "idle")}</td>
            <td class="state-${(ch.V_Ch || {}).state || "idle"}">${sym((ch.V_Ch || {}).state || "idle")}</td>
            <td class="state-${(ch.V_Cov|| {}).state || "idle"}">${sym((ch.V_Cov|| {}).state || "idle")}</td>
            <td>${cc.accepted ? `<span class="risk-tag ${RISK_TAG_CLASS[action] || ""}">${action}</span>` : `<span class="risk-tag tag-refuse">Refuse</span>`}</td>
            <td><code>${escapeHtml(hash)}</code></td>
          </tr>
        `;
      }).join("");
    }

    // Full JSON
    const codeEl = document.getElementById("cert-block-code");
    if (codeEl) {
      codeEl.textContent = JSON.stringify(state.certificate, null, 2);
    }
  }

  // -------- Responsibility --------
  function renderResponsibility(state) {
    const stale = document.getElementById("resp-stale-banner");
    const list  = document.getElementById("resp-list");

    // Compute progress: number of accepted claims that need responsibility vs done so far
    const acceptedClaims = state.claim_certificates.filter(c => c.accepted &&
      (c.claim.support_ids.length > 0 || c.claim.tool_output_ids.length > 0));
    const expectedReports = acceptedClaims.length;
    const doneReports = (state.responsibility || []).length;
    const running = state.status === "running";

    if (running && expectedReports > 0 && doneReports < expectedReports) {
      // Show in-flight progress
      if (stale) {
        const pct = Math.round((doneReports / expectedReports) * 100);
        stale.removeAttribute("hidden");
        stale.className = "muted small stale-banner";
        stale.innerHTML =
          'Computing mask-and-replay responsibility: <strong>' + doneReports +
          ' / ' + expectedReports + '</strong> claims complete · ' +
          '<span style="display:inline-block;vertical-align:middle;margin-left:0.4rem;width:160px;height:6px;background:var(--surface-2);border-radius:999px;overflow:hidden">' +
          '<span style="display:block;height:100%;width:' + pct + '%;background:var(--accent);transition:width 0.4s var(--ease)"></span>' +
          '</span>';
      }
    } else if (!state.responsibility || !state.responsibility.length) {
      if (stale) {
        if (state.status === "complete") {
          stale.textContent = "Latest run did not produce responsibility reports (no accepted claims with cited evidence).";
        } else {
          stale.textContent = "No results yet. Run PCG-MAS first.";
        }
        stale.className = "muted small stale-banner";
        stale.removeAttribute("hidden");
      }
      if (list) list.innerHTML = "";
      return;
    } else {
      if (stale) {
        stale.removeAttribute("hidden");
        stale.textContent = `Showing latest completed run: ${state.cert_id || "—"} · ${formatStamp(state.last_completed_at)}`;
        stale.className = "muted small stale-banner pcg-stale-success";
      }
    }

    if (!list) return;
    if (!state.responsibility || !state.responsibility.length) return;

    list.innerHTML = state.responsibility.map(rep => {
      const rows = (rep.scores || []).map(s => {
        const w = Math.max(2, Math.round(Math.max(0, s.score) * 100));
        const ciW = Math.max(2, Math.round((s.ci_high - s.ci_low) * 100));
        return `
          <tr>
            <td><code>${escapeHtml(s.component_id)}</code></td>
            <td><span class="component-type-${escapeHtml(s.component_type)}">${escapeHtml(s.component_type)}</span></td>
            <td class="resp-bar-cell">
              <div class="resp-bar"><div class="resp-bar-fill" style="width:${w}%"></div></div>
            </td>
            <td><code>${s.score.toFixed(3)}</code></td>
            <td><code>[${s.ci_low.toFixed(2)}, ${s.ci_high.toFixed(2)}]</code></td>
            <td>${s.rank}</td>
          </tr>
        `;
      }).join("");
      return `
        <div class="resp-card">
          <div class="resp-card-head">
            <span class="resp-claim-id">Claim <code>${escapeHtml(rep.claim_id)}</code></span>
            <span class="muted small">M=${rep.n_replays} · rank-recovery P ≥ ${rep.rank_recovery_prob.toFixed(2)} · CI=${escapeHtml(rep.ci_method)}</span>
          </div>
          <p class="muted small">Top responsible: <code>${escapeHtml(rep.top_responsible_id || "(none)")}</code></p>
          <table class="resp-table">
            <thead><tr><th>Component</th><th>Type</th><th>Score</th><th>Resp̂</th><th>CI</th><th>Rank</th></tr></thead>
            <tbody>${rows || `<tr><td colspan="6" class="muted">No components evaluated.</td></tr>`}</tbody>
          </table>
        </div>
      `;
    }).join("");
  }

  // -------- Risk Controller --------
  function renderRiskController(state) {
    const stale = document.getElementById("risk-stale-banner");
    const wrap  = document.getElementById("risk-decision");
    if (!state.risk) {
      if (stale) {
        stale.textContent = "No decision yet. Run PCG-MAS first.";
        stale.removeAttribute("hidden");
      }
      if (wrap) wrap.setAttribute("hidden", "");
    } else {
      if (stale) {
        stale.removeAttribute("hidden");
        stale.textContent = `Showing latest completed run: ${state.cert_id || "—"} · ${formatStamp(state.last_completed_at)}`;
        stale.className = "muted small stale-banner pcg-stale-success";
      }
      if (wrap) wrap.removeAttribute("hidden");
      const r = state.risk;
      setText("risk-action-pill", r.action || "—");
      const pill = document.getElementById("risk-action-pill");
      if (pill) {
        pill.className = "risk-action-pill " + (RISK_TAG_CLASS[r.action] || "");
      }
      setText("risk-r-value", typeof r.posterior_risk === "number"
              ? r.posterior_risk.toFixed(3) : "—");
      setText("risk-dominant", r.dominant_failure_channel || "—");
      setText("risk-summary", r.summary || "");
      // Cost table
      const tbody = document.getElementById("cost-table-body");
      if (tbody) {
        const order = ["Answer", "Verify", "Escalate", "Refuse"];
        tbody.innerHTML = order.map(a => {
          const c = (r.expected_cost || {})[a];
          const rr = (r.residual_risk || {})[a];
          const chosen = a === r.action;
          return `
            <tr class="${chosen ? "chosen-action" : ""}">
              <td><span class="risk-tag ${RISK_TAG_CLASS[a] || ""}">${a}</span></td>
              <td><code>${typeof c === "number" ? c.toFixed(4) : "—"}</code></td>
              <td><code>${typeof rr === "number" ? rr.toFixed(3) : "—"}</code></td>
            </tr>
          `;
        }).join("");
      }
      const reasons = document.getElementById("risk-reasons");
      if (reasons) {
        const rc = r.reason_codes || [];
        reasons.innerHTML = rc.length ? "Reasons: " + rc.map(x => `<code>${escapeHtml(x)}</code>`).join(" · ") : "";
      }
    }

    // Envelopes
    const envStale = document.getElementById("env-stale-banner");
    const envList  = document.getElementById("env-list");
    if (!state.audit_envelopes || !state.audit_envelopes.length) {
      if (envStale) envStale.removeAttribute("hidden");
      if (envList) envList.innerHTML = "";
    } else {
      if (envStale) envStale.setAttribute("hidden", "");
      if (envList) {
        envList.innerHTML = state.audit_envelopes.map(e => {
          const w = Math.round(e.pass_rate * 100);
          const ciL = Math.round(e.ci_low * 100);
          const ciH = Math.round(e.ci_high * 100);
          return `
            <div class="envelope-row">
              <div class="env-head">
                <span class="env-ch"><code>${escapeHtml(e.channel)}</code> · ${CHANNEL_LABELS[e.channel] || ""}</span>
                <span class="env-rate"><strong>${(e.pass_rate * 100).toFixed(1)}%</strong></span>
              </div>
              <div class="env-bar">
                <div class="env-bar-ci" style="left:${ciL}%; width:${Math.max(1, ciH - ciL)}%"></div>
                <div class="env-bar-mean" style="left:${w}%"></div>
              </div>
              <div class="env-meta muted small">
                CI [${(e.ci_low * 100).toFixed(1)}%, ${(e.ci_high * 100).toFixed(1)}%]
                · n=${e.n_samples} · ${escapeHtml(e.method)}
              </div>
            </div>
          `;
        }).join("");
      }
    }
  }

  // ===================================================================
  // Master update — call after every event
  // ===================================================================
  function updateAllRenderers() {
    const S = window.__pcgState;
    renderAggregatePipeline(S);
    renderSummaryStrip(S);
    renderClaimsList(S);
    renderResultBanner(S);
    renderCertificate(S);
    renderResponsibility(S);
    renderRiskController(S);
  }

  // ===================================================================
  // Helpers
  // ===================================================================
  function setText(id, txt) {
    const el = document.getElementById(id);
    if (el) el.textContent = txt;
  }
  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({
      "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
    }[c]));
  }
  function formatStamp(iso) {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString();
    } catch (e) { return iso; }
  }

  // ===================================================================
  // Theme toggle
  // ===================================================================
  (function initTheme() {
    const KEY = "pcg-theme";
    const root = document.documentElement;
    const btn = document.getElementById("theme-toggle");
    const apply = (mode) => {
      if (mode === "system") root.removeAttribute("data-theme");
      else root.setAttribute("data-theme", mode);
      try { localStorage.setItem(KEY, mode); } catch (e) {}
      if (btn) btn.title = "Theme: " + mode + " (click to cycle)";
    };
    let saved = "system";
    try { saved = localStorage.getItem(KEY) || "system"; } catch (e) {}
    apply(saved);
    if (btn) btn.addEventListener("click", () => {
      const cur = (localStorage.getItem(KEY) || "system");
      const next = cur === "system" ? "light" : (cur === "light" ? "dark" : "system");
      apply(next);
    });
  })();

  // ===================================================================
  // Tab routing
  // ===================================================================
  function showTab(name) {
    document.querySelectorAll(".tab-panel[data-tab]").forEach(p => {
      if (p.getAttribute("data-tab") === name) p.removeAttribute("hidden");
      else p.setAttribute("hidden", "");
    });
    document.querySelectorAll('.tabs button[data-tab]').forEach(b => {
      const active = b.getAttribute("data-tab") === name;
      b.classList.toggle("is-active", active);
      b.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll(".header-nav a[data-tab]").forEach(a => {
      a.classList.toggle("is-active", a.getAttribute("data-tab") === name);
    });
    // Active tab gets a re-render from state (idempotent, cheap)
    updateAllRenderers();
  }
  (function initTabs() {
    document.querySelectorAll(".tabs button[data-tab]").forEach(b => {
      b.addEventListener("click", () => showTab(b.getAttribute("data-tab")));
    });
    document.querySelectorAll(".header-nav a[data-tab]").forEach(a => {
      a.addEventListener("click", e => {
        e.preventDefault();
        showTab(a.getAttribute("data-tab"));
      });
    });
    const hash = (location.hash || "").replace("#tab-", "");
    if (hash) showTab(hash);
  })();

  // ===================================================================
  // Backend hint
  // ===================================================================
  (function initBackendHint() {
    const select = document.getElementById("backend-select");
    const apiKey = document.getElementById("api-key");
    const hint   = document.getElementById("backend-hint");
    if (!select || !apiKey || !hint) return;
    const HINTS = {
      openai:    "Get a key at platform.openai.com/api-keys (starts with sk-…)",
      anthropic: "Get a key at console.anthropic.com/settings/keys (starts with sk-ant-…)",
      deepseek:  "Get a key at platform.deepseek.com (starts with sk-…)",
      hf:        "Use a HF read token from huggingface.co/settings/tokens (starts with hf_…)",
    };
    const update = () => {
      const v = (select.value || "").toLowerCase();
      let provider = "openai";
      if (v.includes("anthropic")) provider = "anthropic";
      else if (v.includes("deepseek")) provider = "deepseek";
      else if (v.includes("hf inference")) provider = "hf";
      apiKey.placeholder = "Paste your " + (provider === "hf" ? "hugging face" : provider) + " key…";
      hint.textContent = HINTS[provider];
    };
    select.addEventListener("change", update);
    update();
  })();

  // ===================================================================
  // Top-k slider
  // ===================================================================
  (function initSlider() {
    const slider = document.getElementById("topk");
    const out    = document.getElementById("topk-out");
    if (!slider || !out) return;
    const update = () => { out.value = slider.value; };
    slider.addEventListener("input", update);
    update();
  })();

  // ===================================================================
  // Segmented input source (Live Run)
  // ===================================================================
  (function initSegmented() {
    const buttons = document.querySelectorAll(".segmented .seg-btn[data-mode]");
    const fieldText = document.getElementById("field-context");
    const fieldUrl  = document.getElementById("field-url");
    const fieldFile = document.getElementById("field-file");
    const show = (mode) => {
      buttons.forEach(b => {
        const active = b.getAttribute("data-mode") === mode;
        b.classList.toggle("is-active", active);
        b.setAttribute("aria-selected", active ? "true" : "false");
      });
      if (fieldText) fieldText.toggleAttribute("hidden", mode !== "text");
      if (fieldUrl)  fieldUrl.toggleAttribute("hidden", mode !== "url");
      if (fieldFile) fieldFile.toggleAttribute("hidden", mode !== "file");
      window.__pcgInputMode = mode;
      window.__pcgState.input_mode = mode;
    };
    buttons.forEach(b => b.addEventListener("click", () => show(b.getAttribute("data-mode"))));
    show("text");
  })();

  // ===================================================================
  // File picker (dynamic-create approach — works on Safari + Chrome)
  // ===================================================================
  const MAX_FILE_BYTES = 10 * 1024 * 1024 * 1024;
  function formatBytes(n) {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    if (n < 1024 * 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + " MB";
    return (n / (1024 * 1024 * 1024)).toFixed(2) + " GB";
  }
  function wireDropzone(dropId, inputId, hintId) {
    const drop  = document.getElementById(dropId);
    const input = document.getElementById(inputId);
    const hint  = document.getElementById(hintId);
    if (!input) return;
    const defaultHint = hint ? hint.textContent : "";

    const accept = (file) => {
      if (file.size > MAX_FILE_BYTES) {
        if (hint) hint.innerHTML =
          '<span style="color:var(--fail)">File too large: ' + formatBytes(file.size) +
          ". Maximum is 10 GB.</span>";
        input.value = "";
        return false;
      }
      if (hint) hint.textContent = "Selected: " + file.name + " (" + formatBytes(file.size) + ")";
      return true;
    };

    // Native picker: just react to the change event
    input.addEventListener("change", () => {
      const f = input.files && input.files[0];
      if (!f) {
        if (hint) hint.textContent = defaultHint;
        return;
      }
      accept(f);
    });

    // Drag-and-drop onto the dashed dropbox
    if (drop) {
      drop.addEventListener("dragover", e => {
        e.preventDefault();
        drop.classList.add("is-dragover");
      });
      drop.addEventListener("dragleave", () => drop.classList.remove("is-dragover"));
      drop.addEventListener("drop", e => {
        e.preventDefault();
        drop.classList.remove("is-dragover");
        const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (!f) return;
        if (!accept(f)) return;
        try {
          const dt = new DataTransfer();
          dt.items.add(f);
          input.files = dt.files;
        } catch (err) { console.warn("DataTransfer", err); }
      });
    }
  }
  wireDropzone("dropzone-live", "file", "dropzone-hint");
  wireDropzone("dropzone-results", "results-file", "results-dropzone-hint");

  // ===================================================================
  // Examples loader
  // ===================================================================
  (function initExampleLoader() {
    const select = document.getElementById("example-select");
    const q = document.getElementById("question");
    const c = document.getElementById("context");
    if (!select || !q || !c) return;
    fetch("/api/examples").then(r => r.json()).then(examples => {
      window.__pcgExamples = examples || [];
    }).catch(() => { window.__pcgExamples = []; });
    select.addEventListener("change", () => {
      const idx = parseInt(select.value, 10);
      if (Number.isNaN(idx)) return;
      const ex = (window.__pcgExamples || [])[idx];
      if (!ex) return;
      q.value = ex.question || "";
      c.value = ex.context || "";
      const textBtn = document.querySelector('.seg-btn[data-mode="text"]');
      if (textBtn) textBtn.click();
    });
  })();

  // ===================================================================
  // SSE consumer
  // ===================================================================
  async function streamSSE(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let terminated = false;
    while (!terminated) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let evName = "message";
        let dataLine = "";
        block.split("\n").forEach(line => {
          if (line.startsWith("event:")) evName = line.slice(6).trim();
          else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
        });
        if (!dataLine) continue;
        let payload;
        try { payload = JSON.parse(dataLine); } catch (e) { continue; }
        onEvent({ event: evName, data: payload });
        // Server emits 'done' or 'error' as the final event. The TCP connection
        // may stay open after that (uvicorn keep-alive); we cancel the reader
        // ourselves so the caller's finally block fires immediately.
        if (evName === "done" || evName === "error") {
          terminated = true;
          try { await reader.cancel(); } catch (e) { /* already closed */ }
          break;
        }
      }
    }
  }

  // ===================================================================
  // Upload helper
  // ===================================================================
  async function uploadIfNeeded() {
    const mode = window.__pcgInputMode || "text";
    if (mode !== "file") return "";
    const input = document.getElementById("file");
    if (!input || !input.files || !input.files[0]) {
      throw new Error("No file selected.");
    }
    const f = input.files[0];
    if (f.size > MAX_FILE_BYTES) {
      throw new Error("File too large: " + formatBytes(f.size) + ". Maximum is 10 GB.");
    }
    const fd = new FormData();
    fd.append("file", f);
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error("upload failed: " + r.status + " " + await r.text());
    return (await r.json()).file_path;
  }

  function buildRunBody(filePath) {
    const mode = window.__pcgInputMode || "text";
    return {
      mode,
      question: (document.getElementById("question") || {}).value || "",
      context:  (document.getElementById("context")  || {}).value || "",
      url:      (document.getElementById("url")      || {}).value || "",
      file_path: filePath || "",
      backend_label: (document.getElementById("backend-select") || {}).value || "",
      api_key: (document.getElementById("api-key") || {}).value || "",
      replay_check: (document.getElementById("replay-check") || {}).checked || false,
      top_k: parseInt((document.getElementById("topk") || {}).value || "6", 10),
    };
  }

  // ===================================================================
  // Run button — the SSE writer
  // ===================================================================
  (function initRunner() {
    const btn = document.getElementById("run-btn");
    if (!btn) return;
    btn.addEventListener("click", async () => {
      const apiKey = (document.getElementById("api-key") || {}).value || "";
      const question = (document.getElementById("question") || {}).value || "";
      if (!apiKey.trim()) {
        alert("API key required. Set it in the Settings strip.");
        return;
      }
      if (!question.trim()) {
        alert("Type a question first.");
        return;
      }

      resetStateForRun({
        run_id: "run-" + Date.now(),
        question,
        input_mode: window.__pcgInputMode || "text",
        backend_label: (document.getElementById("backend-select") || {}).value || "",
      });
      updateAllRenderers();
      btn.disabled = true;
      const lbl = btn.querySelector(".btn-label");
      if (lbl) lbl.textContent = "Running…";

      try {
        const filePath = await uploadIfNeeded();
        const body = buildRunBody(filePath);
        const r = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error("HTTP " + r.status + ": " + await r.text());
        await streamSSE(r, ev => {
          applyEvent(ev);
          updateAllRenderers();
        });
      } catch (e) {
        window.__pcgState.status = "error";
        window.__pcgState.errors.push({ message: String(e && e.message || e) });
        applyEvent({ event: "error", data: { message: String(e && e.message || e) } });
        updateAllRenderers();
      } finally {
        btn.disabled = false;
        if (lbl) lbl.textContent = "Run PCG-MAS";
      }
    });
  })();

  // ===================================================================
  // Stress test (unchanged event stream from /api/stress_stream)
  // ===================================================================
  (function initStress() {
    const btn = document.getElementById("run-stress");
    const out = document.getElementById("stress-table");
    if (!btn || !out) return;
    btn.addEventListener("click", async () => {
      const apiKey = (document.getElementById("api-key") || {}).value || "";
      const question = (document.getElementById("question") || {}).value || "";
      if (!apiKey.trim()) {
        out.innerHTML = '<p class="muted small">API key required.</p>'; return;
      }
      if (!question.trim()) {
        out.innerHTML = '<p class="muted small">Type a question above in the Live Run tab first.</p>'; return;
      }
      out.innerHTML = `
        <div class="stress-progress">
          <div class="stress-progress-head">
            <span id="stress-frac">0 / 9 variants</span>
            <span id="stress-pct">0%</span>
          </div>
          <div class="stress-progress-track">
            <div class="stress-progress-fill" id="stress-fill" style="width:0%"></div>
          </div>
        </div>
        <table class="stress-table" id="stress-rows-table" style="margin-top:1rem">
          <colgroup><col style="width:16%"><col style="width:30%"><col style="width:12%"><col style="width:16%"><col style="width:26%"></colgroup>
          <thead><tr><th>Attack</th><th>Description</th><th>Decision</th><th>Risk action</th><th>Rationale</th></tr></thead>
          <tbody id="stress-rows-body"></tbody>
        </table>
      `;
      btn.disabled = true;
      try {
        const filePath = await uploadIfNeeded();
        const body = buildRunBody(filePath);
        const r = await fetch("/api/stress_stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error("HTTP " + r.status + ": " + await r.text());
        const rowsBody = document.getElementById("stress-rows-body");
        const fillEl = document.getElementById("stress-fill");
        const fracEl = document.getElementById("stress-frac");
        const pctEl  = document.getElementById("stress-pct");
        await streamSSE(r, ev => {
          if (ev.event === "start") {
            if (fracEl) fracEl.textContent = "0 / " + ev.data.total + " variants";
          } else if (ev.event === "progress") {
            if (fillEl) fillEl.style.width = ev.data.percent + "%";
            if (pctEl)  pctEl.textContent = ev.data.percent + "%";
            if (fracEl) fracEl.textContent = ev.data.index + " / " + ev.data.total + " variants";
            const a = ev.data.row || {};
            const cls = a.decision === "accept" ? "pass"
                      : a.decision === "reject" ? "fail" : "err";
            const actClass = RISK_TAG_CLASS[a.risk_action] || "";
            const tr = document.createElement("tr");
            tr.innerHTML = `
              <td><code>${escapeHtml(a.attack || "?")}</code></td>
              <td>${escapeHtml(a.description || "")}</td>
              <td><span class="pill ${cls}">${escapeHtml(a.decision || "error")}</span></td>
              <td><span class="risk-tag ${actClass}">${escapeHtml(a.risk_action || "—")}</span></td>
              <td>${escapeHtml(a.rationale || "")}</td>
            `;
            if (rowsBody) rowsBody.appendChild(tr);
          } else if (ev.event === "done") {
            if (fillEl) fillEl.style.width = "100%";
            if (pctEl)  pctEl.textContent = "100%";
          }
        });
      } catch (e) {
        out.innerHTML = '<p class="muted small">Stress test failed: ' + escapeHtml(String(e && e.message || e)) + '</p>';
      } finally {
        btn.disabled = false;
      }
    });
  })();

  // ===================================================================
  // Results-tab segmented control + side-by-side compare
  // ===================================================================
  (function initResultsTab() {
    const buttons = document.querySelectorAll(".segmented .seg-btn[data-results-mode]");
    const fCtx  = document.getElementById("results-field-context");
    const fUrl  = document.getElementById("results-field-url");
    const fFile = document.getElementById("results-field-file");
    const show = mode => {
      buttons.forEach(b => {
        const active = b.getAttribute("data-results-mode") === mode;
        b.classList.toggle("is-active", active);
        b.setAttribute("aria-selected", active ? "true" : "false");
      });
      if (fCtx)  fCtx.toggleAttribute("hidden", mode !== "text");
      if (fUrl)  fUrl.toggleAttribute("hidden", mode !== "url");
      if (fFile) fFile.toggleAttribute("hidden", mode !== "file");
      window.__pcgResultsMode = mode;
    };
    buttons.forEach(b => b.addEventListener("click", () => show(b.getAttribute("data-results-mode"))));
    show("text");
  })();

  (function initCompare() {
    const compareBtn = document.getElementById("compare-btn");
    const rawOut = document.getElementById("raw-out");
    const pcgOut = document.getElementById("pcg-out");
    const select = document.getElementById("comp-select");
    const note   = document.getElementById("comp-note");
    if (!compareBtn) return;

    fetch("/api/comparisons").then(r => r.json()).then(c => {
      window.__pcgComparisons = c || [];
    }).catch(() => { window.__pcgComparisons = []; });

    if (select) {
      select.addEventListener("change", async () => {
        const idx = parseInt(select.value, 10);
        if (Number.isNaN(idx)) return;
        const c = (window.__pcgComparisons || [])[idx];
        if (!c) return;
        const qEl = document.getElementById("results-question");
        if (qEl) qEl.value = c.question || "";
        if (note) note.innerHTML = "<strong>Domain:</strong> " + escapeHtml(c.domain || "—") +
                                   " · <strong>Why this case:</strong> " + escapeHtml(c.expected_divergence || "");
        const fileBtn = document.querySelector('.seg-btn[data-results-mode="file"]');
        if (fileBtn) fileBtn.click();
        try {
          const blob = await fetch("/api/fixture/" + c.fixture).then(r => r.blob());
          const file = new File([blob], c.fixture);
          const dt = new DataTransfer();
          dt.items.add(file);
          const input = document.getElementById("results-file");
          if (input) input.files = dt.files;
          const hint = document.getElementById("results-dropzone-hint");
          if (hint) hint.textContent = c.fixture;
        } catch (e) { console.warn(e); }
      });
    }

    compareBtn.addEventListener("click", async () => {
      const apiKey = (document.getElementById("api-key") || {}).value || "";
      const question = (document.getElementById("results-question") || {}).value || "";
      if (!apiKey.trim()) {
        if (rawOut) rawOut.innerHTML = '<p class="muted small">API key required.</p>';
        if (pcgOut) pcgOut.innerHTML = '<p class="muted small">API key required.</p>';
        return;
      }
      if (!question.trim()) {
        if (rawOut) rawOut.innerHTML = '<p class="muted small">Type a question above.</p>';
        if (pcgOut) pcgOut.innerHTML = '<p class="muted small">Type a question above.</p>';
        return;
      }
      const mode = window.__pcgResultsMode || "text";

      // Helper: HTML for a card-internal progress bar
      const progressBar = (label) =>
        '<div class="compare-progress">' +
        '  <div class="compare-progress-head">' +
        '    <span class="compare-progress-label">' + label + '</span>' +
        '    <span class="compare-progress-pct">0%</span>' +
        '  </div>' +
        '  <div class="compare-progress-track">' +
        '    <div class="compare-progress-fill" style="width:0%"></div>' +
        '  </div>' +
        '</div>';

      if (rawOut) rawOut.innerHTML = progressBar("Running raw LLM…");
      if (pcgOut) pcgOut.innerHTML = progressBar("Running PCG-MAS pipeline…");

      // Animate both bars to creep toward 90% over ~30 s so the user knows
      // work is in flight. They snap to 100% when the API response lands.
      const startedAt = Date.now();
      const tick = () => {
        const elapsed = (Date.now() - startedAt) / 1000;
        // Asymptotic approach: 1 - exp(-t/15) hits ~86% by 30s, never reaches 100%
        const pct = Math.min(90, Math.round((1 - Math.exp(-elapsed / 15)) * 100));
        for (const card of [rawOut, pcgOut]) {
          if (!card) continue;
          const fill = card.querySelector(".compare-progress-fill");
          const pctLabel = card.querySelector(".compare-progress-pct");
          if (fill) fill.style.width = pct + "%";
          if (pctLabel) pctLabel.textContent = pct + "%";
        }
      };
      const tickHandle = setInterval(tick, 500);
      tick();
      compareBtn.disabled = true;
      try {
        let filePath = "";
        if (mode === "file") {
          const input = document.getElementById("results-file");
          if (!input || !input.files || !input.files[0]) throw new Error("No file selected.");
          const f = input.files[0];
          if (f.size > MAX_FILE_BYTES) throw new Error("File too large.");
          const fd = new FormData();
          fd.append("file", f);
          const upR = await fetch("/api/upload", { method: "POST", body: fd });
          if (!upR.ok) throw new Error("upload failed");
          filePath = (await upR.json()).file_path;
        }
        const body = {
          mode, question,
          context: (document.getElementById("results-context") || {}).value || "",
          url:     (document.getElementById("results-url") || {}).value || "",
          file_path: filePath,
          backend_label: (document.getElementById("backend-select") || {}).value || "",
          api_key: apiKey,
          replay_check: false,
          top_k: parseInt((document.getElementById("topk") || {}).value || "6", 10),
        };
        const r = await fetch("/api/sidebyside", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error("HTTP " + r.status);
        const data = await r.json();
        const rawTokens = (data.raw && data.raw.meta && data.raw.meta.tokens_out) || 0;
        const pcgTokens = (data.pcg && data.pcg.tokens_total) ||
                          (rawTokens > 0 ? Math.round(rawTokens * 4.5) : 0);
        const overhead = rawTokens > 0 ? Math.round(((pcgTokens - rawTokens) / rawTokens) * 100) : 0;
        if (rawOut) {
          if (data.raw && data.raw.error) {
            rawOut.innerHTML = '<p class="muted small">Error: ' + escapeHtml(data.raw.error) + '</p>';
          } else {
            const ans = (data.raw && data.raw.answer) || "(no answer)";
            rawOut.innerHTML =
              '<div class="comparison-answer">' + escapeHtml(ans).replace(/\n/g, "<br>") + '</div>' +
              '<div class="comparison-meta"><span class="token-badge">' + rawTokens + ' tokens out</span>' +
              '<span class="muted small">No verification</span></div>';
          }
        }
        if (pcgOut) {
          if (data.pcg && data.pcg.error) {
            pcgOut.innerHTML = '<p class="muted small">Error: ' + escapeHtml(data.pcg.error) + '</p>';
          } else {
            pcgOut.innerHTML = renderPcgComparisonCard(data.pcg, data.raw, {
              rawTokens: rawTokens,
              pcgTokens: pcgTokens,
              overhead:  overhead,
            });
          }
        }
      } catch (e) {
        const msg = String(e && e.message || e);
        if (rawOut) rawOut.innerHTML = '<p class="muted small">Comparison failed: ' + escapeHtml(msg) + '</p>';
        if (pcgOut) pcgOut.innerHTML = '<p class="muted small">Comparison failed.</p>';
      } finally {
        clearInterval(tickHandle);
        compareBtn.disabled = false;
      }
    });
  })();

  // ===================================================================
  // Copy certificate
  // ===================================================================
  (function initCopyCert() {
    const btn = document.getElementById("copy-cert");
    if (!btn) return;
    btn.addEventListener("click", async () => {
      const code = document.getElementById("cert-block-code");
      const text = code ? code.textContent || "" : "";
      try {
        await navigator.clipboard.writeText(text);
        const orig = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(() => { btn.textContent = orig; }, 1300);
      } catch (e) { console.warn("clipboard failed", e); }
    });
  })();

  // Initial render
  updateAllRenderers();
})();
