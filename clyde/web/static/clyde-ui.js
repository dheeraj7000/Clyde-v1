/* Clyde UI primitives — SVG dataviz and design-canvas screen patterns.
   Exposes window.ClydeUI:
     FanChart, DivergenceHeatmap, MiniSpark, ForceGraph, mascot helpers.
   Each chart returns an SVG string so it can be injected via x-html or
   innerHTML — no React dependency. Colors come from the dark-theme tokens
   defined in styles.css / tokens.jsx (#0A0B0D canvas, #8481FF accent). */

(function () {
  const T = {
    bg: "#0A0B0D",
    surface: "#111215",
    surface2: "#16181C",
    border: "rgba(255,255,255,0.06)",
    borderStrong: "rgba(255,255,255,0.10)",
    fg: "#F2F3F5",
    fg2: "#B8BCC4",
    fg3: "#72767E",
    fg4: "#464A52",
    accent: "#8481FF",
    accentDim: "#6663D9",
    accentGlow: "rgba(132,129,255,0.18)",
    central: "#E8E9ED",
    band50: "rgba(132,129,255,0.28)",
    band80: "rgba(132,129,255,0.16)",
    band95: "rgba(132,129,255,0.08)",
    tail: "rgba(255,255,255,0.10)",
    positive: "#4ADE80",
    negative: "#F87171",
    warn: "#FBBF24",
    info: "#60A5FA",
    household: "#4ADE80",
    firm: "#60A5FA",
    bank: "#FBBF24",
    centralbank: "#C084FC",
    shocked: "#F87171",
  };

  // ─── FanChart ────────────────────────────────────────────────────
  // series = { steps: [step indices], central: [...],
  //            band50: [[lo,hi]..], band80, band95: [[lo,hi]..],
  //            tails: [[y0..]..],  // optional spaghetti
  //            branches: [{label, data, color}]  // overlay lines
  //          }
  // Bands fall back to optimistic/pessimistic if 50/80/95 not provided.
  function fanChart({
    series,
    w = 720,
    h = 320,
    yLabel = "value",
    xLabel = "step",
    scrubStep,
    showTails = true,
    showBands = true,
    branches = [],
    title,
  } = {}) {
    if (!series || !Array.isArray(series.central) || !series.central.length) {
      return emptyChart(w, h, "no data");
    }
    const s = series;
    const N = s.central.length;
    const pad = { l: 48, r: 14, t: 12, b: 28 };
    const innerW = w - pad.l - pad.r;
    const innerH = h - pad.t - pad.b;

    // y range over outermost bands + tails + branches
    let ymin = Infinity, ymax = -Infinity;
    const ext = (v) => { if (v == null || isNaN(v)) return; if (v < ymin) ymin = v; if (v > ymax) ymax = v; };
    s.central.forEach(ext);
    if (s.band95) s.band95.forEach((p) => { ext(p[0]); ext(p[1]); });
    if (s.band80) s.band80.forEach((p) => { ext(p[0]); ext(p[1]); });
    if (s.tails) s.tails.forEach((arr) => arr.forEach(ext));
    branches.forEach((b) => (b.data || []).forEach(ext));
    if (!isFinite(ymin) || !isFinite(ymax)) { ymin = 0; ymax = 1; }
    if (ymin === ymax) { ymin -= 0.5; ymax += 0.5; }
    const ypad = (ymax - ymin) * 0.08;
    ymin -= ypad; ymax += ypad;

    const xAt = (i) => pad.l + (i / (N - 1 || 1)) * innerW;
    const yAt = (v) => pad.t + (1 - (v - ymin) / (ymax - ymin)) * innerH;

    const pathLine = (arr) => arr
      .map((v, i) => v == null ? null : `${i === 0 ? "M" : "L"}${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`)
      .filter(Boolean).join(" ");
    const pathBand = (arr) => {
      const top = arr.map((p, i) => `${i === 0 ? "M" : "L"}${xAt(i).toFixed(1)},${yAt(p[1]).toFixed(1)}`).join(" ");
      const bot = arr.slice().reverse().map((p, i) => `L${xAt(N - 1 - i).toFixed(1)},${yAt(p[0]).toFixed(1)}`).join(" ");
      return top + " " + bot + " Z";
    };

    const yTicks = 5;
    const gridYs = Array.from({ length: yTicks }, (_, i) => ymin + (ymax - ymin) * (i / (yTicks - 1)));
    const xTicks = 6;
    const xTickIs = Array.from({ length: xTicks }, (_, i) => Math.round((N - 1) * i / (xTicks - 1)));
    const tickLabel = (i) => (s.steps && s.steps[i] != null) ? s.steps[i] : i;

    const parts = [];
    parts.push(`<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="display:block;overflow:visible">`);
    // gridlines + y-tick labels
    gridYs.forEach((v) => {
      parts.push(`<line x1="${pad.l}" x2="${w - pad.r}" y1="${yAt(v).toFixed(1)}" y2="${yAt(v).toFixed(1)}" stroke="${T.border}" stroke-dasharray="2,3"/>`);
      parts.push(`<text x="${pad.l - 8}" y="${(yAt(v) + 3).toFixed(1)}" text-anchor="end" font-family="JetBrains Mono,monospace" font-size="9" fill="${T.fg3}">${fmt(v)}</text>`);
    });
    // y-axis label
    parts.push(`<text x="10" y="${pad.t + innerH / 2}" font-family="JetBrains Mono,monospace" font-size="9" fill="${T.fg3}" transform="rotate(-90 14 ${pad.t + innerH / 2})">${escape(yLabel)}</text>`);
    // bands
    if (showBands) {
      if (s.band95) parts.push(`<path d="${pathBand(s.band95)}" fill="${T.band95}"/>`);
      if (s.band80) parts.push(`<path d="${pathBand(s.band80)}" fill="${T.band80}"/>`);
      if (s.band50) parts.push(`<path d="${pathBand(s.band50)}" fill="${T.band50}"/>`);
    }
    // tail spaghetti
    if (showTails && s.tails) {
      s.tails.forEach((arr) => parts.push(`<path d="${pathLine(arr)}" fill="none" stroke="${T.tail}" stroke-width="0.7"/>`));
    }
    // central
    parts.push(`<path d="${pathLine(s.central)}" fill="none" stroke="${T.central}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="1400" stroke-dashoffset="1400" style="animation: cl-draw 900ms ease-out 100ms forwards"/>`);
    // branches overlay (dashed)
    branches.forEach((b) => {
      if (!b.data || !b.data.length) return;
      parts.push(`<path d="${pathLine(b.data)}" fill="none" stroke="${b.color || T.accent}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="6,4" opacity="0.95"/>`);
    });
    // scrubber
    if (scrubStep != null && scrubStep >= 0 && scrubStep < N) {
      parts.push(`<line x1="${xAt(scrubStep)}" x2="${xAt(scrubStep)}" y1="${pad.t}" y2="${pad.t + innerH}" stroke="${T.accent}" stroke-width="1" stroke-dasharray="3,2" opacity="0.7"/>`);
      parts.push(`<circle cx="${xAt(scrubStep)}" cy="${yAt(s.central[scrubStep])}" r="3.5" fill="${T.accent}" stroke="${T.bg}" stroke-width="1.5"/>`);
    }
    // x-tick labels
    xTickIs.forEach((i) => {
      parts.push(`<text x="${xAt(i)}" y="${h - pad.b + 14}" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="9" fill="${T.fg3}">${tickLabel(i)}</text>`);
    });
    parts.push(`<text x="${w / 2}" y="${h - 2}" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="9" fill="${T.fg3}">${escape(xLabel)}</text>`);
    if (title) parts.push(`<text x="${pad.l}" y="${pad.t - 2}" font-family="Inter,sans-serif" font-size="11" font-weight="500" fill="${T.fg2}">${escape(title)}</text>`);
    parts.push(`</svg>`);
    return parts.join("");
  }

  // ─── Convert backend optimistic/pessimistic shape into design's bands ───
  // Backend gives optimistic + pessimistic + tail_upper + tail_lower as
  // four arrays per metric. Design expects band50/band80/band95 as
  // [[lo,hi]...]. Map: opt/pess → band50, tail_upper/lower → band95,
  // synthesize band80 by interpolation.
  function bandifyFromQuantiles({ central, optimistic, pessimistic, tail_upper, tail_lower }) {
    const N = central.length;
    const band50 = [], band80 = [], band95 = [];
    for (let i = 0; i < N; i++) {
      const c = central[i];
      const o = optimistic ? optimistic[i] : null;
      const p = pessimistic ? pessimistic[i] : null;
      const tu = tail_upper ? tail_upper[i] : null;
      const tl = tail_lower ? tail_lower[i] : null;
      const lo50 = p != null ? p : (c - 0.02);
      const hi50 = o != null ? o : (c + 0.02);
      const lo95 = tl != null ? tl : (lo50 - 0.04);
      const hi95 = tu != null ? tu : (hi50 + 0.04);
      const lo80 = (lo50 + lo95) / 2;
      const hi80 = (hi50 + hi95) / 2;
      band50.push([lo50, hi50]);
      band80.push([lo80, hi80]);
      band95.push([lo95, hi95]);
    }
    return { band50, band80, band95 };
  }

  // ─── DivergenceHeatmap ─────────────────────────────────────────
  // Rows = variables, value 0..1 contribution. If `series` per row is
  // available it fills in temporal contribution; otherwise renders a
  // single-column intensity bar.
  function divergenceHeatmap({ vars, w = 360, h = 220 } = {}) {
    if (!vars || !vars.length) return emptyChart(w, h, "no divergence drivers");
    const padL = 168, padR = 8, padT = 8, padB = 18;
    const rows = vars.length;
    // Detect column shape: each var may carry .series (length=cols) or .sensitivity
    const cols = Math.max(...vars.map((v) => Array.isArray(v.series) ? v.series.length : 1));
    const cw = (w - padL - padR) / Math.max(1, cols);
    const rh = (h - padT - padB) / rows;
    const colorAt = (a) => {
      const r1 = 132 + (251 - 132) * a;
      const g1 = 129 + (191 - 129) * a;
      const b1 = 255 + (36 - 255) * a;
      return `rgba(${r1.toFixed(0)},${g1.toFixed(0)},${b1.toFixed(0)},${(0.15 + a * 0.65).toFixed(3)})`;
    };
    // When there's a per-step series the rank reads vertically so each row
    // is normalized within itself. For the single-column case the bars only
    // make sense when normalized against the *global* max (top driver = full
    // saturation, weakest = pale).
    const hasSeries = vars.some((v) => Array.isArray(v.series) && v.series.length);
    const globalMax = Math.max(0.0001, ...vars.map((v) => Math.abs(v.sensitivity ?? 0)));
    const parts = [`<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="display:block">`];
    vars.forEach((v, ri) => {
      const y = padT + ri * rh + rh / 2 + 3;
      parts.push(`<text x="${padL - 8}" y="${y}" text-anchor="end" font-family="JetBrains Mono,monospace" font-size="10.5" fill="${T.fg2}">${escape(truncate(v.name || v.variable || "?", 22))}</text>`);
      const row = hasSeries && Array.isArray(v.series) && v.series.length ? v.series : [v.sensitivity ?? 0];
      const norm = hasSeries
        ? Math.max(0.0001, ...row.map((x) => Math.abs(x)))
        : globalMax;
      row.forEach((val, ci) => {
        const a = Math.max(0, Math.min(1, Math.abs(val) / norm));
        parts.push(`<rect x="${(padL + ci * cw + 0.5).toFixed(1)}" y="${(padT + ri * rh + 0.5).toFixed(1)}" width="${(cw - 1).toFixed(1)}" height="${(rh - 1).toFixed(1)}" fill="${colorAt(a)}" rx="1.5"/>`);
      });
    });
    if (cols > 1) {
      [0, 1, 2, 3].forEach((i) => {
        const step = Math.round((cols - 1) * i / 3);
        parts.push(`<text x="${(padL + (step + 0.5) * cw).toFixed(1)}" y="${h - 4}" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="9" fill="${T.fg3}">${step}</text>`);
      });
    }
    parts.push("</svg>");
    return parts.join("");
  }

  // ─── MiniSpark ─────────────────────────────────────────────────
  function miniSpark({ data, w = 80, h = 22, color } = {}) {
    if (!data || !data.length) return `<svg width="${w}" height="${h}"></svg>`;
    const min = Math.min(...data), max = Math.max(...data);
    const xAt = (i) => (i / (data.length - 1 || 1)) * w;
    const yAt = (v) => (1 - (v - min) / (max - min || 1)) * (h - 4) + 2;
    const d = data.map((v, i) => `${i === 0 ? "M" : "L"}${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`).join(" ");
    return `<svg width="${w}" height="${h}" style="display:block"><path d="${d}" fill="none" stroke="${color || T.fg2}" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  }

  // ─── ForceGraph (SVG, layout pre-computed by caller or by us) ───
  // nodes: [{id, type, label?, x?, y?, shocked?, params?}]
  // edges: [{source, target, kind?}]
  // If x/y missing, compute a deterministic radial layout grouped by type.
  function forceGraph({ nodes, edges, w = 640, h = 460, pinnedId, compact = false } = {}) {
    if (!nodes || !nodes.length) return emptyChart(w, h, "no graph");
    const colorByType = {
      household: T.household, firm: T.firm, bank: T.bank, central_bank: T.centralbank, centralbank: T.centralbank,
    };
    const radiusByType = { household: 4.5, firm: 5.5, bank: 7, central_bank: 10, centralbank: 10 };
    // Layout: place central_bank at center, banks at inner ring, firms at mid, households outer.
    const groups = { central_bank: [], bank: [], firm: [], household: [], other: [] };
    nodes.forEach((n) => { (groups[n.type] || groups.other).push(n); });
    const cx = w / 2, cy = h / 2;
    const placed = new Map();
    const place = (arr, rad) => {
      arr.forEach((n, i) => {
        if (typeof n.x === "number" && typeof n.y === "number") {
          placed.set(n.id, { ...n, x: n.x, y: n.y, _r: radiusByType[n.type] || 5 });
          return;
        }
        const a = (i / (arr.length || 1)) * Math.PI * 2 + (n.type === "bank" ? 0.3 : n.type === "firm" ? 0.7 : 0);
        placed.set(n.id, { ...n, x: cx + Math.cos(a) * rad, y: cy + Math.sin(a) * (rad * 0.7), _r: radiusByType[n.type] || 5 });
      });
    };
    place(groups.central_bank, 0);
    place(groups.bank, 90);
    place(groups.firm, 170);
    place(groups.household, 230);
    place(groups.other, 200);

    const related = new Set();
    if (pinnedId) {
      related.add(pinnedId);
      edges.forEach((e) => {
        if (e.source === pinnedId) related.add(e.target);
        if (e.target === pinnedId) related.add(e.source);
      });
    }

    const parts = [`<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet" style="display:block">`];
    edges.forEach((e) => {
      const a = placed.get(e.source), b = placed.get(e.target);
      if (!a || !b) return;
      const dim = pinnedId && !(related.has(e.source) && related.has(e.target));
      const stroke = e.kind === "policy" ? T.accent : e.kind === "credit" ? T.borderStrong : e.kind === "interbank" ? T.bank : T.border;
      const sw = e.kind === "policy" ? 1.2 : 0.7;
      const op = dim ? 0.1 : (e.kind === "policy" ? 0.45 : 0.35);
      const dash = e.kind === "interbank" ? `stroke-dasharray="2,3"` : "";
      parts.push(`<line x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${b.x.toFixed(1)}" y2="${b.y.toFixed(1)}" stroke="${stroke}" stroke-width="${sw}" opacity="${op}" ${dash}/>`);
    });
    placed.forEach((n) => {
      const dim = pinnedId && !related.has(n.id);
      const isPinned = n.id === pinnedId;
      const color = n.shocked ? T.shocked : (colorByType[n.type] || T.fg3);
      parts.push(`<g opacity="${dim ? 0.22 : 1}" data-node-id="${escape(n.id)}" style="cursor:pointer;transition:opacity 200ms">`);
      if (n.shocked) parts.push(`<circle cx="${n.x.toFixed(1)}" cy="${n.y.toFixed(1)}" r="${n._r + 5}" fill="none" stroke="${T.shocked}" stroke-width="1" opacity="0.4" style="animation: cl-pulse 1.4s infinite"/>`);
      if (isPinned) parts.push(`<circle cx="${n.x.toFixed(1)}" cy="${n.y.toFixed(1)}" r="${n._r + 4}" fill="none" stroke="${color}" stroke-width="1" opacity="0.6"/>`);
      parts.push(`<circle cx="${n.x.toFixed(1)}" cy="${n.y.toFixed(1)}" r="${n._r}" fill="${color}"/>`);
      if (!compact && (n.type === "central_bank" || n.type === "centralbank" || isPinned || n.shocked)) {
        parts.push(`<text x="${n.x.toFixed(1)}" y="${(n.y + n._r + 12).toFixed(1)}" text-anchor="middle" font-size="9" fill="${T.fg2}" font-family="JetBrains Mono,monospace">${escape(truncate(n.label || n.id, 14))}</text>`);
      }
      parts.push(`</g>`);
    });
    parts.push("</svg>");
    return parts.join("");
  }

  // ─── Step rail (pipeline progress, design's right rail pattern) ───
  function stepRail({ steps }) {
    // steps: [{name, status, dur, note}] where status ∈ done|running|pending
    const parts = [];
    steps.forEach((s, i) => {
      const isRunning = s.status === "running";
      const isDone = s.status === "done";
      const isPending = !isRunning && !isDone;
      const markerBg = isDone || isRunning ? "rgba(132,129,255,0.18)" : T.surface2;
      const markerBd = isDone || isRunning ? T.accent : T.border;
      const markerFg = isDone || isRunning ? T.accent : T.fg3;
      const inner = isDone
        ? `<svg width="11" height="11" viewBox="0 0 12 12"><path d="M2.5 6l2.5 2.5L9.5 3.5" stroke="currentColor" stroke-width="1.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>`
        : isRunning
          ? `<svg width="11" height="11" viewBox="0 0 12 12" style="animation:cl-pulse 1.2s linear infinite"><circle cx="6" cy="6" r="4" stroke="currentColor" stroke-width="1.4" fill="none" opacity="0.3"/><path d="M10 6a4 4 0 00-4-4" stroke="currentColor" stroke-width="1.4" fill="none" stroke-linecap="round"/></svg>`
          : `<span style="font-size:10px;font-family:JetBrains Mono,monospace">${String(i + 1).padStart(2, "0")}</span>`;
      const railLine = i < steps.length - 1
        ? `<div style="position:absolute;left:23px;top:30px;width:1px;height:calc(100% - 16px);background:${T.border}"></div>`
        : "";
      parts.push(`
        <div style="display:grid;grid-template-columns:48px 1fr auto;padding:12px 18px;position:relative;opacity:${isPending ? 0.5 : 1}">
          ${railLine}
          <div style="display:flex;align-items:center;justify-content:center">
            <div style="width:22px;height:22px;border-radius:11px;background:${markerBg};border:1px solid ${markerBd};display:flex;align-items:center;justify-content:center;color:${markerFg}">${inner}</div>
          </div>
          <div>
            <div style="display:flex;align-items:center;gap:10px">
              <span style="font-size:13px;font-weight:500;color:${T.fg}">${escape(s.name)}</span>
              ${isRunning ? `<span class="cl-mono" style="display:inline-flex;align-items:center;height:20px;padding:0 8px;font-size:10.5px;color:${T.accent};background:${T.accentGlow};border:1px solid rgba(132,129,255,0.3);border-radius:4px">running</span>` : ""}
            </div>
            ${s.note ? `<div style="font-size:11.5px;color:${T.fg3};margin-top:3px;line-height:1.45">${escape(s.note)}</div>` : ""}
          </div>
          <div class="cl-mono" style="font-size:10.5px;color:${T.fg4}">${escape(s.dur || "")}</div>
        </div>`);
    });
    return parts.join("");
  }

  // ─── helpers ────────────────────────────────────────────────────
  function emptyChart(w, h, msg) {
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="display:block"><rect x="0" y="0" width="${w}" height="${h}" fill="transparent"/><text x="${w / 2}" y="${h / 2}" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="11" fill="${T.fg4}">${escape(msg)}</text></svg>`;
  }
  function escape(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  function truncate(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }
  function fmt(v) {
    const a = Math.abs(v);
    if (a >= 1000) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }

  window.ClydeUI = {
    T, fanChart, divergenceHeatmap, miniSpark, forceGraph, stepRail, bandifyFromQuantiles,
  };
})();
