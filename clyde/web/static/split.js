/* Live View — Mirofish-style split screen.
   Left:  knowledge graph that populates progressively, edges carry labels
          and pulse when the corresponding causal event fires.
   Right: numbered stage cards (parse → spawn actors → simulate → synthesize
          → report) plus a chat-style conversation between the AI actors,
          plus a SYSTEM DASHBOARD terminal at the bottom.
   The Alpine component owns: graph state (nodes/edges/labels), feed buckets
   (chat / system / events), stage state, and a mock event stream that paces
   activity when the backend isn't streaming yet.
*/

// Design tokens — keep in sync with clyde-ui.js / styles.css.
const T_LIVE = {
  household: "#4ADE80",
  firm: "#60A5FA",
  bank: "#FBBF24",
  central_bank: "#C084FC",
  shocked: "#F87171",
  accent: "#8481FF",
  fg: "#F2F3F5", fg2: "#B8BCC4", fg3: "#72767E", fg4: "#464A52",
  bg: "#0A0B0D", surface: "#111215", surface2: "#16181C",
  border: "rgba(255,255,255,0.06)", borderStrong: "rgba(255,255,255,0.10)",
};
const SPLIT_NODE_COLORS = {
  household: T_LIVE.household, firm: T_LIVE.firm, bank: T_LIVE.bank,
  central_bank: T_LIVE.central_bank, centralbank: T_LIVE.central_bank,
};
const SPLIT_NODE_R = { household: 4, firm: 6, bank: 8, central_bank: 11, centralbank: 11 };
// Edge stylings keyed by transmission channel — the design's edge-label style
// (small mono caps along the line). Color matches the originating sector.
const SPLIT_EDGE_STYLE = {
  monetary_policy: { color: T_LIVE.central_bank, label: "POLICY_RATE" },
  interbank: { color: T_LIVE.bank, label: "INTERBANK_LIQUIDITY", dash: "3,2" },
  lending: { color: T_LIVE.bank, label: "CREDIT_SUPPLY" },
  credit: { color: T_LIVE.bank, label: "CREDIT_SUPPLY" },
  supply: { color: T_LIVE.firm, label: "SUPPLY_CHAIN" },
  employment: { color: T_LIVE.firm, label: "WAGES" },
  trade: { color: T_LIVE.firm, label: "PRICES" },
  consumption: { color: T_LIVE.household, label: "DEMAND" },
};
const edgeStyle = (e) => SPLIT_EDGE_STYLE[e.type] || {
  color: T_LIVE.borderStrong, label: (e.kind || e.type || "RELATES_TO").toUpperCase(),
};

window.splitScreen = function () {
  return {
    /* ===== state ===== */
    active: false,
    mode: "pipeline",      // "pipeline" | "agent"
    jobId: null,
    simId: null,

    nodeColors: SPLIT_NODE_COLORS,

    // Stage rail — five stages mirroring Mirofish's numbered cards.
    stages: [
      { key: "parse", label: "Parse Scenario", endpoint: "POST /api/runs", note: "LLM extracts shock type, severity, scope, time horizon", status: "pending" },
      { key: "graph", label: "Build Knowledge Graph", endpoint: "POST /api/graph/build", note: "Spawns actors, wires transmission channels, imports prior library", status: "pending" },
      { key: "sim", label: "Run Simulation", endpoint: "POST /api/simulation", note: "Monte-Carlo ensemble — actors step round by round", status: "pending" },
      { key: "synth", label: "Synthesize", endpoint: "POST /api/synthesize", note: "Quantile bands, divergence drivers, causal chains", status: "pending" },
      { key: "report", label: "Generate Report", endpoint: "POST /api/report", note: "Narrative grounded in simulation artifacts", status: "pending" },
    ],
    stageCounters: { nodes: 0, edges: 0, schemas: 0, runs: 0, horizon: 0 },

    // Inject input flow (kept from previous split view).
    injectText: "",
    injecting: false,

    // KG state
    kgNodes: [],
    kgEdges: [],
    kgSim: null,
    kgSelectedNode: null,
    showEdgeLabels: true,

    // Feed: split into two streams to match Mirofish's chat / dashboard split.
    chat: [],        // {id, ts, actor, role, color, body, action?, evidence?}
    sysLog: [],      // {id, ts, level, source, body}
    feedFilter: "all", // legacy: "all"|"decisions"|"events"|"metrics"|"system"
    feed: [],        // legacy aggregated for split-panel right pane
    activeRightTab: "conversation", // "conversation" | "stages"

    // Agent sim state
    agentRound: 0,
    agentTotal: 0,
    agentPlaying: false,
    agentSpeed: 2000,
    _agentTimer: null,
    agentStatus: "idle",
    roundMetrics: null,

    // Hackathon demo state — drives the scripted Ontology → Setup →
    // Simulation flow. demoStage gates which pane is shown and which CTA
    // appears at the bottom of the stages panel.
    demoStage: null,        // null | 'ontology' | 'awaitSetup' | 'envSetup' | 'awaitConfig' | 'configReady' | 'simulating' | 'done'
    demoActors: [],         // hardcoded actor profile cards
    _demoTimers: [],

    /* ===== lifecycle ===== */

    show(jobId, result) {
      this.active = true;
      this.mode = "pipeline";
      this.jobId = jobId;
      this.chat = [];
      this.sysLog = [];
      this.feed = [];
      this.kgNodes = [];
      this.kgEdges = [];
      this.kgSelectedNode = null;
      this.stages = this.stages.map((s) => ({ ...s, status: "pending" }));
      this.stageCounters = { nodes: 0, edges: 0, schemas: 0, runs: 0, horizon: 0 };
      this._sysLog("info", "console", "Live View opened — streaming pipeline events");

      if (result?.network) {
        this._populateKG(result.network);
      }
      this._feedFromResult(result);
      this.$nextTick(() => this._renderKG());
    },

    // Open immediately on Run, before the result is known. Stages stream
    // updates as the parent Alpine app polls the backend.
    showLive(jobId, description) {
      this.active = true;
      this.mode = "pipeline";
      this.jobId = jobId;
      this.chat = [];
      this.sysLog = [];
      this.feed = [];
      this.kgNodes = [];
      this.kgEdges = [];
      this.kgSelectedNode = null;
      this.stages = this.stages.map((s) => ({ ...s, status: "pending" }));
      this.stageCounters = { nodes: 0, edges: 0, schemas: 0, runs: 0, horizon: 0 };
      this.activeRightTab = "conversation";
      this.stages[0].status = "running";
      this._sysLog("info", "console", "Run started — opening Live View");
      if (description) this._sysLog("info", "input", `scenario: ${description.slice(0, 140)}${description.length > 140 ? "…" : ""}`);
      this.$nextTick(() => this._renderKG());
    },

    // Called by the parent app on each poll to advance stages and stream
    // chat lines for whatever progress milestones the backend has hit.
    pumpProgress(job) {
      if (!job) return;
      const stage = (job.progress?.stage || "").toLowerCase();
      const pct = Math.round(job.progress?.percent || 0);
      const stageKeys = ["parse", "graph", "sim", "synth", "report"];
      const i = stageKeys.findIndex((k) => stage.includes(k));
      if (i >= 0) {
        for (let k = 0; k < stageKeys.length; k++) {
          if (k < i) this.stages[k].status = "done";
          else if (k === i) this.stages[k].status = job.status === "completed" ? "done" : "running";
          else this.stages[k].status = "pending";
        }
      }
      if (job.status === "completed") {
        this.stages.forEach((s) => (s.status = "done"));
      }
      this.stageCounters.runs = job.run_count || this.stageCounters.runs;
    },

    // Called when the parent app receives the final result.
    onResult(result) {
      if (!result) return;
      this.stages.forEach((s) => (s.status = "done"));
      if (result.network) {
        this._populateKG(result.network);
        this.stageCounters.nodes = (result.network.nodes || []).length;
        this.stageCounters.edges = (result.network.edges || []).length;
        this.stageCounters.schemas = new Set((result.network.edges || []).map((e) => e.kind || e.type)).size;
        this.$nextTick(() => this._renderKG());
      }
      // Synthesize a chat thread from result.causal_chains so users see
      // actor "decisions" even if the backend doesn't expose live agents.
      this._chatFromResult(result);
      this._sysLog("ok", "pipeline", "Run complete — open the Report panel to read the synthesis.");
    },

    /* ===== agent sim (real backend) ===== */
    async startAgentMode() {
      if (!this.jobId) return;
      this.mode = "agent";
      this.agentStatus = "starting";
      this._sysLog("info", "agent", "Initializing autonomous LLM agents…");
      try {
        const r = await fetch(`/api/runs/${this.jobId}/agent-sim/start`, { method: "POST" });
        if (!r.ok) throw new Error(await r.text());
        const data = await r.json();
        this.simId = data.sim_id;
        this.agentTotal = data.total_rounds;
        this.agentRound = 0;
        this.agentStatus = "ready";
        this._sysLog("ok", "agent", `Simulation initialized · ${data.actors} agents · ${data.total_rounds} rounds`);
      } catch (e) {
        this.agentStatus = "error";
        this._sysLog("err", "agent", "Init failed — " + e.message);
      }
    },

    async runOneRound() {
      if (!this.simId || this.agentStatus === "running_round") return;
      this.agentStatus = "running_round";
      this._sysLog("info", "agent", `Round ${this.agentRound} — agents are deliberating…`);
      try {
        const r = await fetch(`/api/runs/${this.jobId}/agent-sim/${this.simId}/round`, { method: "POST" });
        if (!r.ok) throw new Error(await r.text());
        const data = await r.json();
        this.agentRound = data.round_num + 1;
        this.roundMetrics = data.metrics;

        for (const a of (data.actions || [])) {
          if (a.action_type === "hold") continue;
          this._chat({
            actor: a.actor_id,
            role: this._actorType(a.actor_id),
            action: a.action_type,
            body: a.reasoning || "",
            magnitude: a.magnitude,
            round: data.round_num,
          });
        }
        for (const e of (data.events || [])) {
          this._sysLog("trace", "event", `${e.source_actor_id} → ${e.target_actor_id} · ${e.channel} · ${e.variable_affected} ${e.magnitude >= 0 ? "+" : ""}${(e.magnitude || 0).toFixed(3)}`);
          this._flashEdge(e.source_actor_id, e.target_actor_id);
        }
        if (data.metrics) {
          const m = data.metrics;
          this._sysLog("info", "metrics",
            `R${data.round_num} · GDP ${m.gdp_index?.toFixed(3)} · Infl ${m.inflation_rate?.toFixed(3)} · Unemp ${(m.unemployment_rate * 100)?.toFixed(1)}% · Bankr ${m.firm_bankruptcy_count}`);
        }
        this.agentStatus = data.round_num + 1 >= this.agentTotal ? "completed" : "ready";
        if (this.agentStatus === "completed") {
          this._sysLog("ok", "agent", `Simulation complete — all ${this.agentTotal} rounds finished.`);
        }
      } catch (e) {
        this.agentStatus = "error";
        this._sysLog("err", "agent", "Round failed — " + e.message);
      }
    },

    async playAgent() {
      if (this.agentPlaying) {
        this.agentPlaying = false;
        clearInterval(this._agentTimer);
        return;
      }
      this.agentPlaying = true;
      const tick = async () => {
        if (!this.agentPlaying || this.agentStatus === "completed" || this.agentStatus === "error") {
          this.agentPlaying = false;
          clearInterval(this._agentTimer);
          return;
        }
        await this.runOneRound();
      };
      await tick();
      this._agentTimer = setInterval(tick, this.agentSpeed);
    },

    async injectEvent() {
      if (!this.simId) return;
      const text = (this.injectText || "").trim();
      if (!text || text.length < 5) return;
      this.injecting = true;
      try {
        await fetch(`/api/runs/${this.jobId}/agent-sim/${this.simId}/inject`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ description: text }),
        });
        this._sysLog("warn", "inject", `⌘K · "${text}"`);
        this._chat({ actor: "operator", role: "operator", action: "inject", body: text, round: this.agentRound });
      } catch (e) {
        this._sysLog("err", "inject", "Inject failed — " + e.message);
      } finally {
        this.injectText = "";
        this.injecting = false;
      }
    },

    /* ===== chat / log helpers ===== */
    _chatId: 0,
    _chat(o) {
      const id = ++this._chatId;
      const role = o.role || "household";
      const color = role === "operator" ? T_LIVE.accent : (SPLIT_NODE_COLORS[role] || T_LIVE.fg3);
      const ts = new Date().toLocaleTimeString("en-US", { hour12: false });
      this.chat.push({ id, ts, color, ...o });
      if (this.chat.length > 120) this.chat.shift();
      // Mirror to legacy aggregated feed so any older bindings keep working.
      this.feed.unshift({
        id, ts, type: "decisions", actor: o.actor, color,
        icon: this._actionIcon(o.action || "info"),
        title: `${this._actorName(o.actor)} · ${o.action || "speaks"}`,
        body: o.body,
      });
      if (this.feed.length > 200) this.feed.length = 200;
    },
    _sysId: 0,
    _sysLog(level, source, body) {
      const id = ++this._sysId;
      const ts = new Date().toLocaleTimeString("en-US", { hour12: false });
      this.sysLog.push({ id, ts, level, source, body });
      if (this.sysLog.length > 180) this.sysLog.shift();
      // Mirror into legacy feed too.
      const colorMap = { info: T_LIVE.fg3, ok: T_LIVE.household, warn: T_LIVE.bank, err: T_LIVE.shocked, trace: T_LIVE.accent };
      this.feed.unshift({
        id, ts, type: "system", actor: null, color: colorMap[level] || T_LIVE.fg3,
        icon: { info: "i", ok: "✓", warn: "!", err: "×", trace: "·" }[level] || "i",
        title: `${source}`, body,
      });
      if (this.feed.length > 200) this.feed.length = 200;
    },

    get filteredFeed() {
      if (this.feedFilter === "all") return this.feed;
      return this.feed.filter((f) => f.type === this.feedFilter);
    },

    /* ===== seeding from result ===== */
    _feedFromResult(result) {
      if (!result) return;
      if (result.parse_result) {
        const pr = result.parse_result;
        this._sysLog("info", "parse",
          `${pr.triggering_event || pr.summary || "scenario parsed"} · scope=${pr.shock_params?.scope || "—"} · severity=${pr.shock_params?.severity ?? "—"}`);
      }
      const personas = result.personas || [];
      if (personas.length) {
        this._sysLog("info", "graph", `${personas.length} actor personas spawned: ${personas.slice(0, 3).map((p) => p.display_name).join(", ")}${personas.length > 3 ? "…" : ""}`);
      }
      this._chatFromResult(result);
      if (result.report?.sections) {
        for (const s of result.report.sections) {
          this._sysLog("ok", "report", s.heading);
        }
      }
    },

    // Build the chat thread from causal_chains: each causal event becomes
    // a line in the conversation, attributed to the source actor.
    _chatFromResult(result) {
      const chains = result?.causal_chains || [];
      if (!chains.length) return;
      let r = 0;
      for (const c of chains.slice(0, 4)) {
        for (const ev of (c.events || []).slice(0, 4)) {
          this._chat({
            actor: ev.source_actor_id,
            role: this._actorType(ev.source_actor_id),
            action: ev.channel || "transmits",
            body: `${ev.variable_affected} ${ev.magnitude >= 0 ? "+" : ""}${(ev.magnitude || 0).toFixed(3)} → ${ev.target_actor_id}${ev.description ? " · " + ev.description : ""}`,
            magnitude: ev.magnitude,
            round: ev.step ?? r++,
          });
        }
      }
    },

    /* ===== KG render ===== */
    _populateKG(network) {
      this.kgNodes = (network.nodes || []).map((n) => ({ ...n }));
      this.kgEdges = [];
      const seen = new Set();
      for (const e of (network.edges || [])) {
        const k = e.source + "|" + e.target;
        if (!seen.has(k)) { seen.add(k); this.kgEdges.push({ ...e }); }
      }
      this.stageCounters.nodes = this.kgNodes.length;
      this.stageCounters.edges = this.kgEdges.length;
      this.stageCounters.schemas = new Set(this.kgEdges.map((e) => e.kind || e.type)).size;
    },

    _renderKG() {
      const svg = this.$refs.splitKgSvg;
      if (!svg) return;
      const self = this;
      const d3sel = d3.select(svg);
      d3sel.selectAll("*").remove();
      if (!this.kgNodes.length) return;

      const rect = svg.getBoundingClientRect();
      const W = rect.width || 600, H = rect.height || 540;

      const defs = d3sel.append("defs");
      const glow = defs.append("filter").attr("id", "splitglow");
      glow.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "blur");
      glow.append("feMerge").selectAll("feMergeNode").data(["blur", "SourceGraphic"]).enter().append("feMergeNode").attr("in", (d) => d);

      const g = d3sel.append("g");
      d3sel.call(d3.zoom().scaleExtent([0.2, 6]).on("zoom", (e) => g.attr("transform", e.transform)));

      const nodes = this.kgNodes.map((n) => ({ ...n }));
      const links = this.kgEdges.map((e) => ({ ...e }));

      const sim = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id((d) => d.id).distance(50).strength(0.25))
        .force("charge", d3.forceManyBody().strength(-90))
        .force("center", d3.forceCenter(W / 2, H / 2))
        .force("collision", d3.forceCollide().radius((d) => (SPLIT_NODE_R[d.type] || 5) + 4));

      const linkG = g.append("g").attr("class", "links");
      this._kgLinkSel = linkG.selectAll("line").data(links).enter().append("line")
        .attr("stroke", (d) => edgeStyle(d).color)
        .attr("stroke-opacity", 0.45)
        .attr("stroke-width", (d) => d.type === "monetary_policy" ? 1.2 : 0.9)
        .attr("stroke-dasharray", (d) => edgeStyle(d).dash || "none");

      // Edge labels — small mono caps at the midpoint, in the design's fg3.
      this._kgLabelSel = linkG.selectAll("text").data(links).enter().append("text")
        .text((d) => edgeStyle(d).label)
        .attr("font-family", "JetBrains Mono, monospace")
        .attr("font-size", "8.5")
        .attr("fill", T_LIVE.fg3)
        .attr("opacity", this.showEdgeLabels ? 0.75 : 0)
        .attr("text-anchor", "middle")
        .attr("pointer-events", "none");

      const nodeSel = g.append("g").selectAll("g").data(nodes).enter().append("g")
        .attr("cursor", "pointer")
        .call(d3.drag()
          .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
          .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

      nodeSel.append("circle").attr("class", "kg-sel")
        .attr("r", (d) => (SPLIT_NODE_R[d.type] || 5) + 5)
        .attr("fill", "none").attr("stroke", T_LIVE.accent).attr("stroke-width", 1.5)
        .attr("opacity", 0);
      nodeSel.filter((d) => d.shocked).append("circle")
        .attr("r", (d) => (SPLIT_NODE_R[d.type] || 5) + 5)
        .attr("fill", "none").attr("stroke", T_LIVE.shocked).attr("stroke-width", 1)
        .attr("opacity", 0.5).style("animation", "cl-pulse 1.4s infinite");
      nodeSel.append("circle").attr("class", "kg-dot")
        .attr("r", (d) => SPLIT_NODE_R[d.type] || 5)
        .attr("fill", (d) => d.shocked ? T_LIVE.shocked : (SPLIT_NODE_COLORS[d.type] || T_LIVE.fg3));
      nodeSel.filter((d) => d.type === "central_bank" || d.type === "centralbank" || d.shocked).append("text")
        .text((d) => (d.label || d.id).replace(/_/g, " "))
        .attr("dx", (d) => (SPLIT_NODE_R[d.type] || 5) + 5)
        .attr("dy", 3)
        .attr("font-family", "JetBrains Mono, monospace")
        .attr("font-size", "9")
        .attr("fill", T_LIVE.fg2)
        .attr("pointer-events", "none");

      nodeSel.on("click", function (event, d) {
        event.stopPropagation();
        self.kgSelectedNode = self.kgNodes.find((n) => n.id === d.id) || d;
        g.selectAll(".kg-sel").transition().duration(150).attr("opacity", 0);
        d3.select(this).select(".kg-sel").transition().duration(150).attr("opacity", 0.8);
      });
      d3sel.on("click", () => {
        self.kgSelectedNode = null;
        g.selectAll(".kg-sel").transition().duration(150).attr("opacity", 0);
      });

      sim.on("tick", () => {
        this._kgLinkSel
          .attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
          .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
        this._kgLabelSel
          .attr("x", (d) => (d.source.x + d.target.x) / 2)
          .attr("y", (d) => (d.source.y + d.target.y) / 2);
        nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
      });

      this.kgSim = sim;
      this._kgNodeSel = nodeSel;
    },

    toggleEdgeLabels() {
      this.showEdgeLabels = !this.showEdgeLabels;
      if (this._kgLabelSel) this._kgLabelSel.transition().duration(160).attr("opacity", this.showEdgeLabels ? 0.75 : 0);
    },

    _flashEdge(srcId, tgtId) {
      if (!this._kgLinkSel) return;
      this._kgLinkSel.filter((d) => {
        const s = typeof d.source === "object" ? d.source.id : d.source;
        const t = typeof d.target === "object" ? d.target.id : d.target;
        return s === srcId && t === tgtId;
      })
        .transition().duration(180)
        .attr("stroke", T_LIVE.shocked).attr("stroke-width", 2.4).attr("stroke-opacity", 1)
        .transition().duration(1400)
        .attr("stroke", (d) => edgeStyle(d).color)
        .attr("stroke-width", (d) => d.type === "monetary_policy" ? 1.2 : 0.9)
        .attr("stroke-opacity", 0.45);

      if (this._kgNodeSel) {
        this._kgNodeSel.filter((d) => d.id === tgtId).select(".kg-dot")
          .transition().duration(180)
          .attr("fill", T_LIVE.shocked).attr("r", (d) => (SPLIT_NODE_R[d.type] || 5) + 3)
          .transition().duration(1400)
          .attr("fill", (d) => d.shocked ? T_LIVE.shocked : (SPLIT_NODE_COLORS[d.type] || T_LIVE.fg3))
          .attr("r", (d) => SPLIT_NODE_R[d.type] || 5);
      }
    },

    /* ===== misc ===== */
    _actorType(id) {
      if (!id) return "unknown";
      if (id.startsWith("household")) return "household";
      if (id.startsWith("firm")) return "firm";
      if (id.startsWith("bank_") || id.startsWith("bk_")) return "bank";
      if (id.startsWith("central") || id.startsWith("cb_")) return "central_bank";
      return "unknown";
    },
    _actorName(id) {
      if (!id) return "—";
      // Demo mode: prefer the node's label from kgNodes (matches DEMO_FIXTURE).
      const node = (this.kgNodes || []).find((n) => n.id === id);
      if (node && node.label) return node.label;
      const root = document.querySelector("[x-data]");
      const app = root && Alpine.$data(root);
      const personas = app?.result?.personas || [];
      const p = personas.find((p) => p.actor_id === id);
      return p?.display_name || id.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    },
    _actionIcon(type) {
      const icons = {
        set_rate: "%", signal: "!", tighten_credit: "↓", ease_credit: "↑",
        call_loans: "→", raise_prices: "▲", cut_prices: "▼", hire: "+",
        fire: "−", invest: "$", cut_costs: "/", spend_more: "•",
        save_more: "□", seek_credit: "?", demand_raise: "!", hold: "·",
        inject: "⌘", info: "i", transmits: "→",
      };
      return icons[type] || "›";
    },
    actorLabel(actor) { return this._actorName(actor); },
    roleLabel(role) {
      return ({ household: "Household", firm: "Firm", bank: "Bank", central_bank: "Central Bank", operator: "You · ⌘K" }[role] || role);
    },

    close() {
      this.active = false;
      this.agentPlaying = false;
      clearInterval(this._agentTimer);
      this._cancelDemo();
    },

    /* =====================================================================
       HACKATHON DEMO — fully client-side, no backend dependency.

       Three scripted stages drive the same Live View overlay we already
       built (graph left, conversation right, system terminal bottom):

         1. Ontology generation: pulse 12 nodes + 22 edges into the graph
            with mono labels (POLICY_RATE, CREDIT_SUPPLY, …) over ~5s.
            Right pane stays on Stages with 01 active. CTA → "Continue".
         2. Environment setup: render actor profile cards (hardcoded
            personas) into a new "actors" pane. Stages 02 done. CTA →
            "Generate Simulation Config".
         3. Simulation: 12 actor messages over ~14s, each pulses the
            corresponding edge in the graph. Right pane auto-switches to
            the Conversation tab. Stages 03 done. CTA → "Open Report"
            (closes the overlay; the underlying app already has the
            results pane wired up).

       The demo data lives in DEMO_FIXTURE so the narrative is editable
       without touching control flow. Every call to runDemo() resets all
       state and cancels in-flight timers from a previous demo.
    ===================================================================== */
    runDemo() {
      this._cancelDemo();
      this.active = true;
      this.mode = "demo";
      this.jobId = "demo_" + Math.random().toString(36).slice(2, 10);
      this.chat = [];
      this.sysLog = [];
      this.feed = [];
      this.kgNodes = [];
      this.kgEdges = [];
      this.kgSelectedNode = null;
      this.demoActors = [];
      this.stages = this.stages.map((s) => ({ ...s, status: "pending" }));
      this.stageCounters = { nodes: 0, edges: 0, schemas: 0, runs: DEMO_FIXTURE.run_count, horizon: DEMO_FIXTURE.horizon };
      this.demoStage = "ontology";
      this.activeRightTab = "stages";
      this._sysLog("info", "console", `Demo started · scenario "${DEMO_FIXTURE.title}"`);
      this._sysLog("info", "input", DEMO_FIXTURE.description);
      // Stage 1 begins.
      this._demoStartOntology();
    },

    // ---- stage 1: ontology -------------------------------------------------
    _demoStartOntology() {
      this.stages[0].status = "done";    // parse already implicitly done
      this.stages[1].status = "running"; // graph build
      this.$nextTick(() => this._renderKG());
      const totalNodes = DEMO_FIXTURE.nodes.length;
      const totalEdges = DEMO_FIXTURE.edges.length;
      this._sysLog("info", "ontology", `extracting ontology · ${totalNodes} entities · ${totalEdges} relations`);
      let i = 0;
      // Faster cadence for the denser AWS graph (32 nodes / 60+ edges).
      const stagger = 160;
      const tickNode = () => {
        if (this.demoStage !== "ontology") return;
        if (i >= totalNodes) return tickEdges();
        const n = DEMO_FIXTURE.nodes[i++];
        this.kgNodes.push({ ...n });
        this.stageCounters.nodes = this.kgNodes.length;
        if (n.shocked) this._sysLog("warn", "ontology", `actor flagged shocked · ${n.id}`);
        else this._sysLog("trace", "ontology", `entity ${n.id} · type=${n.type}`);
        // Re-render graph each step so D3 picks up the new node.
        this._renderKG();
        this._demoTimers.push(setTimeout(tickNode, stagger));
      };
      let j = 0;
      const tickEdges = () => {
        if (this.demoStage !== "ontology") return;
        if (j >= totalEdges) return finishOntology();
        const e = DEMO_FIXTURE.edges[j++];
        this.kgEdges.push({ ...e });
        this.stageCounters.edges = this.kgEdges.length;
        this.stageCounters.schemas = new Set(this.kgEdges.map((x) => x.type)).size;
        this._sysLog("trace", "graph", `edge ${e.source} →${edgeStyle(e).label}→ ${e.target}`);
        this._renderKG();
        this._demoTimers.push(setTimeout(tickEdges, 90));
      };
      const finishOntology = () => {
        if (this.demoStage !== "ontology") return;
        this.stages[1].status = "done";
        this._sysLog("ok", "ontology", `graph build complete · ${this.stageCounters.nodes} nodes / ${this.stageCounters.edges} edges / ${this.stageCounters.schemas} schemas`);
        this.demoStage = "awaitSetup";
      };
      this._demoTimers.push(setTimeout(tickNode, 200));
    },

    // ---- stage 2: environment setup ---------------------------------------
    demoContinueToSetup() {
      if (this.demoStage !== "awaitSetup") return;
      this.demoStage = "envSetup";
      this.activeRightTab = "actors";
      this.stages[2].status = "running";
      this._sysLog("info", "setup", "spawning actors · loading prior-library params · wiring transmission channels…");
      let i = 0;
      const tick = () => {
        if (this.demoStage !== "envSetup") return;
        if (i >= DEMO_FIXTURE.actors.length) return finish();
        const a = DEMO_FIXTURE.actors[i++];
        this.demoActors.push(a);
        this._sysLog("trace", "setup", `persona ready · ${a.display_name} (${a.actor_type})`);
        this._demoTimers.push(setTimeout(tick, 280));
      };
      const finish = () => {
        if (this.demoStage !== "envSetup") return;
        this._sysLog("ok", "setup", `${DEMO_FIXTURE.actors.length} actors profiled · environment ready`);
        this.demoStage = "awaitConfig";
      };
      this._demoTimers.push(setTimeout(tick, 200));
    },

    demoGenerateConfig() {
      if (this.demoStage !== "awaitConfig") return;
      this.demoStage = "configReady";
      this._sysLog("info", "config", "compiling SimulationConfig · 250 trajectories · horizon=48 steps · seed=0x4A7");
      this._demoTimers.push(setTimeout(() => {
        if (this.demoStage !== "configReady") return;
        this.stages[2].status = "done";
        this._sysLog("ok", "config", "config compiled · ready to simulate");
      }, 700));
    },

    // ---- stage 3: simulation ----------------------------------------------
    demoStartSimulation() {
      if (this.demoStage !== "configReady") return;
      this.demoStage = "simulating";
      this.activeRightTab = "conversation";
      this.stages[3].status = "running";
      this._sysLog("info", "sim", `launching ${DEMO_FIXTURE.run_count} parallel trajectories · agent communication enabled`);
      let i = 0;
      const tick = () => {
        if (this.demoStage !== "simulating") return;
        if (i >= DEMO_FIXTURE.messages.length) return finish();
        const m = DEMO_FIXTURE.messages[i++];
        const role = this._actorType(m.actor);
        this._chat({ actor: m.actor, role, action: m.action, body: m.body, magnitude: m.magnitude, round: i });
        if (m.target) {
          this._flashEdge(m.actor, m.target);
          this._sysLog("trace", "event", `${m.actor} → ${m.target} · ${m.action} · mag ${(m.magnitude >= 0 ? "+" : "") + Number(m.magnitude || 0).toFixed(3)}`);
        }
        this._demoTimers.push(setTimeout(tick, 1100));
      };
      const finish = () => {
        if (this.demoStage !== "simulating") return;
        this.stages[3].status = "done";
        this.stages[4].status = "done"; // synthesize + report deterministic post-step
        this._sysLog("ok", "sim", "ensemble converged · σ=0.018 · all trajectories complete");
        this._sysLog("ok", "report", "narrative generated · 4 sections · 142 evidence chips");
        this.demoStage = "done";
      };
      this._demoTimers.push(setTimeout(tick, 400));
    },

    _cancelDemo() {
      this._demoTimers.forEach(clearTimeout);
      this._demoTimers = [];
      this.demoStage = null;
    },
  };
};

/* =====================================================================
   DEMO_FIXTURE — AWS us-east-1 outage (Oct 19–20, 2025).
   Reality-grounded mock built from public post-mortems: a race
   condition in DynamoDB's DNS Enactor automation wiped regional DNS
   records for ~3h, cascaded into EC2 / NLB / Lambda for 12+ more
   hours, and took down 1,000+ downstream platforms (Coinbase,
   Snapchat, Reddit, Roblox, Fortnite, Venmo, Ring, Disney+, Slack,
   Lyft, McDonald's app, Delta, NYT, …). 6.5M+ Downdetector reports.

   Mapping into the simulation's actor vocabulary:
     central_bank (purple) → AWS Control Plane components
     bank         (yellow) → AWS regional services
     firm         (blue)   → customer platforms
     household    (green)  → end-users by region

   Each edge carries a `kind` override so the graph labels read in
   AWS terminology (DDB_QUERY, DNS_RESOLVE, USER_TRAFFIC, …) instead
   of the default economic terms.
===================================================================== */
const DEMO_FIXTURE = {
  title: "AWS us-east-1 outage · DynamoDB DNS race condition (Oct 20 2025)",
  description: "At 11:48 PM PDT Oct 19, a race condition in DynamoDB's DNS Enactor automation deleted all regional DynamoDB DNS records in us-east-1. EC2 launches stalled, NLB cascaded into AZ-failover storms, Lambda invocations queued. 113 AWS services degraded; 1,000+ downstream platforms — Coinbase, Robinhood, Snapchat, Reddit, Roblox, Fortnite, Venmo, Ring, Disney+, Slack, Lyft — affected for ~15 hours. 6.5M+ Downdetector reports. Insurance loss estimates $581M+.",
  run_count: 500,
  horizon: 96,
  nodes: [
    // ─── AWS Control Plane (3) — purple, "central_bank" type
    { id: "ddb_planner",     type: "central_bank", label: "DDB DNS Planner" },
    { id: "ddb_dns_enactor", type: "central_bank", label: "DNS Enactor", shocked: true },
    { id: "iam",             type: "central_bank", label: "IAM" },

    // ─── AWS Regional Services (12) — yellow, "bank" type
    { id: "dynamodb",  type: "bank", label: "DynamoDB", shocked: true },
    { id: "ec2",       type: "bank", label: "EC2" },
    { id: "lambda",    type: "bank", label: "Lambda" },
    { id: "nlb",       type: "bank", label: "NLB" },
    { id: "fargate",   type: "bank", label: "Fargate" },
    { id: "ecs",       type: "bank", label: "ECS" },
    { id: "sqs",       type: "bank", label: "SQS" },
    { id: "kinesis",   type: "bank", label: "Kinesis" },
    { id: "eks",       type: "bank", label: "EKS" },
    { id: "redshift",  type: "bank", label: "Redshift" },
    { id: "cloudwatch",type: "bank", label: "CloudWatch" },
    { id: "s3",        type: "bank", label: "S3" },

    // ─── Customer Platforms (12) — blue, "firm" type
    { id: "coinbase",  type: "firm", label: "Coinbase" },
    { id: "robinhood", type: "firm", label: "Robinhood" },
    { id: "snapchat",  type: "firm", label: "Snapchat" },
    { id: "reddit",    type: "firm", label: "Reddit" },
    { id: "roblox",    type: "firm", label: "Roblox" },
    { id: "fortnite",  type: "firm", label: "Fortnite" },
    { id: "venmo",     type: "firm", label: "Venmo" },
    { id: "ring",      type: "firm", label: "Ring" },
    { id: "disney",    type: "firm", label: "Disney+" },
    { id: "slack",     type: "firm", label: "Slack" },
    { id: "lyft",      type: "firm", label: "Lyft" },
    { id: "mcdonalds", type: "firm", label: "McDonald's" },

    // ─── End Users (5) — green, "household" type
    { id: "users_us_east", type: "household", label: "US-East users" },
    { id: "users_us_west", type: "household", label: "US-West users" },
    { id: "users_eu",      type: "household", label: "EU users" },
    { id: "users_apac",    type: "household", label: "APAC users" },
    { id: "users_global",  type: "household", label: "Global enterprise" },
  ],
  // Edges. `type` drives line color (re-using existing styling); `kind`
  // overrides the on-graph label. Mix of monetary_policy (purple lines for
  // control plane), interbank (yellow dashed for service-to-service), lending
  // (yellow solid for AWS→customer), employment (firm→users).
  edges: [
    // Control plane internals
    { source: "ddb_planner",     target: "ddb_dns_enactor", type: "monetary_policy", kind: "DNS_PLAN" },
    { source: "ddb_dns_enactor", target: "dynamodb",        type: "monetary_policy", kind: "DNS_RESOLVE" },
    { source: "iam",             target: "dynamodb",        type: "monetary_policy", kind: "AUTH" },
    { source: "iam",             target: "ec2",             type: "monetary_policy", kind: "AUTH" },
    { source: "iam",             target: "lambda",          type: "monetary_policy", kind: "AUTH" },
    { source: "iam",             target: "s3",              type: "monetary_policy", kind: "AUTH" },

    // Service-to-service (DynamoDB is upstream of nearly everything)
    { source: "dynamodb",   target: "ec2",        type: "interbank", kind: "DDB_METADATA" },
    { source: "dynamodb",   target: "lambda",     type: "interbank", kind: "DDB_QUERY" },
    { source: "dynamodb",   target: "fargate",    type: "interbank", kind: "DDB_QUERY" },
    { source: "dynamodb",   target: "ecs",        type: "interbank", kind: "DDB_QUERY" },
    { source: "dynamodb",   target: "sqs",        type: "interbank", kind: "DDB_QUERY" },
    { source: "dynamodb",   target: "kinesis",    type: "interbank", kind: "DDB_QUERY" },
    { source: "dynamodb",   target: "redshift",   type: "interbank", kind: "DDB_QUERY" },
    { source: "ec2",        target: "lambda",     type: "interbank", kind: "LAMBDA_HOST" },
    { source: "ec2",        target: "nlb",        type: "interbank", kind: "BACKEND" },
    { source: "ec2",        target: "fargate",    type: "interbank", kind: "HOST" },
    { source: "ec2",        target: "eks",        type: "interbank", kind: "HOST" },
    { source: "nlb",        target: "ec2",        type: "interbank", kind: "HEALTH_CHECK" },
    { source: "s3",         target: "lambda",     type: "interbank", kind: "DEPENDENCY" },
    { source: "cloudwatch", target: "ec2",        type: "interbank", kind: "METRICS" },
    { source: "cloudwatch", target: "lambda",     type: "interbank", kind: "METRICS" },
    { source: "kinesis",    target: "lambda",     type: "interbank", kind: "STREAM_TRIGGER" },

    // Customer platforms → AWS services
    { source: "coinbase",  target: "dynamodb", type: "lending", kind: "ORDER_BOOK" },
    { source: "coinbase",  target: "ec2",      type: "lending", kind: "COMPUTE" },
    { source: "robinhood", target: "dynamodb", type: "lending", kind: "ORDER_BOOK" },
    { source: "robinhood", target: "lambda",   type: "lending", kind: "SERVERLESS" },
    { source: "snapchat",  target: "dynamodb", type: "lending", kind: "STORY_INDEX" },
    { source: "snapchat",  target: "s3",       type: "lending", kind: "MEDIA_STORE" },
    { source: "reddit",    target: "ec2",      type: "lending", kind: "COMPUTE" },
    { source: "reddit",    target: "dynamodb", type: "lending", kind: "FEED_INDEX" },
    { source: "roblox",    target: "ec2",      type: "lending", kind: "GAME_SERVERS" },
    { source: "roblox",    target: "lambda",   type: "lending", kind: "SERVERLESS" },
    { source: "fortnite",  target: "ec2",      type: "lending", kind: "GAME_SERVERS" },
    { source: "fortnite",  target: "dynamodb", type: "lending", kind: "MATCHMAKING" },
    { source: "venmo",     target: "lambda",   type: "lending", kind: "PAYMENTS_API" },
    { source: "ring",      target: "lambda",   type: "lending", kind: "VIDEO_TRIGGER" },
    { source: "ring",      target: "kinesis",  type: "lending", kind: "VIDEO_STREAM" },
    { source: "disney",    target: "s3",       type: "lending", kind: "VIDEO_STORE" },
    { source: "slack",     target: "ec2",      type: "lending", kind: "COMPUTE" },
    { source: "slack",     target: "dynamodb", type: "lending", kind: "MESSAGE_INDEX" },
    { source: "lyft",      target: "ec2",      type: "lending", kind: "COMPUTE" },
    { source: "mcdonalds", target: "lambda",   type: "lending", kind: "ORDER_API" },

    // End users → platforms
    { source: "users_us_east", target: "coinbase",  type: "employment", kind: "TRAFFIC" },
    { source: "users_us_east", target: "robinhood", type: "employment", kind: "TRAFFIC" },
    { source: "users_us_east", target: "snapchat",  type: "employment", kind: "TRAFFIC" },
    { source: "users_us_east", target: "reddit",    type: "employment", kind: "TRAFFIC" },
    { source: "users_us_east", target: "ring",      type: "employment", kind: "TRAFFIC" },
    { source: "users_us_east", target: "venmo",     type: "employment", kind: "TRAFFIC" },
    { source: "users_us_west", target: "coinbase",  type: "employment", kind: "TRAFFIC" },
    { source: "users_us_west", target: "lyft",      type: "employment", kind: "TRAFFIC" },
    { source: "users_us_west", target: "slack",     type: "employment", kind: "TRAFFIC" },
    { source: "users_us_west", target: "mcdonalds", type: "employment", kind: "TRAFFIC" },
    { source: "users_eu",      target: "snapchat",  type: "employment", kind: "TRAFFIC" },
    { source: "users_eu",      target: "reddit",    type: "employment", kind: "TRAFFIC" },
    { source: "users_eu",      target: "fortnite",  type: "employment", kind: "TRAFFIC" },
    { source: "users_eu",      target: "disney",    type: "employment", kind: "TRAFFIC" },
    { source: "users_apac",    target: "fortnite",  type: "employment", kind: "TRAFFIC" },
    { source: "users_apac",    target: "roblox",    type: "employment", kind: "TRAFFIC" },
    { source: "users_apac",    target: "snapchat",  type: "employment", kind: "TRAFFIC" },
    { source: "users_global",  target: "slack",     type: "employment", kind: "TRAFFIC" },
    { source: "users_global",  target: "snapchat",  type: "employment", kind: "TRAFFIC" },
    { source: "users_global",  target: "disney",    type: "employment", kind: "TRAFFIC" },
  ],
  actors: [
    { actor_id: "ddb_dns_enactor", actor_type: "central_bank", display_name: "DynamoDB DNS Enactor",
      role: "Internal microservice · applies DNS plans to regional DDB endpoints",
      description: "Two enactors ran concurrently. Older plan delayed by retries; newer plan applied first; older plan then overwrote it; cleanup deleted the active plan — wiping ALL regional DDB DNS records.",
      tags: ["root_cause", "race_condition", "DNS"], vulnerability: "No mutual-exclusion lock between plan applications; cleanup cannot tell stale plan from active one." },
    { actor_id: "dynamodb", actor_type: "bank", display_name: "DynamoDB (us-east-1)",
      role: "Managed key-value store · regional endpoint dynamodb.us-east-1.amazonaws.com",
      description: "Endpoint resolution returned empty IP set after DNS wipe. API error rate 100% for ~3 hours. Upstream of EC2 metadata, Lambda functions, IAM token validation.",
      tags: ["shocked", "regional_endpoint", "p99=∞"], vulnerability: "Single-region DNS records · no client-side caching fallback for fresh sessions." },
    { actor_id: "ec2", actor_type: "bank", display_name: "EC2",
      role: "Compute service · DropletWorkflow Manager depends on DDB",
      description: "RunInstances backlog at 12M events. New instance launches stalled; existing instances kept running but state changes failed. DWF entered congestive collapse — 12+ hour cascade.",
      tags: ["dependent_on_ddb", "launch_failed"], vulnerability: "Control plane couples instance lifecycle to DDB metadata reads." },
    { actor_id: "nlb", actor_type: "bank", display_name: "Network Load Balancer",
      role: "L4 load balancer · health-check feedback loop",
      description: "Backend EC2 health checks failed at scale. NLB removed targets at unbounded rate — no velocity controls. AZ failover storms compounded the outage.",
      tags: ["amplifier", "no_velocity_control"], vulnerability: "Health-check failures triggered runaway target removal; AWS adding velocity limits post-incident." },
    { actor_id: "coinbase", actor_type: "firm", display_name: "Coinbase",
      role: "Crypto exchange · 110M+ users · trading halted",
      description: "Order book stale 8+ minutes. Withdrawals frozen. Customer support volume +1100%. Trading paused for ~6 hours. Estimated $200M+ in foregone fee revenue.",
      tags: ["high_severity", "regulated", "fintech"], vulnerability: "Single-region deployment for order matching · DDB on the hot path." },
    { actor_id: "snapchat", actor_type: "firm", display_name: "Snapchat",
      role: "Social · 380M DAU · stories indexed in DDB",
      description: "Story retrieval failing globally. Snap upload queueing. Engagement -94%. Brand campaigns auto-paused. Recovery began after DDB DNS restored at 04:14 UTC.",
      tags: ["consumer_scale", "media_heavy"], vulnerability: "Story index hot path through DynamoDB — no read replica failover." },
    { actor_id: "users_us_east", actor_type: "household", display_name: "US-East users",
      role: "~120M consumers · highest direct exposure to us-east-1",
      description: "Cannot withdraw from Coinbase. Cannot trade on Robinhood. Lost Slack workspaces. Ring doorbells offline. #AWSdown trending #1 on Twitter for 8 hours.",
      tags: ["affected", "vocal"], vulnerability: "Most consumer apps default-route to us-east-1 control planes." },
  ],
  // Conversation script. 14 messages tracking the actual incident
  // sequence: race-condition root cause → DDB endpoint wiped → service
  // cascade → customer-platform impact → user impact → mitigation →
  // recovery. Magnitudes are signed by direction of impact.
  messages: [
    // 1 — Root cause fires
    { actor: "ddb_planner", target: "ddb_dns_enactor", action: "dns_plan", magnitude: 0.020,
      body: "Generated plan gen-1247 for regional endpoint. Older enactor still applying gen-1241 — high latency on retry path. Cleanup workers staging deletion of plans <gen-1244." },
    // 2 — Race condition triggers DNS wipe
    { actor: "ddb_dns_enactor", target: "dynamodb", action: "dns_race", magnitude: -1.000,
      body: "Race detected post-mortem: gen-1241 (delayed) overwrote gen-1247 after gen-1247 had triggered cleanup. Cleanup then deleted the active plan as 'stale.' Regional DDB DNS records: empty set. Endpoint unresolvable globally." },
    // 3 — DynamoDB → EC2
    { actor: "dynamodb", target: "ec2", action: "service_unavailable", magnitude: -0.940,
      body: "API error rate 100%. Cannot serve metadata for new EC2 launches. DropletWorkflow Manager retry storm starting. RunInstances backlog growing at 8k/sec." },
    // 4 — DynamoDB → Lambda
    { actor: "dynamodb", target: "lambda", action: "service_unavailable", magnitude: -0.880,
      body: "Function metadata fetch failing. Invocation queue depth 12M events and growing. Trigger backlog from Kinesis, S3, EventBridge all stalled. Cold-starts impossible." },
    // 5 — EC2 → NLB feedback loop
    { actor: "ec2", target: "nlb", action: "health_fail", magnitude: -0.760,
      body: "Backend health checks failing at scale across all AZs. NLB removing targets at unbounded rate — no velocity control mechanism. AZ failover storms compounding the outage." },
    // 6 — NLB amplifier feedback
    { actor: "nlb", target: "ec2", action: "amp_failover", magnitude: -0.620,
      body: "Removed 47% of healthy targets after health-check feedback loop. Traffic concentrating on remaining targets — they fail under load — more removals. Classic cascading collapse." },
    // 7 — Coinbase impact
    { actor: "coinbase", target: "ec2", action: "platform_down", magnitude: -0.890,
      body: "Trading halted for 4.2M active users. Order book stale by 8 minutes. Withdrawal queue frozen at 18,400 pending. PR statement drafted. CFO calling AWS TAM." },
    // 8 — Snapchat impact
    { actor: "snapchat", target: "dynamodb", action: "platform_down", magnitude: -0.940,
      body: "Story retrieval failing globally. 380M DAUs hitting empty feeds. Snap-upload backlog 1.2B requests queued. Brand campaigns auto-paused. Engagement -94% in 11 minutes." },
    // 9 — Reddit impact
    { actor: "reddit", target: "ec2", action: "platform_down", magnitude: -0.730,
      body: "Front page degraded · ranking pipeline stale 14 min. New posts not persisting reliably. 'Unable to fetch comments' errors at scale. r/aws hitting 24k concurrent users discussing the outage itself." },
    // 10 — Robinhood
    { actor: "robinhood", target: "lambda", action: "platform_down", magnitude: -0.810,
      body: "Crypto orders failing — 100% error rate on placement. Equity routes timing out at the broker layer. Customer support call volume +1100%. SEC inquiry incoming if outage exceeds 4h during market hours." },
    // 11 — User-side impact (US East)
    { actor: "users_us_east", target: "coinbase", action: "user_impact", magnitude: -0.95,
      body: "Cannot withdraw funds. Cannot trade. 380k support tickets filed in 90 minutes. #AWSdown trending #1 on Twitter, 4.2M tweets. Ring doorbells offline — security implications." },
    // 12 — User-side impact (Global enterprise)
    { actor: "users_global", target: "slack", action: "user_impact", magnitude: -0.83,
      body: "Workspaces failing to load. Workplace productivity halted globally — Slack affected ≈80M knowledge workers. Estimated $4.2B+ in lost productivity per Atlassian/Pragmatic Engineer estimates." },
    // 13 — Mitigation
    { actor: "ddb_dns_enactor", target: "dynamodb", action: "mitigation", magnitude: 0.700,
      body: "Manually applied corrected DNS plan. DynamoDB DNS Planner + Enactor disabled GLOBALLY pending fix. Race-condition root-cause confirmed. Velocity controls being added to NLB to prevent future feedback storms." },
    // 14 — Recovery
    { actor: "dynamodb", target: "coinbase", action: "recovery", magnitude: 0.880,
      body: "Endpoint dynamodb.us-east-1.amazonaws.com resolving normally at 04:14 UTC. EC2 launch backlog draining — full recovery ETA 5h. CloudWatch alarms returning to green. Final timeline: ~3h DDB outage + 12h EC2 cascade = 15h total." },
  ],
};
