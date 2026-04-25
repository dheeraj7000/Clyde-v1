/* Clyde — frontend logic. Alpine.js component + ClydeUI SVG primitives. */

const CORE_METRICS = [
  "gdp_index",
  "inflation_rate",
  "unemployment_rate",
  "gini_coefficient",
  "credit_tightening_index",
  "firm_bankruptcy_count",
  "bank_stress_index",
  "consumer_confidence",
  "interbank_freeze",
];

const PRETTY = {
  gdp_index: "GDP index",
  inflation_rate: "Inflation rate",
  unemployment_rate: "Unemployment rate",
  gini_coefficient: "Gini coefficient",
  credit_tightening_index: "Credit tightening",
  firm_bankruptcy_count: "Firm bankruptcies",
  bank_stress_index: "Bank stress",
  consumer_confidence: "Consumer confidence",
  interbank_freeze: "Interbank freeze",
};

// Branch colors — picked from the design's semantic palette so overlays
// read as alternative futures, not arbitrary brand hues.
const BRANCH_COLORS = ["#FBBF24", "#4ADE80", "#F87171", "#60A5FA", "#C084FC", "#A8A6FF"];

// AWS us-east-1 outage report fixture. The `summary` string is the
// ≤50-line synthesis used both for the on-screen narrative and as the
// payload sent to ElevenLabs for TTS playback. The `tts` field is a
// shorter spoken-cadence variant — same facts, no markdown, no URLs.
const AWS_REPORT_FIXTURE = {
  title: "Predicted Economic Outcome of AWS US-EAST-1 Outage",
  summary: `**Event** · AWS us-east-1 outage · Oct 19–20 2025 · 15-hour impact
**Root cause** · Race condition in DynamoDB DNS Enactor automation. Two enactors processed concurrent plans; a delayed older plan overwrote a newer plan; cleanup then deleted the active plan as stale, wiping all regional DynamoDB DNS records.

**Cascade**
- DynamoDB endpoint unresolvable for ~3 hours
- EC2 RunInstances backlog grew to 12M events
- NLB feedback loop without velocity controls amplified outage
- 113 AWS services degraded for 12+ more hours

**Customer impact (1,000+ platforms · 6.5M+ Downdetector reports)**
- **Finance / payments** · Coinbase, Robinhood, Venmo, Chime, Lloyds — failed transactions, frozen withdrawals
- **Social** · Snapchat (-94% engagement), Reddit, Signal — story / feed retrieval failed
- **Gaming** · Roblox, Fortnite, Clash Royale — sessions interrupted, matchmaking down
- **Streaming** · Disney+, Hulu, Netflix — playback failures
- **Enterprise** · Slack (~80M knowledge workers), Zoom — workspaces unreachable
- **Other** · Ring doorbells, Lyft, McDonald's app, NYT, Delta, United

## Predicted economic loss

| Band | Estimate |
|---|---|
| Conservative | $250M – $400M |
| **Expected** | **$500M – $650M direct U.S. business losses** |
| Severe | $700M – $1B+ |
| Insurance estimate | $581M+ (CyberCube) |

**Confidence: medium.** Indirect global losses likely exceed direct losses (productivity, SLA credits, churn).

## Loss drivers

- Failed transactions — **high**
- Productivity loss — **medium-high** ($4.2B+ Slack/SaaS productivity halt)
- Customer trust / reputation — **medium-high**
- Engineering recovery cost — **high**
- SLA credits + insurance claims — **medium**

## Sector breakdown

| Sector | Severity |
|---|---|
| Finance & payments | Very High |
| Retail / e-commerce | High |
| Gaming & entertainment | High |
| Social platforms | Medium-High |
| Streaming / media | Medium-High |
| Education / government | Medium |

## Recommendations
1. **Multi-region failover** — avoid us-east-1 control-plane dependence
2. **Test disaster recovery** quarterly, not annually
3. **Build fallback modes** for payments, login, ordering, customer comms
4. **Review cloud-outage insurance** — direct + business-interruption coverage

**Sources** · AWS post-event summary, ThousandEyes, InfoQ, Pragmatic Engineer, Parametrix.`,
  // ElevenLabs spoken version — concise, no markdown, no URLs, no tables.
  tts: "On October 19 to 20, 2025, AWS suffered a 15-hour outage in the U.S. East 1 region. " +
       "The root cause was a race condition in DynamoDB's DNS Enactor automation. " +
       "Two enactors ran concurrently. A delayed older plan overwrote a newer plan. " +
       "Cleanup then deleted the active plan as stale, wiping all regional DynamoDB DNS records. " +
       "DynamoDB was unresolvable for about three hours. EC2 launches stalled, with a backlog of 12 million events. " +
       "Network Load Balancers entered a health-check feedback loop without velocity controls, amplifying the outage. " +
       "113 AWS services degraded for more than 12 additional hours. " +
       "Over 1,000 customer platforms were affected: Coinbase, Robinhood, Snapchat, Reddit, Roblox, Fortnite, Venmo, Ring, Disney Plus, Slack, Lyft, and more. " +
       "Predicted economic loss: 500 to 650 million dollars in direct U.S. business losses, with indirect global losses likely higher. " +
       "Confidence: medium. Insurance loss estimates: 581 million dollars and rising. " +
       "The largest impacts were in finance, payments, gaming, and enterprise productivity. " +
       "Top recommendations: deploy multi-region failover, test disaster recovery quarterly, build payment and login fallbacks, and review cloud-outage insurance coverage. " +
       "This is the Clyde economic-impact prediction for the AWS U.S. East 1 outage.",
  stats: [
    { label: "Direct U.S. loss", value: "$500–650M", tone: "negative" },
    { label: "Insurance est.", value: "$581M+", tone: "warn" },
    { label: "Platforms hit", value: "1,000+", tone: "warn" },
    { label: "Outage duration", value: "15h", tone: "negative" },
    { label: "Reports filed", value: "6.5M+", tone: "neutral" },
    { label: "Confidence", value: "Medium", tone: "neutral" },
  ],
};

// Stages we expect from the backend; keys are matched as substrings of progress.stage
const PIPELINE_STAGES = [
  { key: "parse", label: "Parse" },
  { key: "synth", label: "Synthesize" },
  { key: "simul", label: "Simulate" },
  { key: "diverg", label: "Diverge" },
  { key: "report", label: "Report" },
];

function clydeApp() {
  return {
    // health
    health: null,

    // input
    description: "",
    runCount: 24,
    rngSeed: 0,
    horizonSteps: null,
    useAnalogs: true,
    samples: [],

    // job state
    loading: false,
    job: null,
    pollHandle: null,
    result: null,
    _currentJobId: null,

    // ui state
    activeTab: "report",
    tabs: [
      { key: "report", label: "Report" },
      { key: "actors", label: "Actor Profiles" },
      { key: "influence", label: "Influence Config" },
      { key: "causal", label: "Causal Chains" },
      { key: "watchlist", label: "Watchlist" },
      { key: "network", label: "Network Graph" },
      { key: "simulation", label: "Simulation" },
      { key: "branches", label: "Branches" },
      { key: "raw", label: "Raw JSON" },
    ],
    selectedMetric: "gdp_index",
    toast: "",

    // branches
    injection: "",
    forking: false,
    branches: [], // {branch_id, label, injection, color, paths, divergence, status, progress}

    // chart resize observers (Chart.js was replaced by ClydeUI SVG primitives)
    _fanRO: null,
    _divRO: null,

    // Multi-modal input state — ElevenLabs voice + doc upload affordances.
    // Both are demo on-ramps: regardless of which input is used, runDemoFlow()
    // plays the canonical AWS us-east-1 incident from DEMO_FIXTURE.
    voiceState: "idle",       // idle | listening | transcribing
    uploadState: "idle",      // idle | ingesting
    uploadedDocName: "",

    // Report overlay (shown after demo completion, with TTS playback)
    reportOpen: false,
    reportTtsState: "idle",   // idle | loading | playing
    _reportAudio: null,
    reportTitle: AWS_REPORT_FIXTURE.title,
    reportSummary: AWS_REPORT_FIXTURE.summary,
    reportStats: AWS_REPORT_FIXTURE.stats,

    // simulation replay state
    simStep: 0,
    simPlaying: false,
    simSpeed: 600,
    _simTimer: null,
    _networkSim: null,

    // node detail panel state
    selectedNode: null,
    simSelectedNode: null,
    personaFilter: "all",

    stages: PIPELINE_STAGES,

    featureCards: [
      {
        title: "Plain-English shocks",
        body: "An LLM extracts shock type, severity, scope, and time horizon from your scenario.",
        icon: '<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke-linecap="round" stroke-linejoin="round"/></svg>',
      },
      {
        title: "Agent-based ensemble",
        body: "A configurable Monte Carlo across firms, banks, and households produces fan-chart bands.",
        icon: '<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12h4l3-9 4 18 3-9h4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
      },
      {
        title: "Counterfactual branches",
        body: "Inject an intervention mid-flight and Clyde forks the world to overlay the alternative.",
        icon: '<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 3v12m0 0a3 3 0 1 0 0 6 3 3 0 0 0 0-6zm12-12a3 3 0 1 0 0 6 3 3 0 0 0 0-6zm0 6c0 6-12 3-12 9" stroke-linecap="round" stroke-linejoin="round"/></svg>',
      },
    ],

    /* ============================ init ============================ */
    async init() {
      await this.loadHealth();
      await this.loadSamples();
    },

    async loadHealth() {
      try {
        const r = await fetch("/api/health");
        if (!r.ok) throw new Error("health " + r.status);
        this.health = await r.json();
      } catch (e) {
        this.health = { status: "unreachable", provider: "mock" };
        this.flash("Backend unreachable — running with stubs. (" + e.message + ")");
      }
    },

    async loadSamples() {
      try {
        const r = await fetch("/api/scenarios/sample");
        if (!r.ok) throw new Error("samples " + r.status);
        const data = await r.json();
        this.samples = Array.isArray(data) ? data : data.scenarios || [];
      } catch {
        // non-fatal
        this.samples = [];
      }
    },

    loadSample(name) {
      if (!name) return;
      const s = this.samples.find((x) => x.name === name);
      if (s) this.description = s.description || "";
    },

    /* ============================ run ============================ */
    async run() {
      if (!this.description.trim() || this.loading) return;
      this.loading = true;
      this.result = null;
      this.branches = [];
      this.job = null;
      this.toast = "";
      try {
        const r = await fetch("/api/runs", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            description: this.description,
            run_count: this.runCount,
            horizon_steps: this.horizonSteps || null,
            provider: "auto",
            rng_seed: this.rngSeed,
            ensemble_seed: this.rngSeed,
            use_analogs: this.useAnalogs,
          }),
        });
        if (!r.ok) throw new Error("POST /api/runs " + r.status);
        const { job_id } = await r.json();
        this._currentJobId = job_id;
        this.job = { job_id, status: "pending", progress: { stage: "queued", percent: 0 } };
        // Mirofish-style: auto-open Live View as soon as the run starts.
        // The split-screen Alpine component reads job state via pumpProgress()
        // on each poll cycle, so the user sees stages, the graph, and the
        // conversation populate live.
        this._openLiveView(job_id);
        this.pollJob(job_id);
      } catch (e) {
        this.loading = false;
        this.flash("Failed to start run: " + e.message);
      }
    },

    _liveScreen() {
      // x-ref doesn't cross nested x-data scopes in Alpine v3, so resolve
      // the splitView via a plain querySelector. Then unwrap its Alpine data.
      const ref = this.$refs.splitView || document.querySelector('[x-ref="splitView"]');
      if (!ref) return null;
      try { return window.Alpine ? window.Alpine.$data(ref) : (ref._x_dataStack && ref._x_dataStack[0]); }
      catch { return null; }
    },
    _openLiveView(jobId) {
      const live = this._liveScreen();
      if (live) live.showLive(jobId, this.description);
    },
    runDemoFlow() {
      const live = this._liveScreen();
      if (live) live.runDemo();
    },

    // Voice input — browser SpeechRecognition (Chrome/Edge native).
    // Toggles: click while idle → start listening; click while listening
    // → stop and run demo. Live transcript streams into the textarea so
    // the user (and audience) can see exactly what was heard. The mock
    // simulation runs regardless of what was transcribed; if nothing was
    // captured, we fall back to a default AWS scenario phrase.
    _recognition: null,
    _finalTranscript: "",
    startVoiceInput() {
      // Toggle off if already listening.
      if (this.voiceState === "listening") {
        try { this._recognition?.stop(); } catch {}
        return;
      }
      if (this.voiceState !== "idle") return;
      const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!Rec) {
        this.flash("Voice not supported in this browser — try Chrome or Edge.");
        return;
      }
      this._finalTranscript = "";
      this.description = "";
      this.voiceState = "listening";
      const rec = new Rec();
      rec.continuous = true;
      rec.interimResults = true;
      rec.lang = "en-US";
      rec.onresult = (e) => {
        let final = "", interim = "";
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const t = e.results[i][0].transcript;
          if (e.results[i].isFinal) final += t;
          else interim += t;
        }
        if (final) this._finalTranscript += final;
        // Live update — Alpine reactivity picks this up and the textarea
        // shows the transcript as it's spoken.
        this.description = this._finalTranscript + interim;
      };
      rec.onerror = (e) => {
        this.voiceState = "idle";
        const code = e?.error || "unknown";
        if (code === "not-allowed" || code === "service-not-allowed") {
          this.flash("Microphone permission denied — enable mic and retry.");
        } else if (code !== "aborted") {
          this.flash("Voice error: " + code);
        }
      };
      rec.onend = () => {
        // Brief "transcribing" state for visual continuity, then fire demo.
        if (this.voiceState !== "listening") return;
        this.voiceState = "transcribing";
        setTimeout(() => {
          this.voiceState = "idle";
          const final = (this._finalTranscript || "").trim();
          if (final) {
            this.description = final;
          } else {
            // Empty transcript fallback — still play the mock so the demo
            // never dead-ends.
            this.description = "AWS us-east-1 outage from a DynamoDB DNS race condition cascading into EC2 and NLB.";
          }
          this.runDemoFlow();
        }, 350);
      };
      this._recognition = rec;
      try {
        rec.start();
      } catch (e) {
        this.voiceState = "idle";
        this.flash("Voice start failed: " + e.message);
      }
    },

    // ─── Report overlay + ElevenLabs TTS ────────────────────────────
    // Called from the demo's "Open Report" CTA (see split.js close path).
    // Closes the Live View overlay, then opens the Report modal which
    // shows the AWS economic-impact summary with a "Listen" button.
    openReport() {
      this.reportOpen = true;
      this.stopReportTts();
    },
    closeReport() {
      this.reportOpen = false;
      this.stopReportTts();
    },
    stopReportTts() {
      if (this._reportAudio) {
        try { this._reportAudio.pause(); } catch {}
        try { URL.revokeObjectURL(this._reportAudio.src); } catch {}
        this._reportAudio = null;
      }
      this.reportTtsState = "idle";
    },
    async playReportTts() {
      if (this.reportTtsState !== "idle") {
        // toggle off if currently playing
        this.stopReportTts();
        return;
      }
      this.reportTtsState = "loading";
      try {
        const r = await fetch("/api/tts", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ text: AWS_REPORT_FIXTURE.tts }),
        });
        if (!r.ok) {
          const msg = await r.text();
          throw new Error(`TTS ${r.status}: ${msg.slice(0, 200)}`);
        }
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        this._reportAudio = audio;
        audio.addEventListener("ended", () => { this.reportTtsState = "idle"; this._reportAudio = null; URL.revokeObjectURL(url); });
        audio.addEventListener("error", () => { this.reportTtsState = "idle"; this.flash("Audio playback failed"); });
        await audio.play();
        this.reportTtsState = "playing";
      } catch (e) {
        this.reportTtsState = "idle";
        this.flash("TTS failed: " + e.message);
      }
    },

    // Doc upload — accepts a PDF/MD/TXT, ingests for 1.2s, then runs
    // the demo. The file is intentionally not parsed: the user said the
    // mock plays regardless of input; the doc is contextual seed only.
    handleDocUpload(ev) {
      const file = ev.target?.files?.[0];
      if (!file) return;
      this.uploadedDocName = file.name.length > 28 ? file.name.slice(0, 25) + "…" : file.name;
      this.uploadState = "ingesting";
      this.description = "Ingested " + file.name + " — extracting events…";
      setTimeout(() => {
        this.uploadState = "idle";
        this.description = "Loaded scenario from " + file.name + " · AWS us-east-1 outage (Oct 20, 2025)";
        this.runDemoFlow();
      }, 1200);
      // Reset the input so re-uploading the same file fires `change`.
      try { ev.target.value = ""; } catch {}
    },

    async pollJob(jobId) {
      clearTimeout(this.pollHandle);
      try {
        const r = await fetch(`/api/runs/${jobId}`);
        if (!r.ok) throw new Error("poll " + r.status);
        const j = await r.json();
        // Ignore late responses for jobs the user has already replaced
        if (this._currentJobId !== jobId) return;
        this.job = j;
        // Stream progress into the Live View on every tick.
        const live = this._liveScreen();
        if (live) live.pumpProgress(j);
        if (j.status === "completed") {
          this.loading = false;
          this.result = j.result;
          // pick a sensible default metric
          const avail = this.availableMetrics();
          if (!avail.includes(this.selectedMetric)) this.selectedMetric = avail[0] || "gdp_index";
          await this.$nextTick();
          this.renderFanChart();
          this.renderDivChart();
          if (live) live.onResult(j.result);
        } else if (j.status === "failed") {
          this.loading = false;
          this.flash("Run failed: " + (j.error || "unknown error"));
        } else {
          this.pollHandle = setTimeout(() => this.pollJob(jobId), 800);
        }
      } catch (e) {
        if (this._currentJobId !== jobId) return;
        this.loading = false;
        this.flash("Polling error: " + e.message);
      }
    },

    /* ============================ branch ============================ */
    async fork() {
      if (!this.injection.trim() || !this.job) return;
      const text = this.injection.trim();
      this.forking = true;
      try {
        const r = await fetch(`/api/runs/${this.job.job_id}/branches`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ injection_text: text }),
        });
        if (!r.ok) throw new Error("POST branches " + r.status);
        const { branch_id } = await r.json();
        const color = BRANCH_COLORS[this.branches.length % BRANCH_COLORS.length];
        const label = `B${this.branches.length + 1}`;
        const branch = {
          branch_id,
          label,
          injection: text,
          color,
          status: "pending",
          progress: { stage: "queued", percent: 0 },
          paths: null,
          divergence: null,
        };
        this.branches.push(branch);
        this.injection = "";
        this.activeTab = "branches";
        this.pollBranch(branch);
      } catch (e) {
        this.flash("Failed to fork: " + e.message);
      } finally {
        this.forking = false;
      }
    },

    async pollBranch(branch) {
      try {
        const r = await fetch(`/api/runs/${this.job.job_id}/branches/${branch.branch_id}`);
        if (!r.ok) throw new Error("branch poll " + r.status);
        const j = await r.json();
        // Ignore late responses if the branch has been removed (e.g. new run started)
        if (!this.branches.includes(branch)) return;
        branch.status = j.status;
        branch.progress = j.progress || branch.progress;
        if (j.status === "completed") {
          const res = j.result || {};
          branch.paths = res.paths || null;
          branch.divergence = res.divergence || null;
          this.renderFanChart();
          this.renderDivChart();
        } else if (j.status === "failed") {
          this.flash("Branch failed: " + (j.error || "unknown"));
        } else {
          setTimeout(() => this.pollBranch(branch), 800);
        }
      } catch (e) {
        if (!this.branches.includes(branch)) return;
        this.flash("Branch polling: " + e.message);
      }
    },

    /* ============================ helpers ============================ */
    providerLabel(p) {
      if (!p) return "—";
      const m = { openrouter: "OpenRouter", cerebras: "Cerebras", mock: "Demo Mode", openai: "OpenAI" };
      return m[p.toLowerCase()] || p;
    },

    progressStage() {
      if (!this.job) return "Initializing…";
      return this.job.progress?.stage || (this.job.status === "pending" ? "Queued" : "Working…");
    },

    progressPercent() {
      if (!this.job) return 0;
      return Math.max(0, Math.min(100, Math.round(this.job.progress?.percent ?? 0)));
    },

    stageReached(key) {
      const cur = (this.job?.progress?.stage || "").toLowerCase();
      const idxCur = PIPELINE_STAGES.findIndex((s) => cur.includes(s.key));
      const idxThis = PIPELINE_STAGES.findIndex((s) => s.key === key);
      if (idxCur < 0) return false;
      return idxThis <= idxCur;
    },

    availableMetrics() {
      const central = this.result?.paths?.central;
      if (!Array.isArray(central) || !central.length) return CORE_METRICS;
      const sample = central[0] || {};
      const keys = Object.keys(sample).filter((k) => k !== "step");
      // Boolean metrics like interbank_freeze can chart but as 0/1
      return keys.length ? keys : CORE_METRICS;
    },

    prettyMetric(m) {
      return PRETTY[m] || m.replace(/_/g, " ");
    },

    shockTiles() {
      const cfg = this.result?.shock_config || {};
      const parse = this.result?.parse_result || {};
      const pick = (...keys) => {
        for (const k of keys) {
          const v = cfg[k] ?? parse[k];
          if (v !== undefined && v !== null && v !== "") return v;
        }
        return "—";
      };
      return [
        {
          label: "Shock type",
          value: String(pick("shock_type", "type", "category")),
          sub: String(parse.summary || cfg.description || ""),
        },
        {
          label: "Severity",
          value: this.fmtNum(pick("severity", "magnitude", "intensity")),
          sub: cfg.severity_label || parse.severity_label || "",
        },
        {
          label: "Scope",
          value: String(pick("scope", "sector", "region")),
          sub: cfg.scope_detail || "",
        },
        {
          label: "Time horizon",
          value: String(pick("time_horizon", "horizon", "duration")),
          sub: cfg.time_unit || "steps",
        },
      ];
    },

    fmtNum(v) {
      if (v === "—" || v === null || v === undefined) return "—";
      if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
      return String(v);
    },

    watchlistItems() {
      const w = this.result?.watchlist;
      if (!w) return [];
      if (Array.isArray(w)) return w;
      if (Array.isArray(w.indicators)) return w.indicators;
      if (Array.isArray(w.items)) return w.items;
      return [];
    },

    eventTitle(ev) {
      if (typeof ev === "string") return ev;
      return ev.event_type || ev.title || ev.type || ev.actor || "event";
    },

    eventDetail(ev) {
      if (typeof ev === "string") return "";
      const bits = [];
      if (ev.step !== undefined) bits.push("step " + ev.step);
      if (ev.actor) bits.push("actor: " + ev.actor);
      if (ev.target) bits.push("→ " + ev.target);
      if (ev.magnitude !== undefined) bits.push("magnitude " + this.fmtNum(ev.magnitude));
      if (ev.description) bits.push(ev.description);
      return bits.join(" · ");
    },

    renderMd(text) {
      if (!text) return "";
      try {
        const html = window.marked.parse(String(text), { breaks: true, gfm: true });
        return window.DOMPurify ? window.DOMPurify.sanitize(html) : html;
      } catch {
        return String(text).replace(/</g, "&lt;");
      }
    },

    flash(msg) {
      this.toast = msg;
      clearTimeout(this._toastT);
      this._toastT = setTimeout(() => (this.toast = ""), 6000);
    },

    /* ============================ charts ============================ */
    extractSeries(paths, metric) {
      if (!paths) return null;
      const get = (key) => {
        const arr = paths[key];
        if (!Array.isArray(arr)) return null;
        return arr.map((row) => {
          const v = row?.[metric];
          if (typeof v === "boolean") return v ? 1 : 0;
          return typeof v === "number" ? v : null;
        });
      };
      const central = get("central");
      if (!central) return null;
      return {
        steps: paths.central.map((r) => r.step ?? 0),
        central,
        optimistic: get("optimistic"),
        pessimistic: get("pessimistic"),
        tail_upper: get("tail_upper"),
        tail_lower: get("tail_lower"),
      };
    },

    // Render fan chart into the host #fanHost via SVG primitive (replaces Chart.js).
    renderFanChart() {
      const host = this.$refs.fanHost;
      if (!host || !this.result) return;
      const series = this.extractSeries(this.result.paths, this.selectedMetric);
      if (!series) { host.innerHTML = ""; return; }

      const bands = window.ClydeUI.bandifyFromQuantiles(series);
      const branches = this.branches
        .filter((b) => b.paths)
        .map((b) => {
          const bs = this.extractSeries(b.paths, this.selectedMetric);
          return bs ? { label: b.label, data: bs.central, color: b.color } : null;
        })
        .filter(Boolean);

      // Make the chart fill its container responsively; ResizeObserver
      // re-renders when the panel resizes.
      const draw = () => {
        const w = Math.max(360, host.clientWidth || 720);
        const h = Math.max(220, host.clientHeight || 320);
        host.innerHTML = window.ClydeUI.fanChart({
          series: { steps: series.steps, central: series.central, ...bands },
          branches,
          w, h,
          yLabel: this.prettyMetric(this.selectedMetric),
          xLabel: "step",
        });
      };
      draw();
      if (!this._fanRO) {
        this._fanRO = new ResizeObserver(() => draw());
        this._fanRO.observe(host);
      }
    },

    // Render divergence heatmap (replaces horizontal-bar Chart.js).
    renderDivChart() {
      const host = this.$refs.divHost;
      if (!host || !this.result) return;
      const div = this.result.divergence || {};
      const raw = (div.variables || div.drivers || []).slice();
      if (!raw.length) { host.innerHTML = ""; return; }
      raw.sort((a, b) => Math.abs(b.sensitivity ?? 0) - Math.abs(a.sensitivity ?? 0));
      const top = raw.slice(0, 8);
      const draw = () => {
        const w = Math.max(280, host.clientWidth || 360);
        const h = Math.max(180, Math.min(320, top.length * 28 + 26));
        host.innerHTML = window.ClydeUI.divergenceHeatmap({
          vars: top.map((v) => ({
            name: v.name || v.variable || "?",
            sensitivity: v.sensitivity ?? 0,
            // If backend provides per-step sensitivity series, surface it; else
            // approximate a temporal ramp so the heatmap reads as a trend not a flat bar.
            series: Array.isArray(v.series) ? v.series : null,
          })),
          w, h,
        });
      };
      draw();
      if (!this._divRO) {
        this._divRO = new ResizeObserver(() => draw());
        this._divRO.observe(host);
      }
    },

    // Helper: backing values for the metric strip across the top of the
    // results pane — design's Simulation-screen "metric strip" pattern.
    metricStrip() {
      const cfg = this.result?.shock_config || {};
      const parse = this.result?.parse_result || {};
      const central = this.result?.paths?.central;
      const horizon = Array.isArray(central) ? central.length : (cfg.time_horizon ?? "—");
      const sigma = (() => {
        if (!Array.isArray(central) || !central.length) return "—";
        const m = this.selectedMetric;
        const xs = central.map((r) => r?.[m]).filter((v) => typeof v === "number");
        if (xs.length < 2) return "—";
        const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
        const variance = xs.reduce((s, v) => s + (v - mean) ** 2, 0) / xs.length;
        return Math.sqrt(variance).toFixed(3);
      })();
      const runs = this.result?.run_count || cfg.run_count || this.runCount;
      return [
        { l: "shock", v: String(cfg.shock_type || parse.shock_type || "—"), tone: "accent" },
        { l: "magnitude", v: this.fmtNum(cfg.severity ?? cfg.magnitude ?? parse.severity), tone: "warn" },
        { l: "scope", v: String(cfg.scope || cfg.sector || "—"), tone: "neutral" },
        { l: "runs", v: `${runs}/${runs}`, tone: "positive" },
        { l: "horizon", v: `${horizon} steps`, tone: "neutral" },
        { l: "σ", v: sigma, tone: "neutral" },
      ];
    },

    // Pipeline-rail steps for the design's right-rail step pattern.
    pipelineRailSteps() {
      const stage = (this.job?.progress?.stage || "").toLowerCase();
      const pct = this.progressPercent();
      const idxCur = PIPELINE_STAGES.findIndex((s) => stage.includes(s.key));
      return PIPELINE_STAGES.map((s, i) => {
        let status = "pending";
        if (idxCur < 0) status = i === 0 ? "running" : "pending";
        else if (i < idxCur) status = "done";
        else if (i === idxCur) status = (this.job?.status === "completed" ? "done" : "running");
        return {
          name: s.label,
          status,
          dur: status === "running" ? `${pct}%` : status === "done" ? "✓" : "—",
          note: status === "running" ? (this.job?.progress?.message || stage || "in progress") :
                status === "done" ? "complete" : "queued",
        };
      });
    },

    pipelineRailHTML() {
      return window.ClydeUI.stepRail({ steps: this.pipelineRailSteps() });
    },
  };
}

// expose globally for Alpine
window.clydeApp = clydeApp;
