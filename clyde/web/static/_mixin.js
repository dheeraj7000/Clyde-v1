/* ---- Network graph (click-to-inspect) + Simulation replay ---- */
(function () {
  const _origFactory = window.clydeApp;
  window.clydeApp = function () {
    const app = _origFactory();

    const NODE_COLORS = { household: "#34d399", firm: "#60a5fa", bank: "#fbbf24", central_bank: "#a78bfa" };
    const NODE_RADIUS = { household: 5, firm: 8, bank: 10, central_bank: 14 };
    const EDGE_COLORS = { employment: "#34d399", supply: "#60a5fa", interbank: "#fbbf24" };
    const PARAM_LABELS = {
      mpc: "Marginal propensity to consume", precautionary_savings_rate: "Precautionary savings rate",
      unemployment_fear_threshold: "Unemployment fear threshold", wage_demand_elasticity: "Wage demand elasticity",
      inflation_expectation_prior: "Inflation expectation prior", inflation_expectation_lr: "Inflation expectation LR",
      credit_seek_threshold: "Credit-seeking threshold", hurdle_rate: "Investment hurdle rate",
      hiring_elasticity: "Hiring elasticity", firing_threshold: "Firing threshold",
      cost_push_weight: "Cost-push pricing weight", demand_pull_weight: "Demand-pull pricing weight",
      supplier_switch_stress: "Supplier-switch stress", bankruptcy_threshold: "Bankruptcy threshold",
      investment_sensitivity: "Investment sensitivity", npl_tightening_elasticity: "NPL tightening elasticity",
      herding_weight: "Herding weight", reserve_threshold: "Reserve threshold",
      credit_approval_floor: "Credit approval floor", risk_appetite: "Risk appetite",
      taylor_inflation_weight: "Taylor: inflation weight", taylor_output_weight: "Taylor: output gap weight",
      rate_increment: "Policy rate increment", discretionary_band: "Discretionary band", neutral_rate: "Neutral rate",
    };

    /* ---- Node detail helpers ---- */
    app.nodeDisplayName = function (n) { return n ? n.id.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()) : ""; };
    app.nodeTypeLabel = function (t) { return { household: "Household", firm: "Firm", bank: "Bank", central_bank: "Central Bank" }[t] || t; };
    app.nodeTypeBadgeClass = function (t) {
      return { household: "bg-emerald-500/15 border-emerald-500/40 text-emerald-300", firm: "bg-blue-500/15 border-blue-500/40 text-blue-300",
        bank: "bg-amber-500/15 border-amber-500/40 text-amber-300", central_bank: "bg-purple-500/15 border-purple-500/40 text-purple-300" }[t] || "bg-slate-500/15 border-slate-500/40 text-slate-300";
    };
    app.paramLabel = function (k) { return PARAM_LABELS[k] || k.replace(/_/g, " "); };
    app.formatParamValue = function (v) { if (v == null) return "—"; if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4); return String(v); };
    app.connColor = function (t) { return EDGE_COLORS[t] || "#64748b"; };
    app.nodeConnections = function (node) {
      if (!node || !this.result?.network) return [];
      const conns = [], seen = new Set();
      for (const e of (this.result.network.edges || [])) {
        if (e.source === node.id) { const k = e.target+"o"+e.type; if (!seen.has(k)) { seen.add(k); conns.push({ peer: e.target, direction: "outbound", edgeType: e.type, relType: "", weight: e.weight }); } }
        else if (e.target === node.id) { const k = e.source+"i"+e.type; if (!seen.has(k)) { seen.add(k); conns.push({ peer: e.source, direction: "inbound", edgeType: e.type, relType: "", weight: e.weight }); } }
      }
      for (const c of conns) { const rel = (node.relationships || []).find(r => r.target_id === c.peer || r.source_id === c.peer); if (rel) c.relType = rel.rel_type || ""; }
      return conns;
    };
    app.nodeCausalEvents = function (node) {
      if (!node || !this.result?.causal_chains) return [];
      const evs = [];
      for (const c of this.result.causal_chains) for (const ev of (c.events || [])) if (ev.source_actor_id === node.id || ev.target_actor_id === node.id) evs.push(ev);
      evs.sort((a, b) => a.step - b.step); return evs;
    };
    app.selectNodeById = function (id) { const n = (this.result?.network?.nodes || []).find(n => n.id === id); if (n) this.selectedNode = n; };

    /* ---- D3 graph builder (shared) ---- */
    function buildGraph(svgEl, net, self, opts) {
      const d3sel = d3.select(svgEl); d3sel.selectAll("*").remove();
      const rect = svgEl.getBoundingClientRect();
      const W = rect.width || 700, H = rect.height || 540;
      const defs = d3sel.append("defs");
      ["glow","selglow","simglow"].forEach(id => { const f = defs.append("filter").attr("id", id); f.append("feGaussianBlur").attr("stdDeviation", id === "glow" ? "3" : "4").attr("result", "blur"); f.append("feMerge").selectAll("feMergeNode").data(["blur","SourceGraphic"]).enter().append("feMergeNode").attr("in", d => d); });
      const g = d3sel.append("g");
      d3sel.call(d3.zoom().scaleExtent([0.3, 5]).on("zoom", e => g.attr("transform", e.transform)));
      const nodes = net.nodes.map(n => ({ ...n }));
      const edgeSet = new Set(), links = [];
      for (const e of net.edges) { const k = e.source+"|"+e.target; if (!edgeSet.has(k)) { edgeSet.add(k); links.push({ ...e }); } }
      const sim = d3.forceSimulation(nodes).force("link", d3.forceLink(links).id(d => d.id).distance(opts.dist || 40).strength(0.3))
        .force("charge", d3.forceManyBody().strength(opts.charge || -80)).force("center", d3.forceCenter(W/2, H/2))
        .force("collision", d3.forceCollide().radius(d => (NODE_RADIUS[d.type]||6)+3));
      const linkSel = g.append("g").selectAll("line").data(links).enter().append("line")
        .attr("stroke", d => EDGE_COLORS[d.type]||"#334155").attr("stroke-opacity", 0.25).attr("stroke-width", 1);
      const nodeSel = g.append("g").selectAll("g").data(nodes).enter().append("g").attr("cursor","pointer")
        .call(d3.drag().on("start",(e,d)=>{if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y;})
          .on("drag",(e,d)=>{d.fx=e.x;d.fy=e.y;}).on("end",(e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}));
      if (opts.stressRing) nodeSel.append("circle").attr("class","stress-ring").attr("r",d=>(NODE_RADIUS[d.type]||6)+6).attr("fill","none").attr("stroke","#ef4444").attr("stroke-width",2.5).attr("opacity",0).attr("filter","url(#simglow)");
      nodeSel.append("circle").attr("class","sel-ring").attr("r",d=>(NODE_RADIUS[d.type]||6)+7).attr("fill","none").attr("stroke","#3b82f6").attr("stroke-width",2.5).attr("opacity",0).attr("filter","url(#selglow)");
      nodeSel.filter(d=>d.shocked).append("circle").attr("r",d=>(NODE_RADIUS[d.type]||6)+5).attr("fill","none").attr("stroke","#ef4444").attr("stroke-width",2).attr("opacity",0.6).attr("filter","url(#glow)");
      nodeSel.append("circle").attr("class","node-dot").attr("r",d=>NODE_RADIUS[d.type]||6)
        .attr("fill",d=>d.shocked?"#ef4444":(NODE_COLORS[d.type]||"#64748b")).attr("stroke",d=>d.shocked?"#fca5a5":"rgba(255,255,255,0.15)").attr("stroke-width",1);
      if (opts.labels) nodeSel.filter(d=>d.type!=="household").append("text")
        .text(d=>d.id.replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase()).replace(/\d{4}$/,m=>" "+m))
        .attr("dx",d=>(NODE_RADIUS[d.type]||6)+4).attr("dy",3).attr("fill","#94a3b8").attr("font-size","9px").attr("pointer-events","none");
      sim.on("tick",()=>{linkSel.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);nodeSel.attr("transform",d=>`translate(${d.x},${d.y})`);});
      return { d3sel, g, nodeSel, linkSel, sim, nodes, links };
    }

    /* ---- Network Graph tab ---- */
    app.renderNetworkGraph = function () {
      const svgEl = this.$refs.networkSvg; if (!svgEl || !this.result?.network) return;
      const net = this.result.network; if (!net.nodes.length) return;
      const self = this;
      const { d3sel, g, nodeSel, linkSel, sim } = buildGraph(svgEl, net, self, { dist: 40, charge: -80, labels: true });
      nodeSel.on("click", function (event, d) {
        event.stopPropagation();
        self.selectedNode = net.nodes.find(n => n.id === d.id) || d;
        g.selectAll(".sel-ring").transition().duration(200).attr("opacity", 0);
        d3.select(this).select(".sel-ring").transition().duration(200).attr("opacity", 0.8);
        linkSel.transition().duration(200)
          .attr("stroke-opacity", e => (e.source.id===d.id||e.target.id===d.id)?0.7:0.08)
          .attr("stroke-width", e => (e.source.id===d.id||e.target.id===d.id)?2.5:0.5)
          .attr("stroke", e => (e.source.id===d.id||e.target.id===d.id)?"#3b82f6":(EDGE_COLORS[e.type]||"#334155"));
      });
      d3sel.on("click", () => {
        self.selectedNode = null;
        g.selectAll(".sel-ring").transition().duration(200).attr("opacity", 0);
        linkSel.transition().duration(200).attr("stroke-opacity",0.25).attr("stroke-width",1).attr("stroke",d=>EDGE_COLORS[d.type]||"#334155");
      });
      this._networkSim = sim;
    };

    /* ---- Simulation Replay ---- */
    app.simMaxStep = function () { const c = this.result?.paths?.central; return Array.isArray(c)&&c.length ? c.length-1 : 0; };
    app.simMetricNames = function () { return CORE_METRICS.filter(m => m !== "interbank_freeze"); };
    app.simMetricValue = function (m) { const c = this.result?.paths?.central; if (!c||!c[this.simStep]) return "—"; const v = c[this.simStep][m]; if (v==null) return "—"; if (typeof v==="boolean") return v?"Yes":"No"; if (Number.isInteger(v)) return String(v); return Number(v).toFixed(4); };
    app.simMetricColor = function (m) {
      const c = this.result?.paths?.central; if (!c||c.length<2) return "text-slate-200";
      const cur = c[this.simStep]?.[m], prev = c[Math.max(0,this.simStep-1)]?.[m]; if (cur==null||prev==null) return "text-slate-200";
      const LB = new Set(["inflation_rate","unemployment_rate","gini_coefficient","credit_tightening_index","firm_bankruptcy_count","bank_stress_index"]);
      const d = Number(cur)-Number(prev); if (Math.abs(d)<1e-6) return "text-slate-200"; return (LB.has(m)?d<0:d>0)?"text-emerald-400":"text-red-400";
    };
    app.simEventsAtStep = function () { const evs=[]; for (const c of (this.result?.causal_chains||[])) for (const ev of (c.events||[])) if (ev.step===this.simStep) evs.push(ev); return evs; };
    app.simPrev = function () { if (this.simStep>0) { this.simStep--; this._simUpdateGraph(); } };
    app.simNext = function () { if (this.simStep<this.simMaxStep()) { this.simStep++; this._simUpdateGraph(); } };
    app.simPlay = function () {
      if (this.simPlaying) { this.simPlaying=false; clearInterval(this._simTimer); return; }
      if (this.simStep>=this.simMaxStep()) this.simStep=0;
      this.simPlaying=true;
      this._simTimer = setInterval(()=>{ if (this.simStep>=this.simMaxStep()){this.simPlaying=false;clearInterval(this._simTimer);return;} this.simStep++;this._simUpdateGraph(); }, this.simSpeed);
    };
    app.renderSimGraph = function () {
      const svgEl = this.$refs.simSvg; if (!svgEl||!this.result?.network) return;
      const net = this.result.network; if (!net.nodes.length) return;
      const self = this;
      const { d3sel, g, nodeSel, linkSel } = buildGraph(svgEl, net, self, { dist: 45, charge: -70, stressRing: true });
      nodeSel.on("click", function (event, d) {
        event.stopPropagation();
        self.simSelectedNode = net.nodes.find(n => n.id === d.id) || d;
        g.selectAll(".sel-ring").transition().duration(200).attr("opacity", 0);
        d3.select(this).select(".sel-ring").transition().duration(200).attr("opacity", 0.8);
      });
      d3sel.on("click", () => { self.simSelectedNode = null; g.selectAll(".sel-ring").transition().duration(200).attr("opacity", 0); });
      this._simD3 = { g, nodeSel, linkSel }; this.simStep = 0; this.simSelectedNode = null; this._simUpdateGraph();
    };
    app._simUpdateGraph = function () {
      if (!this._simD3||!this.result) return;
      const { nodeSel, linkSel } = this._simD3;
      const events = this.simEventsAtStep();
      const active = new Set(), aEdges = new Set();
      for (const ev of events) { active.add(ev.source_actor_id); active.add(ev.target_actor_id); aEdges.add(ev.source_actor_id+"|"+ev.target_actor_id); }
      const shocked = new Set(this.result.shock_config?.initial_contact_actors||[]);
      const stressed = new Set();
      for (const c of (this.result.causal_chains||[])) for (const ev of (c.events||[])) if (ev.step<=this.simStep) { stressed.add(ev.source_actor_id); stressed.add(ev.target_actor_id); }
      if (this.simStep>=0) for (const id of shocked) stressed.add(id);
      nodeSel.select(".node-dot").transition().duration(300)
        .attr("fill",d=>active.has(d.id)?"#ef4444":stressed.has(d.id)?"#f97316":NODE_COLORS[d.type]||"#64748b")
        .attr("r",d=>(NODE_RADIUS[d.type]||6)+(active.has(d.id)?3:0));
      nodeSel.select(".stress-ring").transition().duration(300).attr("opacity",d=>stressed.has(d.id)?0.5:0).attr("stroke",d=>active.has(d.id)?"#ef4444":"#f97316");
      const ek=d=>(typeof d.source==="object"?d.source.id:d.source)+"|"+(typeof d.target==="object"?d.target.id:d.target);
      linkSel.transition().duration(300).attr("stroke",d=>aEdges.has(ek(d))?"#ef4444":"#1e293b").attr("stroke-opacity",d=>aEdges.has(ek(d))?0.9:0.3).attr("stroke-width",d=>aEdges.has(ek(d))?2.5:1);
    };

    /* ---- Persona & Influence helpers ---- */
    const INFLUENCE_LABELS = {
      monetary_transmission_lag: "Monetary Transmission Lag",
      information_asymmetry: "Information Asymmetry",
      herding_strength: "Herding Strength",
      credit_channel_weight: "Credit Channel Weight",
      expectation_channel_weight: "Expectation Channel Weight",
      supply_chain_friction: "Supply Chain Friction",
      interbank_contagion: "Interbank Contagion",
      confidence_multiplier: "Confidence Multiplier",
    };
    const INFLUENCE_DESC = {
      monetary_transmission_lag: "How slowly central bank rate changes reach the real economy. Higher = slower transmission.",
      information_asymmetry: "How unevenly economic information spreads across actors. Higher = more asymmetric.",
      herding_strength: "How strongly actors copy peer behavior (bank herding, consumer panic buying).",
      credit_channel_weight: "Importance of the bank lending channel in transmitting shocks to firms.",
      expectation_channel_weight: "Importance of inflation expectations in driving household/firm decisions.",
      supply_chain_friction: "How much supply chain disruptions amplify upstream shocks to downstream firms.",
      interbank_contagion: "How fast stress spreads between banks through the interbank lending network.",
      confidence_multiplier: "How much consumer/business confidence amplifies or dampens economic effects.",
    };

    app.personaColor = function (type) { return NODE_COLORS[type] || "#64748b"; };
    app.filteredPersonas = function () {
      const ps = this.result?.personas || [];
      if (this.personaFilter === "all") return ps;
      return ps.filter(p => p.actor_type === this.personaFilter);
    };
    app.influenceLabel = function (key) { return INFLUENCE_LABELS[key] || key.replace(/_/g, " "); };
    app.influenceDescription = function (key) { return INFLUENCE_DESC[key] || ""; };

    /* ---- Lazy render on tab switch ---- */
    const origInit = app.init;
    app.init = async function () {
      await origInit.call(this);
      this.$watch("activeTab", val => {
        if (val === "network") this.$nextTick(() => this.renderNetworkGraph());
        if (val === "simulation") this.$nextTick(() => this.renderSimGraph());
      });
    };
    return app;
  };
})();
