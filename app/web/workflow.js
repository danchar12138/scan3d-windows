/* scan3d UI: the workflow of one task. A graph of its stages (every stage a node: state, on/off,
 * progress), the panel of the selected node (schema-generated parameter forms, models and
 * licences, run buttons, results), the "what will the next run do" line and the templates.
 *
 * The stages, their parameters and dependencies all come from GET /api/workflow/schema
 * (S3.schema); this file has no list of parameters of its own. What the next click will do is
 * asked of the server (POST /api/jobs/<job>/plan), never worked out here.
 *
 * app.js calls  S3.wf.update(job)  with every fresh view of the job and  S3.wf.reset()  when the
 * job page is left. Everything put on the page from data goes through textContent. */
(function () {
  "use strict";

  var S3 = window.S3;
  var $ = S3.$, el = S3.el, setText = S3.setText;

  var NODE_H = 98, GAP_X = 20, GAP_Y = 14, MIN_W = 100, MAX_W = 158, PAD = 6;
  // Which row of the graph a stage sits in (columns come from the dependencies). A stage that is
  // not named here goes into the row of its first dependency.
  var LANE = { normals: 1, dense_seed: 2, dense: 3, mesh: 3 };
  // What the result viewer (jobview.js) can show for a stage, and the output files each view needs.
  var VIEWS = {
    train: [["splat", "看高斯"], ["overlay", "叠加对齐"]],
    mesh: [["mesh", "看网格"], ["collision", "看碰撞网格"]],
    coverage: [["coverage", "看拍摄覆盖"]],
    export: [["report", "看报告"]]
  };
  var VIEW_NEEDS = {
    splat: ["splat"], overlay: ["splat", "collision"], mesh: ["mesh"], collision: ["collision"],
    coverage: ["splat"], report: ["report", "report.json"]
  };
  var REASON = { requested: "", needed: "", forced: "重做", downstream: "跟着重做" };

  var ctx = null;

  function freshContext() {
    return {
      job: null, name: null, selected: null,
      nodes: {}, edges: [], built: false, mode: "graph", width: 0,
      plans: {}, planSig: "",
      effect: null, saving: 0, chain: Promise.resolve(),
      panel: null, tplOpen: false, tplBusy: false, tplMessage: ""
    };
  }
  ctx = freshContext();

  // ------------------------------------------------------------------ schema helpers

  function stageList() { return (S3.schema && S3.schema.stages) || []; }

  function def(id) {
    var list = stageList();
    for (var i = 0; i < list.length; i++) { if (list[i].id === id) { return list[i]; } }
    return null;
  }

  function wfStage(job, id) {
    return job && job.workflow && job.workflow.stages ? job.workflow.stages[id] : null;
  }

  function busy(job) { return !!job && (job.state === "queued" || job.state === "running"); }

  // The stages that will really run: switched on, not optional, or needed by one of those.
  // Mirrors workflow.effective_enabled (only the dependency rule; what runs is the server's plan).
  function effectiveSet(job) {
    var on = {};
    stageList().forEach(function (s) {
      var w = wfStage(job, s.id);
      if (!s.optional || (w && w.enabled)) { on[s.id] = true; }
    });
    var changed = true;
    while (changed) {
      changed = false;
      Object.keys(on).forEach(function (id) {
        (def(id).deps || []).forEach(function (dep) { if (!on[dep]) { on[dep] = true; changed = true; } });
      });
    }
    return on;
  }

  // The stages that are on by themselves and need `id` (directly or through other dependencies).
  function neededBy(job, id) {
    var out = [];
    stageList().forEach(function (s) {
      var w = wfStage(job, s.id);
      if (s.id === id || !(!s.optional || (w && w.enabled))) { return; }
      var seen = {}, todo = (s.deps || []).slice();
      while (todo.length) {
        var d = todo.pop();
        if (seen[d]) { continue; }
        seen[d] = true;
        (def(d).deps || []).forEach(function (x) { todo.push(x); });
      }
      if (seen[id]) { out.push(s.id); }
    });
    return out;
  }

  function stageInfo(job, id) {
    var s = job && job.stages && job.stages[id];
    return s || { state: "pending", progress: null, message: null, seconds: null, error: null, summary: null };
  }

  function stateOf(job, id, eff) {
    var s = job && job.stages && job.stages[id];
    if (s && s.state) { return s.state; }
    return eff && eff[id] ? "pending" : "disabled";
  }

  function statesOf(job) {
    var out = {};
    stageList().forEach(function (s) { out[s.id] = stateOf(job, s.id, null); });
    return out;
  }

  function hasOutput(job, key) {
    return (job.output_files || []).some(function (f) { return f.key === key || f.name === key; });
  }

  function templateLabel(id) {
    var info = S3.templatesInfo, list = (info && info.templates) || (S3.schema && S3.schema.templates) || [];
    for (var i = 0; i < list.length; i++) { if (list[i].id === id) { return list[i].label; } }
    return id || "";
  }

  // ------------------------------------------------------------------ small facts of a stage

  function factText(key, value) {
    var text;
    if (typeof value === "number") { text = S3.fmtCount(value); }
    else if (typeof value === "boolean") { text = value ? "是" : "否"; }
    else if (typeof value === "string") { text = S3.SUMMARY_VALUE[value] || value; }
    else { return null; }
    return { label: S3.SUMMARY_LABEL[key] || key, value: text };
  }

  function facts(summary) {
    var out = [];
    Object.keys(summary || {}).forEach(function (key) {
      var f = factText(key, summary[key]);
      if (f) { out.push(f); }
    });
    return out;
  }

  function shortFacts(summary, max) {
    return facts(summary).slice(0, max || 2).map(function (f) { return f.label + " " + f.value; }).join(" · ");
  }

  // ------------------------------------------------------------------ layout

  function computeLayout() {
    var list = stageList(), layer = {}, row = {}, taken = {}, cols = 0, rows = 0;
    list.forEach(function (s) {
      var l = 0;
      (s.deps || []).concat(s.soft || []).forEach(function (d) { l = Math.max(l, (layer[d] || 0) + 1); });
      layer[s.id] = l;
      cols = Math.max(cols, l + 1);
    });
    list.forEach(function (s) {
      var r = LANE.hasOwnProperty(s.id) ? LANE[s.id] : (s.deps && s.deps.length ? row[s.deps[0]] || 0 : 0);
      while (taken[layer[s.id] + ":" + r]) { r++; }
      taken[layer[s.id] + ":" + r] = true;
      row[s.id] = r;
      rows = Math.max(rows, r + 1);
    });
    return { layer: layer, row: row, cols: cols, rows: rows };
  }

  // Every dependency edge d -> s, minus the ones another path already implies.
  function reduceEdges(edges) {
    function reaches(from, to, skip) {
      var seen = {}, todo = [from];
      while (todo.length) {
        var cur = todo.pop();
        if (cur === to) { return true; }
        if (seen[cur]) { continue; }
        seen[cur] = true;
        edges.forEach(function (e) {
          if (e !== skip && e.from === cur) { todo.push(e.to); }
        });
      }
      return false;
    }
    return edges.filter(function (e) {
      var others = edges.filter(function (x) { return x !== e && x.from === e.from; });
      return !others.some(function (x) { return reaches(x.to, e.to, null); });
    });
  }

  function candidateEdges(only) {
    var out = [];
    stageList().forEach(function (s) {
      (s.deps || []).concat(s.soft || []).forEach(function (d) {
        if (!only || (only[d] && only[s.id])) { out.push({ from: d, to: s.id }); }
      });
    });
    return out;
  }

  // ------------------------------------------------------------------ building the graph

  function buildGraph() {
    var graph = $("wf-graph"), svg = $("wf-edges");
    graph.querySelectorAll(".node").forEach(function (n) { n.remove(); });
    svg.querySelectorAll("path").forEach(function (p) { p.remove(); });
    ctx.nodes = {};
    ctx.edges = [];
    stageList().forEach(function (s) {
      var root = el("div", "node");
      root.setAttribute("data-stage", s.id);
      var main = el("button", "node-main");
      main.type = "button";
      main.setAttribute("aria-controls", "np");
      main.setAttribute("aria-expanded", "false");
      var label = el("span", "node-label", s.label);
      var state = el("span", "node-state");
      var ico = S3.icon("pending", "node-ico");
      var word = el("span", "node-word");
      state.appendChild(ico);
      state.appendChild(word);
      var sum = el("span", "node-sum");
      var track = el("span", "node-bar");
      var fill = el("span", "node-fill");
      track.appendChild(fill);
      var needs = el("span", "node-needs");
      main.appendChild(label);
      main.appendChild(state);
      main.appendChild(sum);
      main.appendChild(track);
      main.appendChild(needs);
      root.appendChild(main);

      var sw = el("button", "switch");
      sw.type = "button";
      sw.setAttribute("role", "switch");
      var swIco = S3.icon("lock", "switch-lock");
      sw.appendChild(el("span", "switch-knob"));
      sw.appendChild(swIco);
      root.appendChild(sw);

      if (s.gpu) { root.appendChild(el("span", "node-gpu", "GPU")); }
      graph.appendChild(root);

      main.addEventListener("click", function () { wf.select(ctx.selected === s.id ? null : s.id); });
      sw.addEventListener("click", function () { toggleStage(s.id); });
      ctx.nodes[s.id] = { root: root, main: main, label: label, ico: ico, word: word, sum: sum,
                          track: track, fill: fill, sw: sw, swIco: swIco, needs: needs, state: null };
    });

    var all = candidateEdges(null);
    var kept = reduceEdges(all);
    kept.forEach(function (e) {
      var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("class", "edge");
      svg.appendChild(path);
      e.path = path;
      ctx.edges.push(e);
    });
    // The edges that exist only while a stage in between is switched off (ingest -> sfm when
    // there is no masks stage) are drawn when needed; made here so they can just be shown.
    var kept2 = {};
    kept.forEach(function (e) { kept2[e.from + ">" + e.to] = true; });
    all.forEach(function (e) {
      if (kept2[e.from + ">" + e.to]) { return; }
      var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("class", "edge is-hidden");
      svg.appendChild(path);
      e.path = path;
      e.extra = true;
      ctx.edges.push(e);
    });
    ctx.built = true;
    ctx.width = 0;
  }

  function layoutGraph() {
    var wrap = $("wf-wrap"), graph = $("wf-graph"), svg = $("wf-edges");
    var avail = wrap.clientWidth;
    if (!avail) { return; }
    var lay = computeLayout();
    var w = Math.floor((avail - (lay.cols - 1) * GAP_X - 2 * PAD) / lay.cols);
    var mode = w < MIN_W ? "list" : "graph";
    if (mode !== ctx.mode || ctx.width !== avail) {
      ctx.mode = mode;
      ctx.width = avail;
      graph.classList.toggle("is-list", mode === "list");
      ctx.lay = lay;
      if (mode === "graph") {
        w = Math.min(w, MAX_W);
        var totalW = lay.cols * w + (lay.cols - 1) * GAP_X + 2 * PAD;
        var totalH = lay.rows * NODE_H + (lay.rows - 1) * GAP_Y + 2 * PAD;
        graph.style.width = totalW + "px";
        graph.style.height = totalH + "px";
        svg.setAttribute("width", totalW);
        svg.setAttribute("height", totalH);
        svg.setAttribute("viewBox", "0 0 " + totalW + " " + totalH);
        ctx.geom = {};
        stageList().forEach(function (s) {
          var n = ctx.nodes[s.id];
          var x = PAD + lay.layer[s.id] * (w + GAP_X), y = PAD + lay.row[s.id] * (NODE_H + GAP_Y);
          n.root.style.left = x + "px";
          n.root.style.top = y + "px";
          n.root.style.width = w + "px";
          n.root.style.height = NODE_H + "px";
          ctx.geom[s.id] = { x: x, y: y, w: w };
        });
        ctx.edges.forEach(function (e) {
          var a = ctx.geom[e.from], b = ctx.geom[e.to];
          var x1 = a.x + a.w, y1 = a.y + NODE_H / 2, x2 = b.x, y2 = b.y + NODE_H / 2;
          var dx = Math.max(18, (x2 - x1) * 0.45);
          var d = Math.abs(y1 - y2) < 1
            ? "M" + x1 + " " + y1 + "L" + x2 + " " + y2
            : "M" + x1 + " " + y1 + "C" + (x1 + dx) + " " + y1 + " " + (x2 - dx) + " " + y2 + " " + x2 + " " + y2;
          d += "M" + (x2 - 6) + " " + (y2 - 4) + "L" + x2 + " " + y2 + "L" + (x2 - 6) + " " + (y2 + 4);
          e.path.setAttribute("d", d);
        });
      } else {
        graph.style.width = "";
        graph.style.height = "";
        stageList().forEach(function (s) {
          var st = ctx.nodes[s.id].root.style;
          st.left = ""; st.top = ""; st.width = ""; st.height = "";
        });
      }
      placePanel();
    }
  }

  // ------------------------------------------------------------------ drawing the state

  function nodeSummary(job, s, st, info, neededList) {
    var text = "";
    if (st === "running") { text = info.message || ""; }
    else if (st === "done") {
      text = shortFacts(info.summary, 1);
      var t = S3.fmtSpent(info.seconds);
      if (t) { text += (text ? " · " : "") + t; }
    }
    else if (st === "stale") { text = "设置或上游改过"; }
    else if (st === "failed") { text = S3.firstLine(info.error || info.message) || "出错了"; }
    else if (st === "queued") { text = "等前面的阶段"; }
    else if (st === "pending") {
      text = neededList && neededList.length ? "被「" + S3.stageLabel(neededList[0]) + "」需要" : "已打开，等运行";
    }
    else if (st === "disabled") { text = s.optional ? "用开关打开" : ""; }
    return text;
  }

  function updateNodes() {
    var job = ctx.job, eff = effectiveSet(job), locked = busy(job);
    stageList().forEach(function (s) {
      var n = ctx.nodes[s.id];
      if (!n) { return; }
      var info = stageInfo(job, s.id), st = stateOf(job, s.id, eff);
      var w = wfStage(job, s.id) || { enabled: false };
      var needed = !s.optional || !w.enabled ? neededBy(job, s.id) : [];
      var forced = eff[s.id] && s.optional && !w.enabled;       // on only because another stage needs it
      var required = !s.optional;
      var cls = "node s-" + st + (s.gpu ? " has-gpu" : "") + (ctx.selected === s.id ? " is-selected" : "") + (ctx.saving ? " is-saving" : "");
      S3.setClass(n.root, cls);
      var meta = S3.STAGE_STATE[st] || S3.STAGE_STATE.pending;
      S3.setIcon(n.ico, meta.icon);
      var pct = typeof info.progress === "number" ? Math.round(100 * S3.clamp01(info.progress)) : null;
      var word = meta.word + (st === "running" && pct !== null ? " " + pct + "%" : "");
      setText(n.word, word);
      var sum = nodeSummary(job, s, st, info, forced ? needed : null);
      setText(n.sum, sum);
      n.main.title = (sum ? sum + "。" : "") + meta.help;
      // progress bar
      var showBar = st === "running";
      n.track.hidden = !showBar;
      S3.setClass(n.track, "node-bar" + (showBar && pct === null ? " is-indeterminate" : ""));
      n.fill.style.width = showBar && pct !== null ? pct + "%" : "";
      // what this stage needs, for the list layout
      var needs = (s.deps || []).length ? "接在「" + (s.deps || []).map(S3.stageLabel).join("、") + "」之后" : "";
      setText(n.needs, needs);
      n.main.setAttribute("aria-expanded", String(ctx.selected === s.id));
      n.main.setAttribute("aria-label", s.label + "，" + word + (sum ? "，" + sum : "") + (s.gpu ? "，用显卡" : ""));

      // the switch
      var on = required || w.enabled || forced;
      var lockedOn = required || forced;
      n.sw.setAttribute("aria-checked", String(!!on));
      n.sw.setAttribute("aria-disabled", String(lockedOn || locked));
      n.sw.disabled = false;
      n.sw.classList.toggle("is-on", !!on);
      n.sw.classList.toggle("is-locked", !!lockedOn);
      n.sw.classList.toggle("is-busy", locked && !lockedOn);
      var why = required ? "必需的阶段，不能关闭"
        : forced ? "「" + S3.joinLabels(needed) + "」需要它，所以一定会运行"
        : locked ? "任务正在运行，现在不能改开关"
        : on ? "点一下关闭" : "点一下打开";
      n.sw.title = s.label + "：" + why;
      n.sw.setAttribute("aria-label", "运行「" + s.label + "」：" + (on ? "已打开" : "已关闭") + (lockedOn ? "（锁定：" + why + "）" : ""));
      n.state = st;
    });
  }

  function updateEdges() {
    var job = ctx.job, eff = effectiveSet(job);
    var active = candidateEdges(eff);
    var activeKept = reduceEdges(active), activeKey = {};
    activeKept.forEach(function (e) { activeKey[e.from + ">" + e.to] = true; });
    var universe = {};
    ctx.edges.forEach(function (e) { if (!e.extra) { universe[e.from + ">" + e.to] = true; } });
    ctx.edges.forEach(function (e) {
      var key = e.from + ">" + e.to, solid = !!activeKey[key];
      var inUniverse = !!universe[key];
      var show = solid || (inUniverse && !(eff[e.from] && eff[e.to]));
      var from = stateOf(job, e.from, eff), to = stateOf(job, e.to, eff);
      var cls = "edge" + (show ? "" : " is-hidden") + (solid ? "" : " is-off");
      if (solid && to === "running") { cls += " is-flow"; }
      else if (solid && (from === "stale" || from === "failed")) { cls += " is-stale"; }
      else if (solid && from === "done") { cls += " is-done"; }
      e.path.setAttribute("class", cls);
    });
  }

  function placePanel() {
    var panel = $("np");
    if (!ctx.selected || panel.hidden) { return; }
    var home = $("np-home");
    if (ctx.mode === "list") {
      var n = ctx.nodes[ctx.selected];
      if (n && n.root.nextSibling !== panel) { n.root.after(panel); }
      panel.classList.add("in-list");
    } else {
      if (home.nextSibling !== panel) { home.after(panel); }
      panel.classList.remove("in-list");
    }
  }

  // ------------------------------------------------------------------ plans (what the next click does)

  var PLAN_KINDS = {
    all: function () { return {}; },
    to: function (id) { return { to: id }; },
    only: function (id) { return { only: [id] }; },
    force: function (id) { return { to: id, force: [id] }; }
  };

  function planSignature(job) {
    var states = {};
    Object.keys(job.stages || {}).forEach(function (id) { states[id] = job.stages[id].state; });
    return JSON.stringify([job.job, job.state, job.workflow, states]);
  }

  // The answer for `body`, from the cache; asks the server once when there is none yet.
  function planFor(body) {
    var key = JSON.stringify(body), sig = ctx.planSig, entry = ctx.plans[key];
    if (entry) { return entry; }
    entry = ctx.plans[key] = { loading: true };
    var jobName = ctx.name;
    S3.api("POST", S3.jobUrl(jobName, "/plan"), body).then(function (res) {
      return { ok: true, steps: res.steps || [], licences: res.licences || [], models: res.models_needed || [] };
    }, function (err) {
      return { ok: false, code: err.code || (err.data && err.data.error) || "", message: err.message };
    }).then(function (result) {
      if (ctx.name !== jobName || ctx.planSig !== sig) { return; }
      ctx.plans[key] = result;
      renderPlans();
    });
    return entry;
  }

  function stepChips(steps, container) {
    container.textContent = "";
    steps.forEach(function (step, i) {
      if (i) { container.appendChild(el("span", "chip-arrow", "›")); }
      var tag = REASON[step.reason] || "";
      var chip = el("span", "chip" + (tag ? " is-redo" : ""), step.label || S3.stageLabel(step.stage));
      if (tag) { chip.appendChild(el("small", null, tag)); }
      container.appendChild(chip);
    });
  }

  function sumMb(models) {
    var total = 0;
    (models || []).forEach(function (m) { total += typeof m.mb === "number" ? m.mb : 0; });
    return total;
  }

  function extrasText(plan) {
    var out = [];
    var mb = sumMb(plan.models);
    if (plan.models && plan.models.length) {
      var names = {};
      plan.models.forEach(function (m) { names[S3.stageLabel(m.stage)] = true; });
      out.push("第一次用到的模型会自动下载，约 " + S3.fmtMb(mb) + "（" + Object.keys(names).join("、") + "）");
    }
    return out;
  }


  function renderNext() {
    var box = $("d-next-body"), job = ctx.job;
    if (!job) { return; }
    var chips = $("d-next-chips"), extra = $("d-next-extra"), title = $("d-next-title");
    extra.textContent = "";
    chips.textContent = "";
    $("d-next").classList.remove("is-warn");
    if (busy(job)) {
      var plan = (job.run && job.run.plan) || [];
      var done = plan.filter(function (id) { return stateOf(job, id, null) === "done"; }).length;
      if (job.state === "queued") { setText(title, "排队中，轮到它时会按计划运行"); }
      else {
        var cur = job.stage ? "现在是「" + S3.stageLabel(job.stage) + "」" : "";
        setText(title, "正在运行：计划共 " + plan.length + " 步，已完成 " + done + " 步" + (cur ? "，" + cur : ""));
      }
      plan.forEach(function (id, i) {
        if (i) { chips.appendChild(el("span", "chip-arrow", "›")); }
        var st = stateOf(job, id, null);
        var chip = el("span", "chip s-" + st);
        chip.appendChild(S3.icon((S3.STAGE_STATE[st] || S3.STAGE_STATE.pending).icon, "chip-ico"));
        chip.appendChild(document.createTextNode(S3.stageLabel(id)));
        chips.appendChild(chip);
      });
      return;
    }
    var result = planFor(PLAN_KINDS.all());
    if (result.loading) { setText(title, "正在算下一次会做什么…"); return; }
    if (!result.ok) {
      if (result.code === "nothing_to_do") {
        setText(title, "打开的阶段都已完成，没有要运行的。想重做某一步，选中它再点「重新运行」。");
      } else {
        $("d-next").classList.add("is-warn");
        setText(title, result.message || "现在无法运行");
      }
      return;
    }
    setText(title, "点「运行」会做 " + result.steps.length + " 步：");
    stepChips(result.steps, chips);
    extrasText(result).forEach(function (t) { extra.appendChild(el("p", "next-note", t)); });
    if (ctx.effect && ctx.effect.text) { extra.appendChild(el("p", "next-effect", ctx.effect.text)); }
  }

  function renderPlans() {
    renderNext();
    if (ctx.panel) { renderActionHints(); }
    updateRunButton();
  }

  function updateRunButton() {
    var btn = $("btn-job-start");
    var job = ctx.job;
    if (!job) { return; }
    var result = busy(job) ? null : planFor(PLAN_KINDS.all());
    var blocked = !busy(job) && result && !result.loading && !result.ok;
    btn.disabled = busy(job) || !!blocked;
    btn.title = busy(job) ? "正在运行" : blocked ? result.message : "";
  }

  // ------------------------------------------------------------------ changing the workflow

  function describeEffect(before, after, what) {
    var stale = [], added = [], removed = [];
    stageList().forEach(function (s) {
      var a = before[s.id], b = after[s.id];
      if (a === b) { return; }
      if (b === "stale") { stale.push(s.id); }
      else if (a === "disabled" && b !== "disabled") { added.push(s.id); }
      else if (b === "disabled" && a !== "disabled") { removed.push(s.id); }
      else if (a === "done" && b === "pending") { stale.push(s.id); }
    });
    var bits = [];
    if (added.length) { bits.push("加入：" + S3.joinLabels(added)); }
    if (removed.length) { bits.push("不再运行：" + S3.joinLabels(removed)); }
    if (stale.length) { bits.push("需要重算：" + S3.joinLabels(stale)); }
    if (!bits.length) { return { text: what + "：不影响已有的结果。", at: Date.now() }; }
    return { text: what + " → " + bits.join("；"), at: Date.now() };
  }

  // Send a partial workflow override ({stage: {enabled?, params?}}, optionally template). One at
  // a time, in order.
  function sendChange(body, what) {
    var job = ctx.job;
    if (!job || busy(job)) { return Promise.resolve(); }
    var name = job.job;
    ctx.saving++;
    updateNodes();
    var run = ctx.chain.then(function () {
      var before = statesOf(ctx.job || job);
      return S3.api("PUT", S3.jobUrl(name, "/workflow"), body).then(function (next) {
        if (ctx.name !== name) { return; }
        ctx.effect = describeEffect(before, statesOf(next), what);
        S3.showError("d-action-error", "");
        S3.app.applyJob(next);
      });
    }).catch(function (err) {
      S3.showError("d-action-error", err.message);
      S3.app.refreshSoon();
    }).then(function () { ctx.saving--; if (ctx.job) { updateNodes(); syncForm(true); } });
    ctx.chain = run;
    return run;
  }

  function toggleStage(id) {
    var job = ctx.job, n = ctx.nodes[id], s = def(id);
    if (!job || !n || busy(job)) { return; }
    var w = wfStage(job, id) || {};
    if (!s.optional || n.sw.classList.contains("is-locked")) {
      ctx.effect = { text: !s.optional ? "「" + s.label + "」是必需的阶段，不能关闭。"
        : "「" + s.label + "」被「" + S3.joinLabels(neededBy(job, id)) + "」需要，要关掉它得先关掉那些。", at: Date.now() };
      renderNext();
      return;
    }
    var body = {};
    body[id] = { enabled: !w.enabled };
    sendChange(body, (w.enabled ? "关闭「" : "打开「") + s.label + "」");
  }

  // ------------------------------------------------------------------ the node panel

  function fieldId(sid, key) { return "p-" + sid + "-" + key; }

  function paramStep(p) {
    if (p.type === "int") { return 1; }
    var span = (p.max !== undefined && p.min !== undefined) ? p.max - p.min : 10;
    return span <= 2 ? 0.05 : span <= 100 ? 0.1 : 1;
  }

  function buildField(sid, p, refs) {
    var row = el("div", "field-row");
    var id = fieldId(sid, p.key);
    var ref = { param: p, row: row, timer: null };
    var errorLine = el("p", "field-error");
    errorLine.id = id + "-err";
    errorLine.hidden = true;

    function commit(value) {
      var current = wfStage(ctx.job, sid);
      if (current && current.params && current.params[p.key] === value) { return; }      // nothing changed
      var body = {};
      body[sid] = { params: {} };
      body[sid].params[p.key] = value;
      sendChange(body, "改了「" + def(sid).label + "·" + p.label + "」");
    }
    function fail(text) {
      errorLine.textContent = text;
      errorLine.hidden = !text;
      if (ref.input) { ref.input.setAttribute("aria-invalid", text ? "true" : "false"); }
    }

    if (p.type === "bool") {
      var label = el("label", "check");
      var box = document.createElement("input");
      box.type = "checkbox";
      box.id = id;
      label.appendChild(box);
      label.appendChild(el("span", null, p.label));
      row.appendChild(label);
      ref.input = box;
      ref.get = function () { return box.checked; };
      ref.set = function (v) { box.checked = !!v; };
      box.addEventListener("change", function () { commit(box.checked); });
    } else {
      var lab = el("label", "field-label", p.label);
      lab.setAttribute("for", id);
      row.appendChild(lab);
      if (p.type === "choice") {
        var choices = p.choices || [];
        if (choices.length <= 3) {
          var group = el("div", "seg");
          group.setAttribute("role", "radiogroup");
          group.setAttribute("aria-label", p.label);
          var radios = [];
          choices.forEach(function (c, i) {
            var l = el("label");
            var r = document.createElement("input");
            r.type = "radio";
            r.name = id;
            r.value = String(i);
            l.appendChild(r);
            l.appendChild(el("span", null, c.label));
            group.appendChild(l);
            radios.push(r);
            r.addEventListener("change", function () { if (r.checked) { commit(c.value); } });
          });
          row.appendChild(group);
          ref.input = radios[0];
          ref.get = function () {
            for (var i = 0; i < radios.length; i++) { if (radios[i].checked) { return choices[i].value; } }
            return null;
          };
          ref.set = function (v) {
            for (var i = 0; i < choices.length; i++) { radios[i].checked = choices[i].value === v; }
          };
          ref.focusable = radios;
        } else {
          var select = document.createElement("select");
          select.id = id;
          choices.forEach(function (c, i) {
            var o = el("option", null, c.label);
            o.value = String(i);
            select.appendChild(o);
          });
          row.appendChild(select);
          ref.input = select;
          ref.get = function () { return choices[select.selectedIndex] ? choices[select.selectedIndex].value : null; };
          ref.set = function (v) {
            for (var i = 0; i < choices.length; i++) { if (choices[i].value === v) { select.selectedIndex = i; } }
          };
          select.addEventListener("change", function () { commit(ref.get()); });
        }
      } else if (p.type === "int" || p.type === "float") {
        var wrap = el("div", "num");
        var input = document.createElement("input");
        input.type = "number";
        input.id = id;
        input.inputMode = p.type === "int" ? "numeric" : "decimal";
        var step = paramStep(p);
        input.step = String(step);
        if (p.min !== undefined) { input.min = String(p.min); }
        if (p.max !== undefined) { input.max = String(p.max); }
        wrap.appendChild(input);
        var slider = null;
        if (p.min !== undefined && p.max !== undefined && p.max - p.min <= 100) {
          slider = document.createElement("input");
          slider.type = "range";
          slider.min = String(p.min);
          slider.max = String(p.max);
          slider.step = String(step);
          slider.setAttribute("aria-label", p.label + "（滑块）");
          wrap.insertBefore(slider, input);
        }
        if (p.min !== undefined && p.max !== undefined) {
          wrap.appendChild(el("span", "num-range", p.min + " – " + p.max));
        }
        row.appendChild(wrap);
        ref.input = input;
        ref.get = function () { return input.value === "" ? NaN : Number(input.value); };
        ref.set = function (v) { input.value = String(v); if (slider) { slider.value = String(v); } };
        function validate() {
          var v = ref.get();
          if (!isFinite(v)) { fail("要填一个数字"); return null; }
          if (p.type === "int" && v !== Math.round(v)) { fail("要填整数"); return null; }
          if ((p.min !== undefined && v < p.min) || (p.max !== undefined && v > p.max)) {
            fail("要在 " + p.min + " 到 " + p.max + " 之间");
            return null;
          }
          fail("");
          return v;
        }
        input.addEventListener("input", function () {
          if (slider && isFinite(ref.get())) { slider.value = input.value; }
          clearTimeout(ref.timer);
          ref.timer = setTimeout(function () { ref.timer = null; var v = validate(); if (v !== null) { commit(v); } }, 700);
        });
        input.addEventListener("change", function () {
          clearTimeout(ref.timer);
          ref.timer = null;
          var v = validate();
          if (v !== null) { commit(v); }
        });
        if (slider) {
          slider.addEventListener("input", function () { input.value = slider.value; fail(""); });
          slider.addEventListener("change", function () { var v = validate(); if (v !== null) { commit(v); } });
        }
      } else {
        var text = document.createElement("input");
        text.type = "text";
        text.id = id;
        text.autocomplete = "off";
        text.spellcheck = false;
        row.appendChild(text);
        ref.input = text;
        ref.get = function () { return text.value; };
        ref.set = function (v) { text.value = String(v); };
        text.addEventListener("input", function () {
          clearTimeout(ref.timer);
          ref.timer = setTimeout(function () { ref.timer = null; commit(text.value); }, 800);
        });
        text.addEventListener("change", function () { clearTimeout(ref.timer); ref.timer = null; commit(text.value); });
      }
    }
    if (p.help) {
      var help = el("p", "hint field-help", p.help);
      help.id = id + "-help";
      row.appendChild(help);
      if (ref.input) { ref.input.setAttribute("aria-describedby", help.id + " " + errorLine.id); }
    }
    row.appendChild(errorLine);
    refs[p.key] = ref;
    return row;
  }

  function isEditing(ref) {
    if (ref.timer) { return true; }
    var active = document.activeElement;
    return !!active && ref.row.contains(active) && active.tagName !== "BUTTON";
  }

  // Put the server's values into the form, except where the user is typing.
  function syncForm(force) {
    var P = ctx.panel, job = ctx.job;
    if (!P || !job) { return; }
    var w = wfStage(job, P.sid);
    if (!w) { return; }
    var locked = busy(job) || ctx.saving > 0;
    Object.keys(P.fields).forEach(function (key) {
      var ref = P.fields[key], value = w.params ? w.params[key] : undefined;
      if (value !== undefined && (force || !isEditing(ref))) {
        ref.set(value);
        var err = ref.row.querySelector(".field-error");
        if (err && !isEditing(ref)) { err.hidden = true; err.textContent = ""; }
      }
      var inputs = ref.focusable || [ref.input];
      inputs.forEach(function (i) { if (i) { i.disabled = busy(job); } });
      var slider = ref.row.querySelector('input[type="range"]');
      if (slider) { slider.disabled = busy(job); }
    });
    P.formNote.hidden = !busy(job);
    if (locked && !busy(job)) { P.formNote.hidden = true; }
  }

  function actionTile(label, hint, cls) {
    var tile = el("div", "action-tile");
    var button = el("button", "btn small" + (cls ? " " + cls : ""), label);
    button.type = "button";
    var line = el("p", "hint action-hint");
    line.id = "ah-" + Math.random().toString(36).slice(2, 8);
    button.setAttribute("aria-describedby", line.id);
    tile.appendChild(button);
    tile.appendChild(line);
    return { tile: tile, button: button, hint: line };
  }

  function buildPanel(sid) {
    var s = def(sid), panel = $("np");
    panel.textContent = "";
    var P = { sid: sid, fields: {}, tiles: {} };

    var head = el("div", "np-head");
    var title = el("div", "np-title");
    P.ico = S3.icon("pending", "np-ico");
    title.appendChild(P.ico);
    var h = el("h2", null, s.label);
    h.id = "np-title";
    title.appendChild(h);
    P.badge = el("span", "badge");
    title.appendChild(P.badge);
    if (s.gpu) { title.appendChild(el("span", "tag", "用显卡")); }
    head.appendChild(title);
    var close = el("button", "btn small quiet", "收起");
    close.type = "button";
    close.addEventListener("click", function () { wf.select(null); });
    head.appendChild(close);
    panel.appendChild(head);
    panel.setAttribute("aria-labelledby", "np-title");
    panel.appendChild(el("p", "np-desc", s.desc));

    P.status = el("div", "np-status");
    panel.appendChild(P.status);

    // on / off
    P.enable = el("div", "np-enable");
    panel.appendChild(P.enable);

    // run buttons
    var actions = el("div", "np-actions");
    var to = actionTile("运行到这里", "", "primary");
    var only = actionTile("只运行这一步", "", "");
    var force = actionTile("重新运行", "", "");
    var reset = actionTile("清除结果", "", "quiet-danger");
    P.tiles = { to: to, only: only, force: force, reset: reset };
    [to, only, force, reset].forEach(function (t) { actions.appendChild(t.tile); });
    panel.appendChild(actions);
    to.button.addEventListener("click", function () { wf.run(PLAN_KINDS.to(sid)); });
    only.button.addEventListener("click", function () { wf.run(PLAN_KINDS.only(sid)); });
    force.button.addEventListener("click", function () { wf.run(PLAN_KINDS.force(sid)); });
    S3.confirmButton(reset.button, "确认清除？", function () {
      S3.showError("d-action-error", "");
      S3.api("POST", S3.jobUrl(ctx.name, "/stages/" + encodeURIComponent(sid) + "/reset"), {}).then(function (job) {
        ctx.effect = { text: "已清除「" + s.label + "」的结果，后面依赖它的阶段会显示「需重算」。", at: Date.now() };
        S3.app.applyJob(job);
      }, function (err) { S3.showError("d-action-error", err.message); });
    });

    // parameters
    var params = s.params || [];
    P.formNote = el("p", "hint form-note", "任务正在运行，设置暂时不能改。");
    P.formNote.hidden = true;
    var normal = params.filter(function (p) { return !p.advanced; });
    var adv = params.filter(function (p) { return p.advanced; });
    if (params.length) {
      var section = el("div", "np-section");
      section.appendChild(el("h3", null, "参数"));
      section.appendChild(P.formNote);
      var form = el("div", "np-form");
      normal.forEach(function (p) { form.appendChild(buildField(sid, p, P.fields)); });
      section.appendChild(form);
      if (adv.length) {
        var details = el("details", "adv");
        details.appendChild(el("summary", null, "高级参数（" + adv.length + " 项）"));
        var advForm = el("div", "np-form");
        adv.forEach(function (p) { advForm.appendChild(buildField(sid, p, P.fields)); });
        details.appendChild(advForm);
        section.appendChild(details);
      }
      panel.appendChild(section);
    } else {
      panel.appendChild(P.formNote);
    }

    // models and licences
    P.models = el("div", "np-section np-models");
    panel.appendChild(P.models);
    // results and logs
    P.results = el("div", "np-section np-results");
    panel.appendChild(P.results);

    ctx.panel = P;
    return P;
  }

  function renderActionHints() {
    var P = ctx.panel, job = ctx.job;
    if (!P || !job) { return; }
    var sid = P.sid, eff = effectiveSet(job), st = stateOf(job, sid, eff);
    var on = !!eff[sid], locked = busy(job);
    ["to", "only", "force"].forEach(function (kind) {
      var tile = P.tiles[kind];
      var hint = tile.hint, button = tile.button;
      hint.classList.remove("is-bad");
      if (locked) { button.disabled = true; setText(hint, "任务正在运行"); return; }
      if (!on) { button.disabled = true; setText(hint, "这一步没有打开，先打开它"); return; }
      var result = planFor(PLAN_KINDS[kind](sid));
      if (result.loading) { button.disabled = true; setText(hint, "正在计算…"); return; }
      if (!result.ok) {
        button.disabled = true;
        hint.classList.toggle("is-bad", result.code !== "nothing_to_do");
        setText(hint, result.code === "nothing_to_do" ? "已经是最新的了" : result.message);
        return;
      }
      var names = result.steps.map(function (x) { return x.label || S3.stageLabel(x.stage); });
      button.disabled = false;
      setText(hint, result.steps.length + " 步：" + names.join(" › "));
    });
    var reset = P.tiles.reset;
    reset.button.disabled = locked || !(st === "done" || st === "stale" || st === "failed");
    setText(reset.hint, locked ? "任务正在运行" : reset.button.disabled ? "没有结果可清除" : "删掉这一步的结果，后面的阶段变成「需重算」");
  }

  function renderStatus(job, st, info) {
    var P = ctx.panel, box = P.status;
    box.textContent = "";
    var meta = S3.STAGE_STATE[st] || S3.STAGE_STATE.pending;
    var pct = typeof info.progress === "number" ? Math.round(100 * S3.clamp01(info.progress)) : null;
    S3.setIcon(P.ico, meta.icon);
    S3.setClass(P.ico, "ico np-ico s-" + st);
    setText(P.badge, meta.word + (st === "running" && pct !== null ? " " + pct + "%" : ""));
    S3.setClass(P.badge, "badge s-" + st);
    box.appendChild(el("p", "np-state-help", meta.help + "。"));
    if (st === "running") {
      var row = el("div", "progress-row");
      var track = el("div", "progress" + (pct === null ? " is-indeterminate" : ""));
      var bar = el("div", "progress-bar");
      if (pct !== null) { bar.style.width = pct + "%"; }
      track.appendChild(bar);
      row.appendChild(track);
      row.appendChild(el("span", "progress-percent", pct !== null ? pct + "%" : ""));
      box.appendChild(row);
    }
    if (info.message && st !== "failed") { box.appendChild(el("p", "np-message", info.message)); }
    if (st === "failed") {
      var err = el("pre", "np-error", info.error || info.message || "出错了，没有更多信息，请看日志。");
      err.setAttribute("role", "alert");
      box.appendChild(err);
    }
    if (st === "stale") {
      box.appendChild(el("p", "np-message", "这一步的结果是用旧设置或旧的上游结果算的。重新运行它（或直接点「运行」）才会更新。"));
    }
    var list = facts(info.summary);
    if ((st === "done" || st === "stale") && (list.length || info.seconds)) {
      var dl = el("dl", "facts");
      list.forEach(function (f) { dl.appendChild(el("dt", null, f.label)); dl.appendChild(el("dd", null, f.value)); });
      if (typeof info.seconds === "number") { dl.appendChild(el("dt", null, "用时")); dl.appendChild(el("dd", null, S3.fmtSpent(info.seconds))); }
      box.appendChild(dl);
    }
  }

  function renderEnable(job, eff) {
    var P = ctx.panel, s = def(P.sid), box = P.enable;
    box.textContent = "";
    var w = wfStage(job, P.sid) || { enabled: false };
    if (!s.optional) { box.appendChild(el("p", "hint", "这一步是必需的，不能关闭。")); return; }
    var forced = eff[P.sid] && !w.enabled;
    var label = el("label", "check");
    var input = document.createElement("input");
    input.type = "checkbox";
    input.checked = !!eff[P.sid];
    input.disabled = busy(job) || forced;
    label.appendChild(input);
    label.appendChild(el("span", null, "运行这一步"));
    box.appendChild(label);
    if (forced) {
      box.appendChild(el("p", "hint", "「" + S3.joinLabels(neededBy(job, P.sid)) + "」需要它，所以会自动运行。"));
    } else if (!w.enabled) {
      box.appendChild(el("p", "hint", "这一步没有打开：不会运行，下面的参数先设好，打开后生效。"));
    }
    input.addEventListener("change", function () { toggleStage(P.sid); });
  }

  // --- models and licences of the stage
  function stageFeatures(sid) {
    var m = S3.state.models;
    return m && Array.isArray(m.features) ? m.features.filter(function (f) { return f.stage === sid; }) : [];
  }

  function featureMissingMb(f) {
    var mb = 0;
    (f.files || []).forEach(function (file) { if (!file.installed) { mb += file.mb || 0; } });
    return mb;
  }

  function renderModels(job) {
    var P = ctx.panel, box = P.models;
    var features = stageFeatures(P.sid);
    box.textContent = "";
    box.hidden = features.length === 0;
    if (!features.length) { return; }
    box.appendChild(el("h3", null, "用到的模型"));
    var needFile = {};
    (job.models_needed || []).forEach(function (m) { needFile[m.file] = true; });
    var loopback = S3.state.info && S3.state.info.loopback;
    features.forEach(function (f) {
      var row = el("div", "model-row");
      var usedNow = (f.files || []).some(function (x) { return needFile[x.name]; });
      var head = el("div", "model-head");
      head.appendChild(el("span", "model-name", f.label || f.id));
      var missing = featureMissingMb(f);
      head.appendChild(el("span", "badge " + (f.ready ? "s-done" : "s-queued"),
        f.ready ? "已下载" : "还没下载 约 " + S3.fmtMb(missing)));
      row.appendChild(head);
      var lic = f.licence;
      var note = el("p", "hint model-note");
      if (lic) {
        note.appendChild(document.createTextNode("协议："));
        if (lic.url) {
          var link = el("a", null, lic.label || lic.id);
          link.href = lic.url; link.target = "_blank"; link.rel = "noopener noreferrer";
          note.appendChild(link);
        } else {
          note.appendChild(document.createTextNode(lic.label || lic.id));
        }
        note.appendChild(document.createTextNode("。使用即表示同意该协议。"));
      }
      if (!f.ready) { note.appendChild(document.createTextNode("首次使用会自动下载约 " + S3.fmtMb(missing) + "。")); }
      if (note.firstChild) { row.appendChild(note); }
      var dlState = S3.state.models && S3.state.models.download;
      var downloading = dlState && dlState.state === "running" && (dlState.features || []).indexOf(f.id) >= 0;
      if (downloading) {
        var track = el("div", "progress");
        var bar = el("div", "progress-bar");
        bar.style.width = (100 * S3.clamp01((dlState.downloaded_mb || 0) / Math.max(1, dlState.total_mb || 1))).toFixed(1) + "%";
        track.appendChild(bar);
        row.appendChild(track);
        row.appendChild(el("p", "hint", "下载中 " + S3.fmtMb(dlState.downloaded_mb) + " / " + S3.fmtMb(dlState.total_mb)));
      } else if (dlState && dlState.state === "failed" && (dlState.features || []).indexOf(f.id) >= 0 && !f.ready) {
        row.appendChild(el("p", "next-warn", "上次下载失败：" + (dlState.error || "")));
      }
      if (usedNow && !f.ready) { row.appendChild(el("p", "next-note", "当前设置会用到它：运行这一步时自动下载。")); }
      if (!f.ready && loopback && !downloading) {
        var dl = el("button", "btn small", "现在下载");
        dl.type = "button";
        dl.addEventListener("click", function () {
          dl.disabled = true;
          setText(dl, "已开始下载…");
          S3.api("POST", "/api/models/download", { features: [f.id] }).then(function () {
            return S3.app.loadModels();
          }, function (err) { S3.showError("d-action-error", err.message); dl.disabled = false; setText(dl, "现在下载"); });
        });
        row.appendChild(dl);
      }
      box.appendChild(row);
    });
  }

  function renderResults(job, st) {
    var P = ctx.panel, box = P.results;
    box.textContent = "";
    var buttons = el("div", "actions");
    (VIEWS[P.sid] || []).forEach(function (view) {
      var available = (VIEW_NEEDS[view[0]] || []).every(function (k) { return hasOutput(job, k); });
      if (!available || !(st === "done" || st === "stale")) { return; }
      var b = el("button", "btn small", view[1]);
      b.type = "button";
      b.addEventListener("click", function () { S3.app.openView(view[0]); });
      buttons.appendChild(b);
    });
    var logName = P.sid + ".log";
    var hasLog = (job.logs || []).indexOf(logName) >= 0;
    var logBtn = el("button", "btn small quiet", hasLog ? "这一步的日志" : "运行日志");
    logBtn.type = "button";
    logBtn.addEventListener("click", function () { S3.app.showLog(hasLog ? logName : "pipeline.log"); });
    buttons.appendChild(logBtn);
    if (buttons.children.length) {
      box.appendChild(el("h3", null, "结果和日志"));
      box.appendChild(buttons);
    }
  }

  function updatePanel() {
    var panel = $("np"), job = ctx.job;
    if (!ctx.selected || !job || !def(ctx.selected)) {
      panel.hidden = true;
      ctx.panel = null;
      return;
    }
    var sid = ctx.selected;
    if (!ctx.panel || ctx.panel.sid !== sid) { buildPanel(sid); }
    panel.hidden = false;
    var eff = effectiveSet(job), st = stateOf(job, sid, eff), info = stageInfo(job, sid);
    var signature = JSON.stringify([st, info.message, info.error, info.summary, info.seconds,
      Math.round(100 * (info.progress || 0)), job.state, job.workflow && job.workflow.stages[sid] && job.workflow.stages[sid].enabled,
      eff[sid], (job.logs || []).join(), (job.output_files || []).map(function (f) { return f.path + f.size; }).join(),
      JSON.stringify(job.licences), JSON.stringify(job.models_needed), JSON.stringify(S3.state.models)]);
    var P = ctx.panel;
    if (P.signature !== signature) {
      P.signature = signature;
      renderStatus(job, st, info);
      renderEnable(job, eff);
      renderModels(job);
      renderResults(job, st);
    }
    renderActionHints();
    syncForm(false);
    placePanel();
  }

  // ------------------------------------------------------------------ templates

  function templateRows() {
    var info = S3.templatesInfo || {};
    return info.templates || (S3.schema && S3.schema.templates) || [];
  }

  function buildTemplates() {
    var box = $("wf-templates"), job = ctx.job;
    box.textContent = "";
    var info = S3.templatesInfo || {};
    var locked = busy(job);
    box.appendChild(el("p", "hint", "模板是一组预先选好的开关和参数。换模板会用模板的设置替换这个任务现在的设置；已经算出来的结果不会删，设置变了的阶段会显示「需重算」。"));
    var list = el("ul", "tpl-list");
    templateRows().forEach(function (t) {
      var li = el("li", "tpl-item" + (t.id === job.template ? " is-current" : ""));
      var text = el("div", "tpl-text");
      var name = el("span", "tpl-name", t.label);
      if (t.id === job.template) { name.appendChild(el("span", "tag is-accent", "当前")); }
      if (t.id === info.default_template) { name.appendChild(el("span", "tag", "默认")); }
      if (!t.builtin) { name.appendChild(el("span", "tag", "我的")); }
      text.appendChild(name);
      if (t.desc) { text.appendChild(el("span", "tpl-desc", t.desc)); }
      var runs = S3.templateStages(t);
      if (runs) { text.appendChild(el("span", "tpl-desc", runs)); }
      li.appendChild(text);
      var actions = el("div", "tpl-actions");
      if (t.id !== job.template) {
        var use = el("button", "btn small", "用这个");
        use.type = "button";
        use.disabled = locked;
        S3.confirmButton(use, "确认替换设置？", function () {
          sendChange({ template: t.id }, "换成模板「" + t.label + "」").then(function () { wf.toggleTemplates(false); });
        });
        actions.appendChild(use);
      }
      if (t.id !== info.default_template) {
        var def_ = el("button", "btn small quiet", "设为默认");
        def_.type = "button";
        def_.addEventListener("click", function () {
          S3.api("PUT", "/api/templates/default", { id: t.id }).then(function () { return wf.refreshTemplates(); },
            function (err) { ctx.tplMessage = err.message; buildTemplates(); });
        });
        actions.appendChild(def_);
      }
      if (!t.builtin) {
        var del = el("button", "btn small quiet-danger", "删除");
        del.type = "button";
        S3.confirmButton(del, "确认删除？", function () {
          S3.api("DELETE", "/api/templates/" + encodeURIComponent(t.id)).then(function () { return wf.refreshTemplates(); },
            function (err) { ctx.tplMessage = err.message; buildTemplates(); });
        });
        actions.appendChild(del);
      }
      li.appendChild(actions);
      list.appendChild(li);
    });
    box.appendChild(list);

    var save = el("form", "tpl-save");
    save.setAttribute("aria-label", "把当前设置存为模板");
    var nameLab = el("label", "field");
    nameLab.appendChild(el("span", "field-label", "把这个任务现在的设置存为模板"));
    var name = document.createElement("input");
    name.type = "text";
    name.maxLength = 40;
    name.placeholder = "模板名称，例如 室内物体";
    name.autocomplete = "off";
    nameLab.appendChild(name);
    save.appendChild(nameLab);
    var go = el("button", "btn small", "存为模板");
    go.type = "submit";
    save.appendChild(go);
    save.addEventListener("submit", function (event) {
      event.preventDefault();
      var label = name.value.trim();
      if (!label) { ctx.tplMessage = "请先给模板起个名字。"; buildTemplates(); return; }
      var stages = {};
      stageList().forEach(function (s) {
        var w = wfStage(ctx.job, s.id);
        stages[s.id] = { enabled: s.optional ? !!w.enabled : true, params: JSON.parse(JSON.stringify(w.params || {})) };
      });
      var id = "my-" + Date.now().toString(36);
      go.disabled = true;
      S3.api("PUT", "/api/templates/" + id, { label: label, base: ctx.job.template || undefined, stages: stages }).then(function () {
        ctx.tplMessage = "已存为模板「" + label + "」。";
        return wf.refreshTemplates();
      }, function (err) { ctx.tplMessage = err.message; buildTemplates(); });
    });
    box.appendChild(save);
    if (ctx.tplMessage) {
      var msg = el("p", "hint tpl-message", ctx.tplMessage);
      msg.setAttribute("role", "status");
      box.appendChild(msg);
    }
  }

  function updateTemplateButton() {
    var job = ctx.job, btn = $("wf-template-btn");
    var changed = job.changes ? Object.keys(job.changes).length : 0;
    setText($("wf-template-name"), (templateLabel(job.template) || "（自定义）") + (changed ? "（已改动 " + changed + " 个阶段）" : ""));
    btn.setAttribute("aria-expanded", String(ctx.tplOpen));
  }

  // ------------------------------------------------------------------ public

  var wf = S3.wf = {};

  wf.init = function () {
    // the legend
    var legend = $("wf-legend");
    S3.STATE_ORDER.forEach(function (st) {
      var meta = S3.STAGE_STATE[st];
      var li = el("li", "legend-item s-" + st);
      li.appendChild(S3.icon(meta.icon, "legend-ico"));
      li.appendChild(el("span", null, meta.word));
      li.title = meta.help;
      legend.appendChild(li);
    });
    $("wf-template-btn").addEventListener("click", function () { wf.toggleTemplates(!ctx.tplOpen); });
    if (typeof ResizeObserver === "function") {
      new ResizeObserver(function () { if (ctx.job && ctx.built) { ctx.width = 0; layoutGraph(); } }).observe($("wf-wrap"));
    } else {
      window.addEventListener("resize", function () { if (ctx.job && ctx.built) { ctx.width = 0; layoutGraph(); } });
    }
  };

  wf.toggleTemplates = function (open) {
    ctx.tplOpen = open;
    $("wf-templates").hidden = !open;
    if (open) { ctx.tplMessage = ""; buildTemplates(); }
    if (ctx.job) { updateTemplateButton(); }
  };

  wf.refreshTemplates = function () {
    return S3.api("GET", "/api/templates").then(function (info) {
      S3.templatesInfo = info;
      if (ctx.job) { updateTemplateButton(); if (ctx.tplOpen) { buildTemplates(); } }
      S3.app.templatesChanged();
    });
  };

  wf.select = function (sid) {
    ctx.selected = sid;
    if (ctx.job) {
      updateNodes();
      updatePanel();
      if (sid && ctx.panel && ctx.mode === "list") {
        var panel = $("np");
        if (panel.scrollIntoView) { panel.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
      }
    }
    if (sid) { S3.app.loadModels(); }
  };

  wf.selected = function () { return ctx.selected; };

  wf.run = function (body) {
    var job = ctx.job;
    if (!job) { return Promise.resolve(); }
    S3.showError("d-action-error", "");
    return S3.api("POST", S3.jobUrl(job.job, "/run"), body || {}).then(function (next) {
      ctx.effect = null;
      S3.app.applyJob(next);
    }, function (err) {
      S3.showError("d-action-error", err.message);
      S3.app.refreshSoon();
    });
  };

  wf.update = function (job) {
    if (!S3.schema) { return; }
    var jobChanged = ctx.name !== job.job;
    if (jobChanged) {
      var keep = ctx.selected;
      wf.reset();
      ctx.name = job.job;
      ctx.selected = null;
      if (keep && false) { ctx.selected = keep; }
    }
    ctx.job = job;
    if (!ctx.built) { buildGraph(); }
    // a failed stage is shown once, so the reason is on screen
    if (jobChanged || ctx.lastFailed !== job.job + ":" + (job.stage || "")) {
      var failedId = null;
      stageList().forEach(function (s) { if (stateOf(job, s.id, null) === "failed" && !failedId) { failedId = s.id; } });
      if (failedId && ctx.failedShown !== job.job + ":" + failedId) {
        ctx.failedShown = job.job + ":" + failedId;
        ctx.selected = failedId;
      }
      ctx.lastFailed = job.job + ":" + (job.stage || "");
    }
    var sig = planSignature(job);
    if (sig !== ctx.planSig) { ctx.planSig = sig; ctx.plans = {}; }
    layoutGraph();
    updateNodes();
    updateEdges();
    updatePanel();
    renderNext();
    updateRunButton();
    updateTemplateButton();
    if (ctx.tplOpen && !ctx.tplBuilt) { buildTemplates(); }
    if (ctx.effect && Date.now() - ctx.effect.at > 60000) { ctx.effect = null; }
    $("wf-card").classList.toggle("is-busy", busy(job));
    setText($("wf-busy-note"), busy(job) ? "任务正在运行，开关和参数暂时不能改。" : "");
  };

  wf.reset = function () {
    var panel = $("np"), home = $("np-home");
    if (panel && home && home.nextSibling !== panel) { home.after(panel); }
    panel.hidden = true;
    var old = ctx;
    ctx = freshContext();
    ctx.tplOpen = false;
    $("wf-templates").hidden = true;
    var graph = $("wf-graph");
    graph.querySelectorAll(".node").forEach(function (n) { n.remove(); });
    $("wf-edges").querySelectorAll("path").forEach(function (p) { p.remove(); });
    void old;
  };

  wf.templatesChanged = function () {
    if (ctx.job) { updateTemplateButton(); if (ctx.tplOpen) { buildTemplates(); } }
  };

  wf.modelsChanged = function () {
    if (ctx.job && ctx.panel) { ctx.panel.signature = null; updatePanel(); }
  };

  wf.relayout = function () { if (ctx.job && ctx.built) { ctx.width = 0; layoutGraph(); } };
})();
