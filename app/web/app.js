/* scan3d UI: plain JavaScript, no dependencies. Talks to the JSON API in app/server.py.
 * Everything put on the page from data goes through textContent, never innerHTML. */
(function () {
  "use strict";

  var STAGES = [
    ["ingest", "采集数据"], ["sfm", "相机位姿"], ["dataset", "几何先验"],
    ["train", "训练高斯"], ["mesh", "生成网格"], ["export", "导出"]
  ];
  var STATE_LABEL = {
    created: "待开始", queued: "排队中", running: "进行中",
    done: "已完成", failed: "失败", cancelled: "已取消"
  };
  var QUALITY_LABEL = { preview: "预览", standard: "标准", high: "高质量" };
  var OUTPUT_LABEL = {
    splat: "高斯点云", mesh: "网格模型", collision: "碰撞网格", report: "扫描报告",
    preview: "网页预览（完整）", preview_lite: "网页预览（轻量）"     // what "在线查看" shows
  };
  var OUTPUT_NAME_LABEL = {     // files in output/ that status.json does not name
    "scene_transform.json": "场景变换", "preview.spz": "网页预览（完整）", "preview-lite.spz": "网页预览（轻量）"
  };
  var OUTPUT_OUTDATED_LABEL = "网页预览（旧版，已不使用）";   // the viewer made itself a newer one
  var EXTENSIONS = ["jpg", "jpeg", "png", "webp", "heic", "heif", "tif", "tiff", "bmp",
                    "mp4", "mov", "m4v", "mkv", "avi", "webm"];
  var JOB_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

  var POLL_ACTIVE_MS = 1500;      // while something is queued or running
  var POLL_IDLE_MS = 6000;
  var POLL_ERROR_MS = 3000;
  var API_TIMEOUT_MS = 15000;
  var UPLOAD_STALL_MS = 60000;    // no progress for this long: abort the request and retry
  var RETRY_DELAYS_MS = [1000, 2000, 4000, 8000, 10000, 10000, 10000, 10000];
  var LOG_KEEP_CHARS = 300000;

  var state = {
    info: null,
    route: { view: "home", job: null, tab: null },
    mode: "upload",        // "upload" | "path"
    items: [],             // files chosen for the next scan
    upload: null,          // { job, running, aborted, xhr, startedAt, startedBytes }
    busy: false,           // a create/start request for a local path is in flight
    detail: null,          // last view of the job shown in the detail page
    detailAt: 0,
    log: { job: null, file: null, offset: null, raw: "" },
    logOpenedFor: null,
    rows: new Map(),
    models: null,          // last answer of /api/models
    modelsProblem: "",     // why it could not be read
    modelsBusy: false,     // an accept or download request is in flight
    modelsWanted: null,    // whether the card had a reason to be open at the last look
    modelsOpenedByUser: false,
    masksTouched: false    // the user changed the "remove people" box: only then is it sent
  };

  function $(id) { return document.getElementById(id); }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
  }

  function setText(node, text) {
    if (node.textContent !== text) { node.textContent = text; }
  }

  function sleep(ms) { return new Promise(function (resolve) { setTimeout(resolve, ms); }); }

  // ------------------------------------------------------------------ formatting

  function fmtBytes(n) {
    if (typeof n !== "number" || !isFinite(n)) { return ""; }
    if (n < 1024) { return n + " B"; }
    var units = ["KB", "MB", "GB", "TB"];
    var value = n / 1024, i = 0;
    while (value >= 1024 && i < units.length - 1) { value /= 1024; i++; }
    return (value >= 100 ? value.toFixed(0) : value.toFixed(1)) + " " + units[i];
  }

  function pad2(n) { return n < 10 ? "0" + n : String(n); }

  function fmtDuration(seconds) {
    if (typeof seconds !== "number" || !isFinite(seconds) || seconds < 0) { return ""; }
    var s = Math.floor(seconds);
    var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return (h > 0 ? h + ":" + pad2(m) : String(m)) + ":" + pad2(s % 60);
  }

  function fmtRemaining(seconds) {
    if (!isFinite(seconds) || seconds <= 0) { return ""; }
    if (seconds < 60) { return "不到 1 分钟"; }
    if (seconds < 3600) { return "约 " + Math.round(seconds / 60) + " 分钟"; }
    return "约 " + (seconds / 3600).toFixed(1) + " 小时";
  }

  function fmtTime(iso) {
    if (!iso) { return ""; }
    var d = new Date(iso);
    if (isNaN(d.getTime())) { return ""; }
    return (d.getMonth() + 1) + "月" + d.getDate() + "日 " + pad2(d.getHours()) + ":" + pad2(d.getMinutes());
  }

  function stageLabel(key) {
    for (var i = 0; i < STAGES.length; i++) { if (STAGES[i][0] === key) { return STAGES[i][1]; } }
    return "";
  }

  function stageIndex(key) {
    for (var i = 0; i < STAGES.length; i++) { if (STAGES[i][0] === key) { return i; } }
    return -1;
  }

  function firstLine(text) {
    return String(text || "").split("\n")[0];
  }

  // ------------------------------------------------------------------ API

  function apiError(message, status, data) {
    var err = new Error(message);
    err.status = status || 0;
    err.data = data || null;
    return err;
  }

  function api(method, path, body) {
    var controller = typeof AbortController === "function" ? new AbortController() : null;
    var timer = setTimeout(function () { if (controller) { controller.abort(); } }, API_TIMEOUT_MS);
    var options = { method: method, cache: "no-store", credentials: "same-origin" };
    if (controller) { options.signal = controller.signal; }
    if (body !== undefined) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }
    return fetch(path, options).then(function (res) {
      return res.text().then(function (text) {
        var data = null;
        try { data = text ? JSON.parse(text) : null; } catch (e) { data = null; }
        if (!res.ok) {
          throw apiError((data && data.message) || ("请求失败（" + res.status + "）"), res.status, data);
        }
        return data;
      });
    }, function () {
      throw apiError("连接不上电脑上的扫描程序", 0, null);
    }).finally(function () { clearTimeout(timer); });
  }

  function jobUrl(job, rest) {
    return "/api/jobs/" + encodeURIComponent(job) + (rest || "");
  }

  // ------------------------------------------------------------------ banner and errors

  var bannerKind = null;

  function showBanner(kind, text) {
    bannerKind = kind;
    var banner = $("banner");
    setText(banner, text);
    banner.hidden = false;
  }

  function hideBanner(kind) {
    if (bannerKind === kind || kind === undefined) {
      bannerKind = null;
      $("banner").hidden = true;
    }
  }

  function showError(id, text) {
    var node = $(id);
    node.textContent = text;
    node.hidden = !text;
  }

  // ------------------------------------------------------------------ models and licences

  var FEATURE_DESC = {      // what each feature does, in one line
    masks: "把照片里的人这类会动的东西抠掉，不参与重建。",
    subject_only: "扫描单个物体时只保留主体，背景不重建。",
    normals: "估计照片里每处表面的朝向，让墙面、地面这类平整表面保持平整。",
    dense_seed: "训练前先算出一份稠密点云作为起点。",
    sfm: "常规特征算不出相机位置时改用的另一套特征。"
  };
  var FEATURE_OFF_NOTE = {  // said instead of plain "默认关闭" for a feature no preset turns on
    subject_only: "默认关闭，新建扫描时选“物体”才使用。",
    dense_seed: "默认关闭：两个测试场景里有一个效果变差。要启用需修改 app/settings.json。"
  };
  var MODEL_BADGE = { ready: "s-done", licence: "s-queued", missing: "s-queued", downloading: "s-running", failed: "s-failed" };
  var modelRows = {};       // feature -> the nodes that change while a download runs
  var modelsSignature = null;

  function fmtMb(n) { return String(Math.round(typeof n === "number" && isFinite(n) ? n : 0)); }

  function featureProgress(feature, download) {
    var done = 0, total = 0;
    (download.files || []).forEach(function (file) {
      if (feature.files.indexOf(file.name) >= 0) { done += file.downloaded_mb; total += file.size_mb; }
    });
    return { done: done, total: total };
  }

  // Where one feature stands: kind is ready | licence | missing | downloading | failed, and
  // text says it in words. null while the state of the models is not known.
  function featureState(key) {
    var models = state.models;
    var feature = models && models.features[key];
    if (!feature) { return null; }
    var download = models.download || {};
    var mine = (download.features || []).indexOf(key) >= 0;
    if (download.state === "running" && mine) {
      var p = featureProgress(feature, download);
      return { kind: "downloading", text: "下载中 " + fmtMb(p.done) + " / " + fmtMb(p.total) + " MB", done: p.done, total: p.total };
    }
    if (feature.ready) { return { kind: "ready", text: "可用" }; }
    if (feature.licences_needed.length) { return { kind: "licence", text: "需要同意许可" }; }
    if (download.state === "failed" && mine) { return { kind: "failed", text: "下载失败" }; }
    return { kind: "missing", text: "需要下载 " + fmtMb(feature.download_mb) + " MB" };
  }

  function licenceNames(feature) {
    var licences = state.models.licences;
    return feature.licences_needed.map(function (id) { return (licences[id] && licences[id].name) || id; });
  }

  // An accept or download request; its answer is the new overview.
  function modelsRequest(request) {
    state.modelsBusy = true;
    showError("models-error", "");
    renderModels();
    request.then(function (overview) { state.models = overview; }, function (err) { showError("models-error", err.message); })
      .then(function () {
        state.modelsBusy = false;
        renderModels();
        renderOptions();
        poll();
      });
  }

  function downloadFeature(key) {
    modelsRequest(api("POST", "/api/models/download", { features: [key] }));
  }

  // Runs only from a click on a button whose text names the licences in `ids`.
  function acceptLicences(key, ids) {
    modelsRequest(api("POST", "/api/models/accept", { licences: ids }).then(function (overview) {
      state.models = overview;      // the acceptance stands, whatever happens to the download
      var feature = overview.features[key];
      if (feature && feature.files_missing.length && overview.download.state !== "running") {
        return api("POST", "/api/models/download", { features: [key] });
      }
      return overview;
    }));
  }

  function actionButton(text, onClick) {
    var button = el("button", "btn small", text);
    button.type = "button";
    button.disabled = state.modelsBusy;
    button.addEventListener("click", onClick);
    return button;
  }

  function buildModelRow(key, feature) {
    var models = state.models;
    var st = featureState(key);
    var downloading = models.download.state === "running";
    var row = el("li", "model-row");
    var head = el("div", "model-head");
    var badge = el("span", "badge " + MODEL_BADGE[st.kind], st.text);
    head.appendChild(el("span", "model-name", feature.label || key));
    head.appendChild(badge);
    row.appendChild(head);
    row.appendChild(el("p", "hint", (FEATURE_DESC[key] || "") +
      (feature.enabled ? "默认开启。" : (FEATURE_OFF_NOTE[key] || "默认关闭。"))));
    modelRows[key] = { badge: badge, bar: null };

    feature.licences_needed.forEach(function (id) {
      var licence = models.licences[id] || {};
      var line = el("p", "model-licence", "许可：");
      var secure = false;
      try { secure = new URL(licence.url).protocol === "https:"; } catch (e) { secure = false; }
      if (secure) {     // the one link on the page that leaves this PC; it opens in a new tab
        var link = el("a", null, licence.name || id);
        link.href = licence.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        line.appendChild(link);
        line.appendChild(document.createTextNode("（在新标签页打开）"));
      } else {
        line.appendChild(document.createTextNode(licence.name || id));
      }
      row.appendChild(line);
    });

    if (st.kind === "downloading") {
      var track = el("div", "progress");
      var bar = el("div", "progress-bar");
      track.appendChild(bar);
      row.appendChild(track);
      modelRows[key].bar = bar;
    }
    if (st.kind === "failed") {
      row.appendChild(el("p", "model-error", models.download.error || "没有错误信息。"));
    }
    if (!models.can_manage || st.kind === "ready" || st.kind === "downloading") { return row; }

    var actions = el("div", "actions");
    if (st.kind === "licence") {
      var ids = feature.licences_needed.slice();
      var names = licenceNames(feature).join("、");
      // While another download runs this one cannot start: accept now, download afterwards.
      var withDownload = feature.download_mb > 0 && !downloading;
      actions.appendChild(actionButton(
        "同意 " + names + (withDownload ? " 并下载（" + fmtMb(feature.download_mb) + " MB）" : ""),
        function () { acceptLicences(key, ids); }));
    } else {
      var button = actionButton(st.kind === "failed" ? "重试" : "下载（" + fmtMb(feature.download_mb) + " MB）",
        function () { downloadFeature(key); });
      if (downloading) { button.disabled = true; }
      actions.appendChild(button);
      if (downloading) { row.appendChild(el("p", "hint", "一次只下载一项，等当前的下载结束。")); }
    }
    row.appendChild(actions);
    return row;
  }

  function renderModels() {
    var card = $("models-card");
    var models = state.models;
    if (!models) {
      card.hidden = !state.modelsProblem;
      setText($("models-summary"), state.modelsProblem);
      return;
    }
    card.hidden = false;
    var features = models.features, download = models.download;
    var keys = Object.keys(features);
    var needed = keys.filter(function (k) { return features[k].enabled && !features[k].ready; });
    var optional = keys.filter(function (k) { return !features[k].enabled && !features[k].ready; });
    var running = download.state === "running";
    var failed = download.state === "failed" && download.features.some(function (k) { return features[k] && !features[k].ready; });

    var summary;
    if (state.modelsProblem) { summary = state.modelsProblem; }
    else if (running) { summary = "下载中 " + fmtMb(download.downloaded_mb) + " / " + fmtMb(download.total_mb) + " MB"; }
    else if (failed) { summary = "下载失败"; }
    else if (needed.length) { summary = needed.length + " 项默认功能还不能用"; }
    else if (optional.length) { summary = "默认功能都可用；" + optional.length + " 项可选功能未准备"; }
    else { summary = "全部可用"; }
    setText($("models-summary"), summary);

    // Open by itself only when a default scan lacks something or a download needs watching,
    // and closed again when that is over, unless the user opened the card.
    var wanted = needed.length > 0 || running || failed;
    if (wanted !== state.modelsWanted) {
      state.modelsWanted = wanted;
      if (wanted) { card.open = true; }
      else if (!state.modelsOpenedByUser) { card.open = false; }
    }

    var signature = JSON.stringify([features, models.licences, download.state, download.features,
                                    download.error, models.can_manage, state.modelsBusy]);
    var list = $("model-list");
    if (signature !== modelsSignature) {
      modelsSignature = signature;
      modelRows = {};
      list.textContent = "";
      keys.forEach(function (key) { list.appendChild(buildModelRow(key, features[key])); });
    }
    keys.forEach(function (key) {     // the numbers of a running download change between rebuilds
      var st = featureState(key), nodes = modelRows[key];
      if (!nodes) { return; }
      setText(nodes.badge, st.text);
      if (nodes.bar && st.kind === "downloading") {
        nodes.bar.style.width = (100 * st.done / Math.max(1, st.total)).toFixed(1) + "%";
      }
    });
    $("models-remote").hidden = models.can_manage || (needed.length === 0 && optional.length === 0);
  }

  function refreshModels() {
    return api("GET", "/api/models").then(function (overview) {
      state.models = overview;
      state.modelsProblem = "";
      renderModels();
      renderOptions();
      return overview.download.state === "running";
    }, function (err) {
      if (err.status !== 500) { throw err; }
      // The state of the models cannot be read; the rest of the page works without it.
      state.modelsProblem = err.message;
      renderModels();
      return false;
    });
  }

  // ------------------------------------------------------------------ new scan: choosing files

  var stripControl;
  try { stripControl = new RegExp("[\\p{Cc}\\p{Cf}\\p{Cs}]", "gu"); }
  catch (e) { stripControl = /[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]/g; }

  // The server refuses names it considers unsafe; make an acceptable name from what the
  // phone or camera produced instead of failing the upload.
  function safeName(original) {
    var name = String(original || "");
    if (name.normalize) { name = name.normalize("NFC"); }
    name = name.replace(stripControl, "").replace(/[<>:"\/\\|?*]/g, "_");
    name = name.replace(/^[.\s]+/, "").replace(/[.\s]+$/, "");
    var dot = name.lastIndexOf(".");
    var ext = dot > 0 ? name.slice(dot) : "";
    var stem = dot > 0 ? name.slice(0, dot) : name;
    stem = stem.replace(/[.\s]+$/, "");
    if (stem.length + ext.length > 100) { stem = stem.slice(0, Math.max(1, 100 - ext.length)); }
    if (!stem || /^(con|prn|aux|nul|conin\$|conout\$|com\d|lpt\d)$/i.test(stem.split(".")[0])) {
      stem = "file_" + stem;
    }
    return stem + ext;
  }

  function extensionOf(name) {
    var dot = name.lastIndexOf(".");
    return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
  }

  function uploadLocked() { return !!(state.upload && (state.upload.running || state.upload.job)); }

  function addFiles(fileList) {
    if (uploadLocked() || !fileList) { return; }
    var taken = {};
    var known = {};
    state.items.forEach(function (item) {
      taken[item.name.toLowerCase()] = true;
      known[item.file.name + "|" + item.file.size + "|" + item.file.lastModified] = true;
    });
    for (var i = 0; i < fileList.length; i++) {
      var file = fileList[i];
      var identity = file.name + "|" + file.size + "|" + file.lastModified;
      if (known[identity]) { continue; }
      known[identity] = true;
      var name = safeName(file.name);
      var problem = null;
      if (EXTENSIONS.indexOf(extensionOf(name)) < 0) { problem = "不支持"; }
      else if (file.size === 0) { problem = "空文件"; }
      if (!problem) {
        var dot = name.lastIndexOf(".");
        var stem = name.slice(0, dot), ext = name.slice(dot);
        var n = 2;
        while (taken[name.toLowerCase()]) { name = stem + "-" + n + ext; n++; }
        taken[name.toLowerCase()] = true;
      }
      state.items.push({ file: file, name: name, size: file.size, sent: 0,
                         status: problem ? "rejected" : "pending", problem: problem, row: null });
    }
    showError("form-error", "");
    renderFiles();
  }

  function usableItems() {
    return state.items.filter(function (item) { return item.status !== "rejected"; });
  }

  function renderFiles() {
    var list = $("file-list");
    list.textContent = "";
    state.items.forEach(function (item) {
      var row = el("li", "file-row");
      var bar = el("div", "file-bar");
      var fill = el("span");
      bar.appendChild(fill);
      item.row = { li: row, status: el("span", "file-status"), fill: fill, bar: bar };
      row.appendChild(el("span", "file-name", item.name));
      row.appendChild(el("span", "file-size", fmtBytes(item.size)));
      row.appendChild(item.row.status);
      row.appendChild(bar);
      list.appendChild(row);
      updateFileRow(item);
    });
    var usable = usableItems();
    var total = usable.reduce(function (sum, item) { return sum + item.size; }, 0);
    var rejected = state.items.length - usable.length;
    var text = usable.length + " 个文件，共 " + fmtBytes(total);
    if (rejected) { text += "（另有 " + rejected + " 个不会上传）"; }
    setText($("file-summary-text"), text);
    $("file-summary").hidden = state.items.length === 0;
    list.hidden = state.items.length === 0;
    $("btn-clear-files").hidden = uploadLocked();
  }

  function updateFileRow(item) {
    var row = item.row;
    if (!row) { return; }
    var text = "", cls = "file-row", showBar = false;
    if (item.status === "rejected") { text = item.problem; cls += " is-rejected"; }
    else if (item.status === "uploading") {
      text = Math.floor(100 * item.sent / Math.max(1, item.size)) + "%";
      cls += " is-active"; showBar = true;
    }
    else if (item.status === "retry") { text = "重试中…"; cls += " is-active"; showBar = true; }
    else if (item.status === "done") { text = "已上传"; cls += " is-done"; }
    else if (item.status === "error") { text = "失败"; cls += " is-error"; }
    else if (uploadLocked()) { text = "等待"; }
    setText(row.status, text);
    if (row.li.className !== cls) { row.li.className = cls; }
    row.bar.hidden = !showBar;
    row.fill.style.width = (100 * item.sent / Math.max(1, item.size)).toFixed(1) + "%";
  }

  // ------------------------------------------------------------------ new scan: uploading

  function selectedQuality() {
    var checked = document.querySelector('input[name="quality"]:checked');
    return checked ? checked.value : "standard";
  }

  function scanKind() {
    var checked = document.querySelector('input[name="scan-kind"]:checked');
    return checked ? checked.value : "scene";
  }

  // The per-scan switches as the server takes them. A switch that is left out is decided by
  // the quality preset, so "remove people" is sent only once the user has changed the box.
  function selectedOptions() {
    var options = {};
    if (scanKind() === "object") { options.subject_only = true; }
    else if (state.masksTouched) { options.masks = $("opt-masks").checked; }
    return options;
  }

  // What to say when a step that was chosen needs a model that is not usable yet.
  function notReadyNote(key, what) {
    var st = featureState(key);
    if (!st || st.kind === "ready") { return ""; }
    if (st.kind === "licence") { return what + "还不能用（" + st.text + "）：这一步会被跳过，任务里会有提示。"; }
    return what + "还不能用（" + st.text + "）：扫描会在这一步出错，请先在上面的“模型与许可”里下载。";
  }

  function renderOptions() {
    var up = state.upload;
    var locked = !!(up && up.running) || state.busy;
    var object = scanKind() === "object";
    var radios = document.querySelectorAll('input[name="scan-kind"]');
    for (var i = 0; i < radios.length; i++) { radios[i].disabled = locked; }
    var box = $("opt-masks");
    box.disabled = locked || object;

    var kindNote = object ? notReadyNote("subject_only", "主体分割模型") : "";
    setText($("scan-kind-note"), kindNote);
    $("scan-kind-note").hidden = !kindNote;

    var note = "";
    if (object) { note = "物体扫描用主体蒙版代替这一步。"; }
    else if (box.checked) {
      var preview = selectedQuality() === "preview";
      if (preview && !state.masksTouched) { note = "预览质量默认不抠除。"; }
      else { note = (preview ? "预览质量默认不抠除；你已勾选，这次会抠除。" : "") + notReadyNote("masks", "抠除用的模型"); }
    }
    setText($("opt-masks-note"), note);
    $("opt-masks-note").hidden = !note;
  }

  function renderForm() {
    var up = state.upload;
    var running = !!(up && up.running);
    var paused = !!(up && up.job && !up.running);
    var locked = running || state.busy;
    renderOptions();
    $("btn-start").disabled = locked;
    setText($("btn-start"), paused ? "继续上传" : (running ? "正在上传…" : "开始扫描"));
    $("btn-abort").hidden = !(running || paused);
    setText($("btn-abort"), paused ? "放弃并删除" : "取消上传");
    $("file-input").disabled = uploadLocked();
    $("job-name").disabled = locked || paused;
    $("local-path").disabled = locked;
    $("tab-upload").disabled = uploadLocked() || state.busy;
    $("tab-path").disabled = uploadLocked() || state.busy;
    var radios = document.querySelectorAll('input[name="quality"]');
    for (var i = 0; i < radios.length; i++) { radios[i].disabled = locked; }
    $("btn-clear-files").hidden = uploadLocked();
    $("upload-progress").hidden = !(running || paused);
    $("dropzone").classList.toggle("is-disabled", uploadLocked());
    state.items.forEach(updateFileRow);   // waiting rows say so only while an upload is on
  }

  function renderUploadProgress() {
    var up = state.upload;
    if (!up) { return; }
    var usable = usableItems();
    var total = 0, sent = 0, done = 0;
    usable.forEach(function (item) {
      total += item.size;
      sent += item.status === "done" ? item.size : Math.min(item.sent, item.size);
      if (item.status === "done") { done++; }
    });
    $("upload-bar").style.width = (100 * sent / Math.max(1, total)).toFixed(1) + "%";
    var text = "已上传 " + done + " / " + usable.length + " 个文件 · " + fmtBytes(sent) + " / " + fmtBytes(total);
    if (up.running) {
      var seconds = (Date.now() - up.startedAt) / 1000;
      var moved = sent - up.startedBytes;
      if (seconds > 3 && moved > 0) {
        var speed = moved / seconds;
        text += " · " + fmtBytes(speed) + "/秒";
        var left = fmtRemaining((total - sent) / speed);
        if (left) { text += " · 剩余" + left; }
      }
    }
    setText($("upload-text"), text);
  }

  function putFile(up, item, offset) {
    return new Promise(function (resolve, reject) {
      var xhr = new XMLHttpRequest();
      var lastActivity = Date.now();
      var stalled = false;
      var watchdog = setInterval(function () {
        if (Date.now() - lastActivity > UPLOAD_STALL_MS) { stalled = true; xhr.abort(); }
      }, 5000);
      function finish(fn, value) { clearInterval(watchdog); up.xhr = null; fn(value); }

      up.xhr = xhr;
      xhr.open("PUT", jobUrl(up.job, "/files/" + encodeURIComponent(item.name)) + (offset ? "?offset=" + offset : ""));
      xhr.setRequestHeader("Content-Type", "application/octet-stream");
      xhr.upload.onprogress = function (event) {
        lastActivity = Date.now();
        item.sent = offset + event.loaded;
        updateFileRow(item);
        renderUploadProgress();
      };
      xhr.onload = function () {
        if (xhr.status >= 200 && xhr.status < 300) { finish(resolve); return; }
        var data = null;
        try { data = JSON.parse(xhr.responseText); } catch (e) { data = null; }
        finish(reject, apiError((data && data.message) || ("上传失败（" + xhr.status + "）"), xhr.status, data));
      };
      xhr.onerror = function () { finish(reject, apiError("网络中断", 0, null)); };
      xhr.onabort = function () { finish(reject, apiError(stalled ? "上传长时间没有进展" : "已取消", 0, null)); };
      xhr.send(offset ? item.file.slice(offset) : item.file);
    });
  }

  function uploadItem(up, item) {
    var attempt = 0;
    function tryOnce() {
      if (up.aborted) { return Promise.reject(apiError("已取消", 0, null)); }
      var resume = Promise.resolve(0);
      if (attempt > 0 || item.sent > 0) {
        // Ask how much of this file the PC already has, and continue from there.
        resume = api("GET", jobUrl(up.job, "/files/" + encodeURIComponent(item.name))).then(function (info) {
          if (info.complete) { return info.size === item.size ? -1 : 0; }
          return info.size <= item.size ? info.size : 0;
        });
      }
      return resume.then(function (offset) {
        if (offset < 0) { return; }
        if (offset === item.size) { offset = 0; }  // nothing left to send: start this file over
        item.status = "uploading";
        item.sent = offset;
        updateFileRow(item);
        return putFile(up, item, offset);
      }).then(function () {
        item.status = "done";
        item.sent = item.size;
        updateFileRow(item);
        renderUploadProgress();
      }, function (err) {
        if (up.aborted) { throw err; }
        var retriable = !err.status || err.status >= 500 || [408, 409, 429].indexOf(err.status) >= 0;
        if (!retriable || attempt >= RETRY_DELAYS_MS.length) {
          item.status = "error";
          updateFileRow(item);
          throw err;
        }
        item.status = "retry";
        updateFileRow(item);
        return sleep(RETRY_DELAYS_MS[attempt++]).then(tryOnce);
      });
    }
    return tryOnce();
  }

  function runUpload(name, quality, options) {
    var up = state.upload || (state.upload = { job: null });
    up.running = true;
    up.aborted = false;
    showError("form-error", "");
    renderForm();

    var created = up.job ? Promise.resolve() : api("POST", "/api/jobs", { name: name || null, quality: quality, options: options })
      .then(function (job) { up.job = job.job; });

    created.then(function () {
      up.startedAt = Date.now();
      up.startedBytes = usableItems().reduce(function (sum, item) {
        return sum + (item.status === "done" ? item.size : 0);
      }, 0);
      state.items.forEach(function (item) {
        if (item.status === "error" || item.status === "retry") { item.status = "pending"; updateFileRow(item); }
      });
      renderForm();
      renderUploadProgress();
      var queue = usableItems().filter(function (item) { return item.status !== "done"; });
      return queue.reduce(function (chain, item) {
        return chain.then(function () {
          if (item.row && item.row.li.scrollIntoView) {
            item.row.li.scrollIntoView({ block: "nearest" });
          }
          return uploadItem(up, item);
        });
      }, Promise.resolve());
    }).then(function () {
      // Quality and switches as they are on the form now (an upload can be continued later).
      return api("POST", jobUrl(up.job, "/start"), { quality: quality, options: options });
    }).then(function () {
      var job = up.job;
      state.upload = null;
      state.items = [];
      $("job-name").value = "";
      $("file-input").value = "";
      renderFiles();
      renderForm();
      location.hash = "#/job/" + job;
    }).catch(function (err) {
      up.running = false;
      if (up.aborted) { return discardUpload(); }
      renderForm();
      renderUploadProgress();
      if (up.job) {
        showError("form-error", "上传没有完成：" + err.message + "\n已经传完的文件还在，点“继续上传”接着传。");
      } else {
        state.upload = null;
        renderForm();
        showError("form-error", err.message);
      }
    });
  }

  // Remove the half-uploaded job on the PC; the chosen files stay selected in the page.
  function discardUpload() {
    var up = state.upload;
    state.upload = null;
    state.items.forEach(function (item) {
      if (item.status !== "rejected") { item.status = "pending"; item.sent = 0; }
    });
    renderFiles();
    renderForm();
    if (!up || !up.job) { return Promise.resolve(); }
    var tries = 0;
    function attempt() {
      return api("DELETE", jobUrl(up.job)).catch(function (err) {
        // The PC may still be closing the interrupted transfer; give it a moment.
        if (err.status === 404 || ++tries >= 8) { return; }
        return sleep(1500).then(attempt);
      });
    }
    return attempt().then(function () { poll(); });
  }

  function onStart() {
    var name = $("job-name").value.trim();
    var quality = selectedQuality();
    var options = selectedOptions();
    showError("form-error", "");
    if (name && !JOB_NAME_RE.test(name)) {
      showError("form-error", "名称只能包含英文字母、数字、- 和 _，以字母或数字开头，最长 64 个字符。");
      return;
    }
    if (state.mode === "path") {
      var path = $("local-path").value.trim();
      if (!path) { showError("form-error", "请填写文件夹或视频文件的完整路径。"); return; }
      state.busy = true;
      renderForm();
      api("POST", "/api/jobs", { name: name || null, quality: quality, path: path, start: true, options: options })
        .then(function (job) {
          $("job-name").value = "";
          $("local-path").value = "";
          location.hash = "#/job/" + job.job;
        }, function (err) { showError("form-error", err.message); })
        .then(function () { state.busy = false; renderForm(); });
      return;
    }
    if (!usableItems().length) {
      showError("form-error", "请先选择照片或视频。");
      return;
    }
    runUpload(name, quality, options);
  }

  function onAbort() {
    var up = state.upload;
    if (!up) { return; }
    if (up.running) {
      up.aborted = true;
      if (up.xhr) { up.xhr.abort(); }
      return;  // runUpload's error path calls discardUpload
    }
    discardUpload();
  }

  function setMode(mode) {
    state.mode = mode;
    $("tab-upload").classList.toggle("is-active", mode === "upload");
    $("tab-path").classList.toggle("is-active", mode === "path");
    $("tab-upload").setAttribute("aria-selected", String(mode === "upload"));
    $("tab-path").setAttribute("aria-selected", String(mode === "path"));
    $("source-upload").hidden = mode !== "upload";
    $("source-path").hidden = mode !== "path";
    showError("form-error", "");
  }

  // ------------------------------------------------------------------ job list

  function makeJobRow(name) {
    var li = el("li", "job-item");
    var link = el("a");
    link.href = "#/job/" + encodeURIComponent(name);
    var row = {
      li: li,
      name: el("span", "job-name", name),
      badge: el("span", "badge"),
      sub: el("span", "job-sub"),
      progress: el("div", "progress"),
      bar: el("div", "progress-bar")
    };
    row.progress.appendChild(row.bar);
    link.appendChild(row.name);
    link.appendChild(row.badge);
    link.appendChild(row.sub);
    link.appendChild(row.progress);
    li.appendChild(link);
    return row;
  }

  function setBadge(node, jobState) {
    setText(node, STATE_LABEL[jobState] || jobState || "");
    var cls = "badge s-" + jobState;
    if (node.className !== cls) { node.className = cls; }
  }

  function setProgress(track, bar, job) {
    var mode = "is-idle", width = "0";
    if (job.state === "running") {
      if (typeof job.stage_progress === "number") {
        mode = "";
        width = (100 * Math.max(0, Math.min(1, job.stage_progress))).toFixed(1) + "%";
      } else {
        mode = "is-indeterminate";
        width = "";
      }
    } else if (job.state === "done") {
      mode = "";
      width = "100%";
    }
    var cls = "progress" + (mode ? " " + mode : "");
    if (track.className !== cls) { track.className = cls; }
    bar.style.width = width;
  }

  function jobSummary(job) {
    var parts = [];
    var quality = QUALITY_LABEL[job.quality];
    if (job.state === "running") {
      var label = stageLabel(job.stage);
      if (label) {
        parts.push(typeof job.stage_progress === "number"
          ? label + " " + Math.round(100 * Math.max(0, Math.min(1, job.stage_progress))) + "%" : label);
      }
      if (typeof job.elapsed_seconds === "number") { parts.push("已用时 " + fmtDuration(job.elapsed_seconds)); }
    } else if (job.state === "queued") {
      parts.push(job.queue_position ? "排队第 " + job.queue_position + " 位" : "等待中");
      if (quality) { parts.push(quality); }
    } else if (job.state === "created") {
      parts.push("素材还没有开始处理");
    } else if (job.state === "failed") {
      parts.push(firstLine(job.error || job.message) || "出错了");
    } else {
      if (quality) { parts.push(quality); }
      if (job.state === "done" && typeof job.elapsed_seconds === "number") {
        parts.push("用时 " + fmtDuration(job.elapsed_seconds));
      }
      var when = fmtTime(job.finished_at || job.updated_at);
      if (when) { parts.push(when); }
    }
    return parts.join(" · ");
  }

  function renderJobs(data) {
    var list = $("job-list");
    var seen = {};
    data.jobs.forEach(function (job, index) {
      var row = state.rows.get(job.job);
      if (!row) { row = makeJobRow(job.job); state.rows.set(job.job, row); }
      setBadge(row.badge, job.state);
      setText(row.sub, jobSummary(job));
      setProgress(row.progress, row.bar, job);
      row.progress.hidden = job.state !== "running";
      if (list.children[index] !== row.li) { list.insertBefore(row.li, list.children[index] || null); }
      seen[job.job] = true;
    });
    state.rows.forEach(function (row, name) {
      if (!seen[name]) { row.li.remove(); state.rows.delete(name); }
    });
    $("jobs-empty").hidden = data.jobs.length > 0;
  }

  function refreshJobs() {
    return api("GET", "/api/jobs").then(function (data) {
      renderJobs(data);
      return data.jobs.some(function (job) { return job.state === "queued" || job.state === "running"; });
    });
  }

  // ------------------------------------------------------------------ job detail

  function has(job, action) { return (job.actions || []).indexOf(action) >= 0; }

  function renderStages(job) {
    var list = $("d-stages");
    if (!list.children.length) {
      STAGES.forEach(function (stage) { list.appendChild(el("li", "stage", stage[1])); });
    }
    var current = stageIndex(job.stage);
    for (var i = 0; i < STAGES.length; i++) {
      var cls = "stage";
      if (job.state === "done") { cls += " is-done"; }
      else if (current >= 0 && job.state !== "queued" && job.state !== "created") {
        if (i < current) { cls += " is-done"; }
        else if (i === current) {
          if (job.state === "running") { cls += " is-current"; }
          else if (job.state === "failed") { cls += " is-failed"; }
        }
      }
      if (list.children[i].className !== cls) { list.children[i].className = cls; }
      if (i === current && job.state === "running") { list.children[i].setAttribute("aria-current", "step"); }
      else { list.children[i].removeAttribute("aria-current"); }
    }
  }

  function renderMeta(job) {
    var parts = [];
    if (QUALITY_LABEL[job.quality]) { parts.push("质量：" + QUALITY_LABEL[job.quality]); }
    var options = job.options || {};      // the switches chosen for this scan, if any
    if (options.subject_only === true) { parts.push("类型：物体（只保留主体）"); }
    else if (options.subject_only === false) { parts.push("类型：场景"); }
    if (options.masks === true) { parts.push("抠除人物：开"); }
    else if (options.masks === false) { parts.push("抠除人物：关"); }
    var elapsed = job.elapsed_seconds;
    if (typeof elapsed === "number") {
      if (job.state === "running") { elapsed += (Date.now() - state.detailAt) / 1000; }
      parts.push((job.state === "running" ? "已用时 " : "用时 ") + fmtDuration(elapsed));
    }
    if (job.state === "queued" && job.queue_position) { parts.push("排队第 " + job.queue_position + " 位"); }
    if (job.state === "created" && job.uploads) {
      parts.push("已上传 " + job.uploads.count + " 个文件（" + fmtBytes(job.uploads.bytes) + "）");
    }
    var started = fmtTime(job.started_at);
    if (started) { parts.push("开始于 " + started); }
    if (job.input_path && job.source === "path") { parts.push("素材：" + job.input_path); }
    setText($("d-meta"), parts.join(" · "));
  }

  function renderOutputs(job) {
    var files = job.output_files || [];
    var show = job.state === "done" && files.length > 0;
    $("d-outputs").hidden = !show;
    if (!show) { return; }
    var signature = files.map(function (f) { return f.path + ":" + f.size + (f.outdated ? ":old" : ""); }).join("|");
    var list = $("d-output-list");
    if (list.getAttribute("data-signature") !== signature) {
      list.setAttribute("data-signature", signature);
      list.textContent = "";
      files.forEach(function (file) {
        var row = el("li", "output-row");
        var label = OUTPUT_LABEL[file.key] || OUTPUT_NAME_LABEL[file.name] || file.name;
        row.appendChild(el("span", "output-label", file.outdated ? OUTPUT_OUTDATED_LABEL : label));
        row.appendChild(el("span", "output-sub", file.name + " · " + fmtBytes(file.size)));
        var link = el("a", "btn small", "下载");
        link.href = jobUrl(job.job, "/download/" + file.path.split("/").map(encodeURIComponent).join("/"));
        link.setAttribute("download", "");
        row.appendChild(link);
        list.appendChild(row);
      });
    }
    $("btn-open-folder").hidden = !has(job, "open_folder");
    $("btn-open-viewer").hidden = !has(job, "open_viewer");
  }

  function renderLogFiles(job) {
    var select = $("d-log-file");
    var names = (job.logs && job.logs.length) ? job.logs : ["pipeline.log"];
    var signature = names.join("|");
    if (select.getAttribute("data-signature") === signature) { return; }
    select.setAttribute("data-signature", signature);
    select.textContent = "";
    names.forEach(function (name) {
      var option = el("option", null, name);
      option.value = name;
      select.appendChild(option);
    });
    // The main log unless the user picked another one (the list grows while a job runs).
    var wanted = state.logChoice && state.logChoice.job === job.job ? state.logChoice.file : "pipeline.log";
    select.value = names.indexOf(wanted) >= 0 ? wanted : names[0];
    select.hidden = names.length < 2;
  }

  function renderDetail(job) {
    state.detail = job;
    state.detailAt = Date.now();
    setText($("d-name"), job.job);
    setBadge($("d-state"), job.state);
    renderMeta(job);
    renderStages(job);

    setProgress($("d-progress"), $("d-bar"), job);
    $("d-progress-row").hidden = job.state !== "running" && job.state !== "done";
    setText($("d-percent"), job.state === "running" && typeof job.stage_progress === "number"
      ? Math.round(100 * Math.max(0, Math.min(1, job.stage_progress))) + "%" : "");

    var message = job.message || "";
    if (job.cancelling) {
      message = "正在取消，等待当前步骤停下来…";
    } else if (job.state === "queued") {
      message = job.queue_position > 1 ? "排队中，前面还有 " + (job.queue_position - 1) + " 个任务等待。"
                                       : "排队中，等待当前任务结束。";
      if (!job.queue_position) { message = "即将开始…"; }
    } else if (job.state === "created") {
      message = job.uploads && job.uploads.count ? "素材已上传，还没有开始处理。" : "还没有上传素材。";
    } else if (job.state === "failed") {
      message = "";
    }
    setText($("d-message"), message);
    $("d-message").hidden = !message;

    var failed = job.state === "failed";
    $("d-error").hidden = !failed;
    if (failed) {
      var text = job.error || job.message || "没有错误信息，请查看下面的运行日志。";
      if (job.error === "interrupted") { text = "任务被中断：扫描程序在任务完成前被关闭或重新启动了。可以点“重试”从中断的地方继续。"; }
      setText($("d-error-text"), text);
    }

    var warnings = Array.isArray(job.warnings) ? job.warnings : [];
    var warningList = $("d-warnings");
    var warningSignature = warnings.join("\n");
    if (warningList.getAttribute("data-signature") !== warningSignature) {
      warningList.setAttribute("data-signature", warningSignature);
      warningList.textContent = "";
      warnings.forEach(function (text) { warningList.appendChild(el("li", null, String(text))); });
    }
    warningList.hidden = warnings.length === 0;

    renderOutputs(job);
    window.scan3dJobView.update(job, state.route.tab);

    var start = $("btn-job-start");
    start.hidden = !has(job, "start") || (job.state === "created" && !(job.uploads && job.uploads.count) && job.source === "upload");
    setText(start, job.state === "created" ? "开始扫描" : "重试");
    $("btn-job-cancel").hidden = !has(job, "cancel") || !!job.cancelling;
    $("btn-job-delete").hidden = !has(job, "delete");
    $("btn-job-folder").hidden = !has(job, "open_folder") || !$("d-outputs").hidden;

    renderLogFiles(job);
    if (failed && state.logOpenedFor !== job.job) {
      state.logOpenedFor = job.job;   // show the log once; the user may close it again
      $("d-log").open = true;
    }
  }

  function refreshDetail() {
    var name = state.route.job;
    return api("GET", jobUrl(name)).then(function (job) {
      if (state.route.view !== "detail" || state.route.job !== name) { return false; }
      renderDetail(job);
      var active = job.state === "queued" || job.state === "running";
      // A log that cannot be read must not look like a lost connection.
      return refreshLog().then(function () { return active; }, function () { return active; });
    }, function (err) {
      if (err.status === 404 || err.status === 400) {
        if (state.route.job === name) { location.hash = "#/"; }
        return false;
      }
      throw err;
    });
  }

  // Tools redraw one console line with "\r"; show only the final form of each line.
  function cleanLog(raw) {
    var lines = raw.split("\n");
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].replace(/\r+$/, "");
      var cr = line.lastIndexOf("\r");
      lines[i] = cr >= 0 ? line.slice(cr + 1) : line;
    }
    return lines.join("\n");
  }

  function refreshLog() {
    var details = $("d-log");
    var job = state.detail;
    if (!details.open || !job) { return Promise.resolve(); }
    var file = $("d-log-file").value || "pipeline.log";
    var log = state.log;
    if (log.job !== job.job || log.file !== file) {
      log = state.log = { job: job.job, file: file, offset: null, raw: "" };
      $("d-log-text").textContent = "";
      setText($("d-log-note"), "");
    }
    var url = jobUrl(job.job, "/log?file=" + encodeURIComponent(file));
    if (log.offset !== null) { url += "&offset=" + log.offset; }
    return api("GET", url).then(function (chunk) {
      if (state.log !== log) { return; }
      var first = log.offset === null;
      if (!first && chunk.offset !== log.offset) { log.raw = ""; first = true; }  // the file was replaced
      log.offset = chunk.next_offset;
      if (!chunk.text && !first) { return; }
      log.raw += chunk.text;
      var cut = chunk.truncated;
      if (log.raw.length > LOG_KEEP_CHARS) {
        log.raw = log.raw.slice(log.raw.indexOf("\n", log.raw.length - LOG_KEEP_CHARS * 0.8) + 1);
        cut = true;
      }
      if (cut) { log.cut = true; }
      var pre = $("d-log-text");
      var atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
      pre.textContent = cleanLog(log.raw);
      if (atBottom || first) { pre.scrollTop = pre.scrollHeight; }
      setText($("d-log-note"), log.cut ? "只显示最后一部分" : "");
    });
  }

  function jobAction(method, rest, body) {
    var job = state.detail;
    if (!job) { return Promise.resolve(); }
    showError("d-action-error", "");
    return api(method, jobUrl(job.job, rest), body).then(function (result) {
      poll();
      return result;
    }, function (err) {
      showError("d-action-error", err.message);
      poll();
    });
  }

  // First click arms the button, a second click within a few seconds runs the action.
  function confirmButton(button, armedText, action) {
    var timer = null, label = "";
    function disarm() {
      clearTimeout(timer);
      timer = null;
      button.classList.remove("is-armed");
      button.textContent = label;
    }
    button.addEventListener("click", function () {
      if (timer !== null) { disarm(); action(); return; }
      label = button.textContent;
      button.textContent = armedText;
      button.classList.add("is-armed");
      timer = setTimeout(disarm, 4000);
    });
  }

  // ------------------------------------------------------------------ polling and routing

  var pollTimer = null, pollRunning = false, pollAgain = false;

  function poll() {
    clearTimeout(pollTimer);
    if (pollRunning) { pollAgain = true; return; }
    if (document.hidden) { return; }  // resumed by visibilitychange
    pollRunning = true;
    var work = state.info ? Promise.resolve() : loadInfo();
    work.then(function () {
      if (state.route.view === "detail") { return refreshDetail(); }
      return Promise.all([refreshJobs(), refreshModels()]).then(function (active) {
        return active[0] || active[1];      // a job or a model download is under way
      });
    }).then(function (active) {
      hideBanner("connection");
      hideBanner("auth");
      return active ? POLL_ACTIVE_MS : POLL_IDLE_MS;
    }, function (err) {
      if (err.status === 401 || err.status === 403) {
        showBanner("auth", err.message || "没有访问权限。");
        return POLL_IDLE_MS;
      }
      if (err.status === 429) { showBanner("auth", err.message); return 15000; }
      showBanner("connection", "连接不上电脑上的扫描程序，正在重试…");
      return POLL_ERROR_MS;
    }).then(function (delay) {
      pollRunning = false;
      if (pollAgain) { pollAgain = false; poll(); return; }
      pollTimer = setTimeout(poll, delay);
    });
  }

  // ------------------------------------------------------------------ "open on a phone" panel

  var qrScript = null;

  // The QR encoder is a local file (vendor/qrcode-generator); only this PC ever needs it.
  function loadQrEncoder() {
    if (!qrScript) {
      qrScript = new Promise(function (resolve, reject) {
        var script = document.createElement("script");
        script.src = "/static/vendor/qrcode-generator/qrcode.js";
        script.onload = function () { resolve(window.qrcode); };
        script.onerror = function () { qrScript = null; reject(new Error("qr")); };
        document.head.appendChild(script);
      });
    }
    return qrScript;
  }

  function drawQr(holder, text) {
    holder.textContent = "";
    return loadQrEncoder().then(function (encoder) {
      var qr = encoder(0, "M");
      qr.addData(text);
      qr.make();
      var count = qr.getModuleCount(), quiet = 4;
      var scale = Math.max(2, Math.floor(720 / (count + 2 * quiet)));
      var canvas = document.createElement("canvas");
      canvas.width = canvas.height = (count + 2 * quiet) * scale;
      var ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ffffff";      // always dark on white: phone cameras expect it
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#000000";
      for (var r = 0; r < count; r++) {
        for (var c = 0; c < count; c++) {
          if (qr.isDark(r, c)) { ctx.fillRect((c + quiet) * scale, (r + quiet) * scale, scale, scale); }
        }
      }
      holder.textContent = "";
      holder.appendChild(canvas);
      holder.setAttribute("data-url", text);
    }, function () {
      holder.textContent = "二维码组件加载失败，请直接输入地址。";
    });
  }

  function copyText(text) {
    function oldWay() {
      return new Promise(function (resolve, reject) {
        var area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "");
        area.className = "visually-hidden";
        document.body.appendChild(area);
        area.select();
        var ok = false;
        try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
        area.remove();
        if (ok) { resolve(); } else { reject(new Error("copy")); }
      });
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).catch(oldWay);
    }
    return oldWay();
  }

  function renderPhonePanel(urls) {
    var list = $("lan-urls");
    list.textContent = "";
    $("lan-body").hidden = urls.length === 0;
    if (!urls.length) {
      setText($("lan-text"), "没有找到这台电脑的局域网地址。把电脑连上 Wi-Fi 或网线后刷新本页面，这里会显示手机要打开的地址。");
      return;
    }
    setText($("lan-text"), "手机和这台电脑连同一个 Wi-Fi，扫二维码或打开下面的地址，就能在手机上上传素材、查看进度和结果。");
    function choose(url, row) {
      for (var i = 0; i < list.children.length; i++) { list.children[i].classList.toggle("is-chosen", list.children[i] === row); }
      drawQr($("lan-qr"), url);
    }
    urls.forEach(function (url, index) {
      var row = el("li", "lan-url");
      row.appendChild(el("code", "lan-url-text", url));
      var copy = el("button", "btn small", "复制");
      copy.type = "button";
      copy.addEventListener("click", function () {
        copyText(url).then(function () { copy.textContent = "已复制"; }, function () { copy.textContent = "请手动复制"; })
          .then(function () { setTimeout(function () { copy.textContent = "复制"; }, 2000); });
      });
      row.appendChild(copy);
      if (urls.length > 1) {
        var pick = el("button", "btn small", "二维码");
        pick.type = "button";
        pick.addEventListener("click", function () { choose(url, row); });
        row.appendChild(pick);
      }
      list.appendChild(row);
      if (index === 0) { choose(url, row); }
    });
    if (urls.length > 1) {
      setText($("lan-text"), $("lan-text").textContent + "这台电脑有多个网络地址，通常第一个就是；打不开时换下一个。");
    }
  }

  function loadInfo() {
    return api("GET", "/api/info").then(function (info) {
      state.info = info;
      $("source-tabs").hidden = !info.loopback;
      if (!info.loopback && state.mode === "path") { setMode("upload"); }
      // Only the browser on the PC itself is told the phone addresses (they carry the token).
      $("lan-card").hidden = !info.loopback;
      if (info.loopback) { renderPhonePanel(info.lan_urls || []); }
    });
  }

  function applyRoute() {
    // #/job/<name> or #/job/<name>/view/<tab> (report, splat, mesh, collision, overlay, coverage)
    var match = /^#\/job\/([^/?#]+)(?:\/view\/([a-z]+))?\/?$/.exec(location.hash);
    var job = null;
    if (match) {
      try { job = decodeURIComponent(match[1]); } catch (e) { job = null; }
      if (job && !JOB_NAME_RE.test(job)) { job = null; }
    }
    var previous = state.route.job;
    state.route = job ? { view: "detail", job: job, tab: match[2] || null } : { view: "home", job: null, tab: null };
    $("view-home").hidden = !!job;
    $("view-detail").hidden = !job;
    if (job !== previous) { window.scan3dJobView.close(); }    // frees the 3D view's GPU memory
    if (job) {
      if (!state.detail || state.detail.job !== job) {
        state.detail = null;
        setText($("d-name"), job);
        $("d-state").className = "badge";
        ["d-state", "d-meta", "d-message"].forEach(function (id) { setText($(id), ""); });
        ["d-error", "d-warnings", "d-outputs", "d-viewer", "btn-job-start", "btn-job-cancel", "btn-job-delete", "btn-job-folder", "d-progress-row"]
          .forEach(function (id) { $(id).hidden = true; });
        $("d-log").open = false;
        $("d-log-text").textContent = "";
        $("d-log-file").removeAttribute("data-signature");
        $("d-stages").textContent = "";
        state.log = { job: null, file: null, offset: null, raw: "" };
      } else {
        window.scan3dJobView.update(state.detail, state.route.tab);   // a tab was chosen
      }
      showError("d-action-error", "");
      if (job !== previous) { window.scrollTo(0, 0); }
    }
    poll();
  }

  // ------------------------------------------------------------------ wiring

  function init() {
    $("btn-start").addEventListener("click", onStart);
    $("btn-abort").addEventListener("click", onAbort);
    $("tab-upload").addEventListener("click", function () { setMode("upload"); });
    $("tab-path").addEventListener("click", function () { setMode("path"); });
    $("btn-clear-files").addEventListener("click", function () {
      if (uploadLocked()) { return; }
      state.items = [];
      $("file-input").value = "";
      renderFiles();
    });
    $("file-input").addEventListener("change", function (event) {
      addFiles(event.target.files);
      event.target.value = "";   // choosing the same file again must fire "change" again
    });

    var zone = $("dropzone"), depth = 0;
    zone.addEventListener("dragenter", function (event) {
      event.preventDefault(); depth++; zone.classList.add("is-over");
    });
    zone.addEventListener("dragleave", function () {
      depth = Math.max(0, depth - 1);
      if (!depth) { zone.classList.remove("is-over"); }
    });
    zone.addEventListener("drop", function (event) {
      event.preventDefault();
      depth = 0;
      zone.classList.remove("is-over");
      if (event.dataTransfer) { addFiles(event.dataTransfer.files); }
    });
    // A file dropped beside the drop zone must not make the browser navigate to it.
    window.addEventListener("dragover", function (event) { event.preventDefault(); });
    window.addEventListener("drop", function (event) { event.preventDefault(); });

    $("local-path").addEventListener("keydown", function (event) {
      if (event.key === "Enter") { onStart(); }
    });

    $("opt-masks").addEventListener("change", function () {
      state.masksTouched = true;
      renderOptions();
    });
    var choices = document.querySelectorAll('input[name="scan-kind"], input[name="quality"]');
    for (var i = 0; i < choices.length; i++) { choices[i].addEventListener("change", renderOptions); }
    // Whether the card is open by the user's own choice (it then stays open, see renderModels).
    $("models-card").querySelector("summary").addEventListener("click", function () {
      state.modelsOpenedByUser = !$("models-card").open;
    });

    $("btn-job-start").addEventListener("click", function () { jobAction("POST", "/start", {}); });
    confirmButton($("btn-job-cancel"), "确认取消？", function () { jobAction("POST", "/cancel", {}); });
    confirmButton($("btn-job-delete"), "确认删除？", function () {
      jobAction("DELETE", "").then(function (result) {
        if (result) { state.detail = null; location.hash = "#/"; }
      });
    });
    function openFolder() { jobAction("POST", "/open-folder", {}); }
    $("btn-open-folder").addEventListener("click", openFolder);
    $("btn-job-folder").addEventListener("click", openFolder);
    $("btn-open-viewer").addEventListener("click", function () { jobAction("POST", "/open-viewer", {}); });

    $("d-log").addEventListener("toggle", function () { refreshLog().catch(function () {}); });
    $("d-log-file").addEventListener("change", function () {
      if (state.detail) { state.logChoice = { job: state.detail.job, file: $("d-log-file").value }; }
      refreshLog().catch(function () {});
    });

    setInterval(function () {   // keep the running clock moving between polls
      if (state.route.view === "detail" && state.detail && state.detail.state === "running") {
        renderMeta(state.detail);
      }
    }, 1000);

    window.addEventListener("hashchange", applyRoute);
    document.addEventListener("visibilitychange", function () { if (!document.hidden) { poll(); } });
    window.addEventListener("online", poll);
    window.addEventListener("beforeunload", function (event) {
      if (state.upload && state.upload.running) { event.preventDefault(); event.returnValue = ""; }
    });

    renderFiles();
    renderForm();
    applyRoute();
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", init); }
  else { init(); }
})();
