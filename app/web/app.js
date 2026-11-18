/* scan3d UI: the page itself. Task list, new task (upload or local path), task page (header,
 * overview, logs), models panel, phone pairing, polling and routing. The workflow graph and the
 * node panel are in workflow.js, the result viewer in jobview.js, shared helpers in common.js.
 * Talks to the JSON API in app/server.py. Everything put on the page from data goes through
 * textContent, never innerHTML. */
(function () {
  "use strict";

  var S3 = window.S3;
  var $ = S3.$, el = S3.el, setText = S3.setText, api = S3.api, jobUrl = S3.jobUrl;
  var fmtBytes = S3.fmtBytes, showError = S3.showError;

  var OUTPUT_LABEL = {
    splat: "高斯点云", mesh: "网格模型", collision: "碰撞网格", report: "扫描报告",
    preview: "网页预览（完整）", preview_lite: "网页预览（轻量）", coverage: "拍摄覆盖"
  };
  var OUTPUT_NAME_LABEL = {     // files in output/ that status.json does not name
    "scene_transform.json": "场景变换", "preview.spz": "网页预览（完整）", "preview-lite.spz": "网页预览（轻量）",
    "coverage.json": "拍摄覆盖"
  };
  var OUTPUT_OUTDATED_LABEL = "网页预览（旧版，已不使用）";
  var EXTENSIONS = ["jpg", "jpeg", "png", "webp", "heic", "heif", "tif", "tiff", "bmp",
                    "mp4", "mov", "m4v", "mkv", "avi", "webm"];
  var JOB_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

  var POLL_ACTIVE_MS = 1500;      // while something is queued or running
  var POLL_IDLE_MS = 6000;
  var POLL_ERROR_MS = 3000;
  var UPLOAD_STALL_MS = 60000;    // no progress for this long: abort the request and retry
  var RETRY_DELAYS_MS = [1000, 2000, 4000, 8000, 10000, 10000, 10000, 10000];
  var LOG_KEEP_CHARS = 300000;

  var state = S3.state = {
    info: null,
    route: { view: "home", job: null, tab: null },
    mode: "upload",        // "upload" | "path"
    items: [],             // files chosen for the next task
    upload: null,          // { job, running, aborted, xhr, startedAt, startedBytes }
    busy: false,           // a create request for a local path is in flight
    jobs: [],              // last answer of /api/jobs
    detail: null,          // last view of the job shown in the task page
    detailAt: 0,
    log: { job: null, file: null, offset: null, raw: "" },
    logOpenedFor: null,
    logChoice: null,
    rows: new Map(),
    models: null,          // last answer of /api/models
    modelsProblem: "",
    template: null,        // template chosen for the next task
    startNow: true
  };

  // ------------------------------------------------------------------ banner

  var bannerKind = null;

  function showBanner(kind, text) {
    bannerKind = kind;
    setText($("banner"), text);
    $("banner").hidden = false;
  }

  function hideBanner(kind) {
    if (bannerKind === kind || kind === undefined) {
      bannerKind = null;
      $("banner").hidden = true;
    }
  }

  // ------------------------------------------------------------------ schema, templates, models

  function loadSchema() {
    if (S3.schema) { return Promise.resolve(); }
    return api("GET", "/api/workflow/schema").then(function (schema) {
      S3.schema = schema;
      return loadTemplates();
    });
  }

  function loadTemplates() {
    return api("GET", "/api/templates").then(function (info) {
      S3.templatesInfo = info;
    }, function () { /* the schema carries the built-in ones too */ }).then(function () {
      renderTemplatePicker();
      if (S3.wf) { S3.wf.templatesChanged(); }
    });
  }

  function templateList() {
    var info = S3.templatesInfo;
    if (info && info.templates) { return info.templates; }
    return (S3.schema && S3.schema.templates) || [];
  }

  function defaultTemplate() {
    var info = S3.templatesInfo;
    return (info && info.default_template) || (S3.schema && S3.schema.default_template) || null;
  }

  // "做：抠图、法线先验 …" for a template: the optional stages it switches on (the server's `enabled` list)
  S3.templateStages = function (t) {
    if (!t || !Array.isArray(t.enabled)) { return ""; }
    var optional = {};
    ((S3.schema && S3.schema.stages) || []).forEach(function (s) { if (s.optional) { optional[s.id] = true; } });
    var names = t.enabled.filter(function (id) { return optional[id]; }).map(S3.stageLabel);
    return "会运行：" + (names.length ? "基础步骤 + " + names.join("、") : "只有基础步骤（整理素材、相机位姿、导出）");
  };

  function loadModels() {
    return api("GET", "/api/models").then(function (m) {
      state.models = m;
      state.modelsProblem = "";
      renderModels();
      if (S3.wf) { S3.wf.modelsChanged(); }
      return m;
    }, function (err) {
      state.modelsProblem = err.message;
      renderModels();
    });
  }

  function renderModels() {
    var card = $("models-card"), m = state.models;
    if (!m) {
      card.hidden = !state.modelsProblem;
      setText($("models-summary"), state.modelsProblem);
      return;
    }
    card.hidden = false;
    var features = m.features || [];
    var missing = features.filter(function (f) { return !f.ready; });
    setText($("models-summary"), missing.length
      ? missing.length + " 项模型还没下载（用到时会自动下载）" : "模型都已下载");
    var list = $("model-list");
    list.textContent = "";
    var loopback = state.info && state.info.loopback;
    features.forEach(function (f) {
      var row = el("li", "model-row");
      var head = el("div", "model-head");
      var name = el("span", "model-name", f.label || f.id);
      head.appendChild(name);
      var missingMb = 0;
      (f.files || []).forEach(function (file) { if (!file.installed) { missingMb += file.mb || 0; } });
      head.appendChild(el("span", "badge " + (f.ready ? "s-done" : "s-queued"),
        f.ready ? "已下载" : "还没下载 约 " + S3.fmtMb(missingMb)));
      row.appendChild(head);
      var bits = [];
      if (f.stage) { bits.push("用于「" + S3.stageLabel(f.stage) + "」"); }
      var sizes = (f.files || []).map(function (file) { return file.name + "（" + S3.fmtMb(file.mb) + (file.installed ? "，已下载" : "") + "）"; });
      if (sizes.length) { bits.push(sizes.join("、")); }
      row.appendChild(el("p", "hint", bits.join(" · ")));
      var lic = f.licence;
      if (lic) {
        var line = el("p", "hint", "协议：");
        if (lic.url) {
          var link = el("a", null, lic.label || lic.id);
          link.href = lic.url; link.target = "_blank"; link.rel = "noopener noreferrer";
          line.appendChild(link);
        } else { line.appendChild(document.createTextNode(lic.label || lic.id)); }
        line.appendChild(document.createTextNode("。打开用到它的功能就是同意这个协议。"));
        row.appendChild(line);
      }
      if (!f.ready && loopback) {
        var btn = el("button", "btn small", "现在下载");
        btn.type = "button";
        btn.addEventListener("click", function () {
          btn.disabled = true;
          setText(btn, "已开始下载…");
          api("POST", "/api/models/download", { features: [f.id] }).then(loadModels, function (err) {
            showError("models-error", err.message);
            btn.disabled = false;
            setText(btn, "现在下载");
          });
        });
        row.appendChild(btn);
      }
      list.appendChild(row);
    });
    if (typeof m.disk_mb === "number") {
      setText($("models-disk"), "模型文件夹占用 " + S3.fmtMb(m.disk_mb) + "。");
    }
    $("models-remote").hidden = !!loopback;
  }

  // ------------------------------------------------------------------ new task: template picker

  function renderTemplatePicker() {
    var holder = $("tpl-picker");
    var list = templateList();
    if (!list.length) { holder.hidden = true; return; }
    holder.hidden = false;
    var chosen = state.template && list.some(function (t) { return t.id === state.template; })
      ? state.template : defaultTemplate() || list[0].id;
    state.template = chosen;
    var box = $("tpl-options");
    box.textContent = "";
    list.forEach(function (t) {
      var label = el("label", "quality-option");
      var input = document.createElement("input");
      input.type = "radio";
      input.name = "template";
      input.value = t.id;
      input.checked = t.id === chosen;
      input.disabled = uploadLocked() || state.busy;
      input.addEventListener("change", function () { state.template = t.id; });
      var body = el("span", "quality-body");
      var name = el("span", "quality-name", t.label);
      if (t.id === defaultTemplate()) { name.appendChild(el("span", "tag", "默认")); }
      body.appendChild(name);
      if (t.desc) { body.appendChild(el("span", "quality-desc", t.desc)); }
      var runs = S3.templateStages(t);
      if (runs) { body.appendChild(el("span", "quality-desc quality-runs", runs)); }
      label.appendChild(input);
      label.appendChild(body);
      box.appendChild(label);
    });
  }

  // ------------------------------------------------------------------ new task: choosing files

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
    S3.setClass(row.li, cls);
    row.bar.hidden = !showBar;
    row.fill.style.width = (100 * item.sent / Math.max(1, item.size)).toFixed(1) + "%";
  }

  // ------------------------------------------------------------------ new task: uploading

  function renderForm() {
    var up = state.upload;
    var running = !!(up && up.running);
    var paused = !!(up && up.job && !up.running);
    var locked = running || state.busy;
    $("btn-start").disabled = locked;
    setText($("btn-start"), paused ? "继续上传" : (running ? "正在上传…" : ($("opt-start").checked ? "上传并运行" : "上传")));
    if (state.mode === "path") { setText($("btn-start"), locked ? "正在创建…" : ($("opt-start").checked ? "创建并运行" : "创建")); }
    $("btn-abort").hidden = !(running || paused);
    setText($("btn-abort"), paused ? "放弃并删除" : "取消上传");
    $("file-input").disabled = uploadLocked();
    $("job-name").disabled = locked || paused;
    $("local-path").disabled = locked;
    $("opt-start").disabled = locked;
    $("tab-upload").disabled = uploadLocked() || state.busy;
    $("tab-path").disabled = uploadLocked() || state.busy;
    var radios = document.querySelectorAll('input[name="template"]');
    for (var i = 0; i < radios.length; i++) { radios[i].disabled = locked || paused; }
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
        var left = S3.fmtRemaining((total - sent) / speed);
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
        finish(reject, S3.apiError((data && data.message) || ("上传失败（" + xhr.status + "）"), xhr.status, data));
      };
      xhr.onerror = function () { finish(reject, S3.apiError("网络中断", 0, null)); };
      xhr.onabort = function () { finish(reject, S3.apiError(stalled ? "上传长时间没有进展" : "已取消", 0, null)); };
      xhr.send(offset ? item.file.slice(offset) : item.file);
    });
  }

  function uploadItem(up, item) {
    var attempt = 0;
    function tryOnce() {
      if (up.aborted) { return Promise.reject(S3.apiError("已取消", 0, null)); }
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
        return S3.sleep(RETRY_DELAYS_MS[attempt++]).then(tryOnce);
      });
    }
    return tryOnce();
  }

  function runUpload(name, template, startNow) {
    var up = state.upload || (state.upload = { job: null });
    up.running = true;
    up.aborted = false;
    showError("form-error", "");
    renderForm();

    var created = up.job ? Promise.resolve() : api("POST", "/api/jobs", { name: name || null, template: template })
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
      return startNow ? api("POST", jobUrl(up.job, "/run"), {}) : null;
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
        return S3.sleep(1500).then(attempt);
      });
    }
    return attempt().then(function () { poll(); });
  }

  function onStart() {
    var name = $("job-name").value.trim();
    var template = state.template;
    var startNow = $("opt-start").checked;
    showError("form-error", "");
    if (name && !JOB_NAME_RE.test(name)) {
      showError("form-error", "名称只能包含英文字母、数字、- 和 _，以字母或数字开头，最长 64 个字符。");
      return;
    }
    if (!template) { showError("form-error", "还没有读到模板列表，请稍等一下再试。"); return; }
    if (state.mode === "path") {
      var path = $("local-path").value.trim();
      if (!path) { showError("form-error", "请填写文件夹或视频文件的完整路径。"); return; }
      state.busy = true;
      renderForm();
      api("POST", "/api/jobs", { name: name || null, template: template, path: path, start: startNow })
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
    runUpload(name, template, startNow);
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
    renderForm();
  }

  // ------------------------------------------------------------------ task list

  function stageProgressOf(job) {
    var s = job.stages && job.stage && job.stages[job.stage];
    if (s && typeof s.progress === "number") { return s.progress; }
    return typeof job.stage_progress === "number" ? job.stage_progress : null;
  }

  function stageCounts(job) {
    var total = 0, done = 0;
    Object.keys(job.stages || {}).forEach(function (id) {
      var st = job.stages[id].state;
      if (st === "disabled") { return; }
      total++;
      if (st === "done") { done++; }
    });
    return { done: done, total: total };
  }

  function failedStage(job) {
    var found = null;
    Object.keys(job.stages || {}).forEach(function (id) { if (!found && job.stages[id].state === "failed") { found = id; } });
    return found;
  }

  function jobSummary(job) {
    var parts = [];
    var counts = stageCounts(job);
    if (job.state === "running") {
      var label = S3.stageLabel(job.stage);
      var p = stageProgressOf(job);
      if (label) { parts.push(p !== null ? label + " " + Math.round(100 * S3.clamp01(p)) + "%" : label); }
      if (typeof job.elapsed_seconds === "number") { parts.push("已用时 " + S3.fmtDuration(job.elapsed_seconds)); }
    } else if (job.state === "queued") {
      parts.push(job.queue_position ? "排队第 " + job.queue_position + " 位" : "等待中");
    } else if (job.state === "created") {
      parts.push(job.uploads && job.uploads.count ? "素材已上传，还没运行" : "素材还没有开始处理");
    } else if (job.state === "failed") {
      var f = failedStage(job);
      var line = S3.firstLine(job.error || job.message) || "出错了";
      parts.push(f && line.indexOf(S3.stageLabel(f)) < 0 ? "「" + S3.stageLabel(f) + "」失败：" + line : line);
    } else if (job.state === "cancelled") {
      parts.push("已取消");
    } else {
      if (counts.total) { parts.push("完成 " + counts.done + "/" + counts.total + " 步"); }
      var when = S3.fmtTime(job.finished_at || job.updated_at);
      if (when) { parts.push(when); }
    }
    return parts.join(" · ");
  }

  function setBadge(node, jobState) {
    var label = S3.JOB_STATE[jobState] || jobState || "";
    setText(node, label);
    S3.setClass(node, "badge s-" + jobState);
  }

  function makeJobRow(name) {
    var li = el("li", "job-item");
    var link = el("a", "job-link");
    link.href = "#/job/" + encodeURIComponent(name);
    var row = {
      li: li, link: link,
      name: el("span", "job-name", name),
      badge: el("span", "badge"),
      sub: el("span", "job-sub"),
      dots: el("span", "dots"),
      progress: el("div", "progress"),
      bar: el("div", "progress-bar"),
      act: null, actKind: null, dotEls: []
    };
    row.dots.setAttribute("aria-hidden", "true");
    row.progress.appendChild(row.bar);
    link.appendChild(row.name);
    link.appendChild(row.badge);
    link.appendChild(row.sub);
    link.appendChild(row.dots);
    link.appendChild(row.progress);
    li.appendChild(link);
    return row;
  }

  function updateDots(row, job) {
    var schema = (S3.schema && S3.schema.stages) || [];
    if (row.dotEls.length !== schema.length) {
      row.dots.textContent = "";
      row.dotEls = schema.map(function () { var d = el("i", "dot"); row.dots.appendChild(d); return d; });
    }
    schema.forEach(function (s, i) {
      var st = (job.stages && job.stages[s.id] && job.stages[s.id].state) || "";
      var cls = "dot" + (st ? " s-" + st : "");
      S3.setClass(row.dotEls[i], cls);
      var tip = s.label + (st ? "：" + (S3.STAGE_STATE[st] ? S3.STAGE_STATE[st].word : st) : "");
      if (row.dotEls[i].title !== tip) { row.dotEls[i].title = tip; }
    });
  }

  function updateJobAction(row, job) {
    var kind = null;
    if (job.state === "running" || job.state === "queued") { kind = "stop"; }
    if (kind !== row.actKind) {
      if (row.act) { row.act.remove(); row.act = null; }
      row.actKind = kind;
      if (kind === "stop") {
        var b = el("button", "job-act");
        b.type = "button";
        b.appendChild(S3.icon("stop"));
        b.appendChild(el("span", "visually-hidden", "停止任务"));
        b.title = "停止这个任务";
        var timer = null;
        b.addEventListener("click", function () {
          if (timer === null) {      // first click arms
            b.classList.add("is-armed");
            b.title = "再点一次确认停止";
            timer = setTimeout(function () { timer = null; b.classList.remove("is-armed"); b.title = "停止这个任务"; }, 3500);
            return;
          }
          clearTimeout(timer); timer = null;
          api("POST", jobUrl(row.jobName, "/cancel"), {}).then(poll, function (err) { showBanner("auth", err.message); });
        });
        row.act = b;
        row.li.appendChild(b);
      }
    }
  }

  function renderJobs(data) {
    var list = $("job-list");
    var seen = {};
    state.jobs = data.jobs;
    data.jobs.forEach(function (job, index) {
      var row = state.rows.get(job.job);
      if (!row) { row = makeJobRow(job.job); state.rows.set(job.job, row); }
      row.jobName = job.job;
      setBadge(row.badge, job.state);
      setText(row.sub, jobSummary(job));
      updateDots(row, job);
      var p = stageProgressOf(job);
      S3.setClass(row.progress, "progress" + (job.state === "running" && p === null ? " is-indeterminate" : ""));
      row.bar.style.width = job.state === "running" && p !== null ? (100 * S3.clamp01(p)).toFixed(1) + "%" : "";
      row.progress.hidden = job.state !== "running";
      if (state.route.job === job.job) { row.link.setAttribute("aria-current", "page"); } else { row.link.removeAttribute("aria-current"); }
      S3.setClass(row.li, "job-item" + (state.route.job === job.job ? " is-current" : ""));
      updateJobAction(row, job);
      if (list.children[index] !== row.li) { list.insertBefore(row.li, list.children[index] || null); }
      seen[job.job] = true;
    });
    state.rows.forEach(function (row, name) {
      if (!seen[name]) { row.li.remove(); state.rows.delete(name); }
    });
    $("jobs-empty").hidden = data.jobs.length > 0;
    setText($("side-count"), data.jobs.length ? String(data.jobs.length) : "");
  }

  function refreshJobs() {
    return api("GET", "/api/jobs").then(function (data) {
      renderJobs(data);
      return data.jobs.some(function (job) { return job.state === "queued" || job.state === "running"; });
    });
  }

  // ------------------------------------------------------------------ task page

  function has(job, action) { return (job.actions || []).indexOf(action) >= 0; }
  function isBusy(job) { return job.state === "queued" || job.state === "running"; }

  function renderMeta(job) {
    var parts = [];
    var tpl = S3.wf ? null : null;
    void tpl;
    var elapsed = job.elapsed_seconds;
    if (typeof elapsed === "number") {
      if (job.state === "running") { elapsed += (Date.now() - state.detailAt) / 1000; }
      parts.push((job.state === "running" ? "已用时 " : "用时 ") + S3.fmtDuration(elapsed));
    }
    if (job.state === "queued" && job.queue_position) { parts.push("排队第 " + job.queue_position + " 位"); }
    if (job.uploads && job.uploads.count) {
      parts.push("已上传 " + job.uploads.count + " 个文件（" + fmtBytes(job.uploads.bytes) + "）");
    }
    var started = S3.fmtTime(job.started_at);
    if (started) { parts.push("开始于 " + started); }
    if (job.input_path && job.source === "path") { parts.push("素材：" + job.input_path); }
    setText($("d-meta"), parts.join(" · "));
  }

  function renderOutputs(job) {
    var files = job.output_files || [];
    var box = $("d-outputs");
    box.hidden = files.length === 0;
    if (!files.length) { return; }
    setText($("d-outputs-count"), "输出文件（" + files.length + "）");
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
      var option = el("option", null, logLabel(name));
      option.value = name;
      select.appendChild(option);
    });
    var wanted = state.logChoice && state.logChoice.job === job.job ? state.logChoice.file : "pipeline.log";
    select.value = names.indexOf(wanted) >= 0 ? wanted : names[0];
    select.hidden = names.length < 2;
  }

  function logLabel(name) {
    if (name === "pipeline.log") { return "总日志（pipeline.log）"; }
    var id = name.replace(/\.log$/, "");
    var label = S3.stageLabel(id);
    return label && label !== id ? label + "（" + name + "）" : name;
  }

  function renderDetail(job) {
    state.detail = job;
    state.detailAt = Date.now();
    setText($("d-name"), job.job);
    setBadge($("d-state"), job.state);
    renderMeta(job);

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
    } else if (job.state === "done" || job.state === "cancelled") {
      message = job.state === "cancelled" ? "已取消。做完的阶段保留着，再点「运行」会从没做完的地方接着做。" : "";
    }
    setText($("d-message"), message);
    $("d-message").hidden = !message;

    var failed = job.state === "failed";
    $("d-error").hidden = !failed;
    if (failed) {
      var text = job.error || job.message || "没有错误信息，请查看下面的运行日志。";
      if (job.error === "interrupted") { text = "任务被中断：扫描程序在任务完成前被关闭或重新启动了。点「运行」会从没做完的地方继续。"; }
      var f = failedStage(job);
      setText($("d-error-title"), f ? "「" + S3.stageLabel(f) + "」这一步失败了" : "运行失败");
      setText($("d-error-text"), text);
    }

    var warnings = Array.isArray(job.warnings) ? job.warnings : [];
    var warningList = $("d-warnings");
    var warningSignature = warnings.join("\n");
    if (warningList.getAttribute("data-signature") !== warningSignature) {
      warningList.setAttribute("data-signature", warningSignature);
      warningList.textContent = "";
      warnings.forEach(function (t) { warningList.appendChild(el("li", null, String(t))); });
    }
    warningList.hidden = warnings.length === 0;

    renderOutputs(job);
    if (S3.wf) { S3.wf.update(job); }
    window.scan3dJobView.update(job, state.route.tab);

    var busy = isBusy(job);
    var empty = job.state === "created" && !(job.uploads && job.uploads.count) && job.source === "upload";
    var run = $("btn-job-start");
    run.hidden = busy || empty;
    setText(run, job.state === "created" ? "运行" : "运行");
    var stop = $("btn-job-cancel");
    stop.hidden = !(busy || has(job, "cancel")) || !!job.cancelling;
    $("btn-job-delete").hidden = !has(job, "delete");
    $("btn-job-folder").hidden = !has(job, "open_folder");

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
      var active = isBusy(job);
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

  // ------------------------------------------------------------------ polling and routing

  var pollTimer = null, pollRunning = false, pollAgain = false;

  function poll() {
    clearTimeout(pollTimer);
    if (pollRunning) { pollAgain = true; return; }
    if (document.hidden) { return; }  // resumed by visibilitychange
    pollRunning = true;
    var work = state.info ? Promise.resolve() : loadInfo();
    work.then(function () { return loadSchema(); }).then(function () {
      var jobs = refreshJobs();
      if (state.route.view === "detail") {
        return Promise.all([jobs, refreshDetail(), refreshModelsIfNeeded()]).then(function (r) { return r[0] || r[1]; });
      }
      return Promise.all([jobs, loadModels().then(function (m) { return !!(m && m.download && m.download.state === "running"); })])
        .then(function (r) { return r[0] || r[1]; });
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
      if (typeof err.status !== "number") {      // a bug in the page, not the connection
        console.error(err);
        showBanner("connection", "页面出错了：" + (err && err.message) + "（详情见浏览器控制台）");
        return POLL_ERROR_MS;
      }
      showBanner("connection", "连接不上电脑上的扫描程序，正在重试…");
      return POLL_ERROR_MS;
    }).then(function (delay) {
      pollRunning = false;
      if (pollAgain) { pollAgain = false; poll(); return; }
      pollTimer = setTimeout(poll, delay);
    });
  }

  // In the task page the models matter while a stage's panel is open (installed or not).
  function refreshModelsIfNeeded() {
    if (!S3.wf || !S3.wf.selected()) { return Promise.resolve(); }
    return loadModels().then(function () {}, function () {});
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
    document.body.classList.toggle("is-detail", !!job);
    if (job !== previous) { window.scan3dJobView.close(); }    // frees the 3D view's GPU memory
    if (job) {
      if (!state.detail || state.detail.job !== job) {
        state.detail = null;
        if (S3.wf) { S3.wf.reset(); }
        setText($("d-name"), job);
        $("d-state").className = "badge";
        ["d-state", "d-meta", "d-message"].forEach(function (id) { setText($(id), ""); });
        ["d-error", "d-warnings", "d-outputs", "d-viewer", "btn-job-start", "btn-job-cancel", "btn-job-delete", "btn-job-folder"]
          .forEach(function (id) { $(id).hidden = true; });
        $("d-log").open = false;
        $("d-log-text").textContent = "";
        $("d-log-file").removeAttribute("data-signature");
        state.log = { job: null, file: null, offset: null, raw: "" };
      } else {
        window.scan3dJobView.update(state.detail, state.route.tab);   // a tab was chosen
        if (state.route.tab) { scrollToViewer(); }
      }
      showError("d-action-error", "");
      if (job !== previous) { window.scrollTo(0, 0); }
    }
    state.rows.forEach(function (row, name) {
      if (name === job) { row.link.setAttribute("aria-current", "page"); } else { row.link.removeAttribute("aria-current"); }
      S3.setClass(row.li, "job-item" + (name === job ? " is-current" : ""));
    });
    poll();
  }

  function scrollToViewer() {
    var card = $("d-viewer");
    if (card && !card.hidden && card.scrollIntoView) { card.scrollIntoView({ block: "start", behavior: "smooth" }); }
  }

  // ------------------------------------------------------------------ what workflow.js and jobview.js call

  S3.app = {
    // A fresh view of the job (an answer to a PUT or POST): draw it now, poll soon.
    applyJob: function (job) {
      if (state.route.view === "detail" && state.route.job === job.job) { renderDetail(job); }
      poll();
    },
    refreshSoon: function () { poll(); },
    loadModels: loadModels,
    templatesChanged: function () { renderTemplatePicker(); },
    openView: function (tab) {
      var job = state.route.job;
      if (!job) { return; }
      var target = "#/job/" + encodeURIComponent(job) + "/view/" + tab;
      if (location.hash === target) { window.scan3dJobView.update(state.detail, tab); scrollToViewer(); return; }
      location.hash = target;
      setTimeout(scrollToViewer, 60);
    },
    showLog: function (file) {
      var details = $("d-log");
      details.open = true;
      var select = $("d-log-file");
      if (state.detail) { state.logChoice = { job: state.detail.job, file: file }; }
      var found = false;
      for (var i = 0; i < select.options.length; i++) { if (select.options[i].value === file) { found = true; } }
      if (found) { select.value = file; }
      refreshLog().catch(function () {});
      if (details.scrollIntoView) { details.scrollIntoView({ block: "start", behavior: "smooth" }); }
    }
  };

  // ------------------------------------------------------------------ wiring

  function init() {
    $("btn-new").addEventListener("click", function () {      // on the home page the link does not change the address
      if (state.route.view === "home") { $("new-scan").scrollIntoView({ block: "start", behavior: "smooth" }); }
    });
    $("btn-start").addEventListener("click", onStart);
    $("btn-abort").addEventListener("click", onAbort);
    $("opt-start").addEventListener("change", renderForm);
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

    $("btn-job-start").addEventListener("click", function () { S3.wf.run({}); });
    S3.confirmButton($("btn-job-cancel"), "确认停止？", function () { jobAction("POST", "/cancel", {}); });
    S3.confirmButton($("btn-job-delete"), "确认删除？", function () {
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

    S3.wf.init();
    renderFiles();
    renderForm();
    applyRoute();
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", init); }
  else { init(); }
})();
