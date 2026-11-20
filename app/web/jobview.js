/* scan3d "view online" card of the job page: tabs for the report and the 3D views.
 * Plain JavaScript. The 3D part (viewer3d.js and the libraries under vendor/) is loaded only
 * when a 3D tab is opened. Everything put on the page from data goes through textContent.
 *
 * app.js calls  scan3dJobView.update(job, tab)  whenever it has a fresh view of the job that
 * is on screen (tab comes from the address: #/job/<name>/view/<tab>), and
 * scan3dJobView.close()  when the job page is left. */
(function () {
  "use strict";

  var TABS = ["report", "splat", "mesh", "collision", "overlay", "coverage"];
  var SPLAT_TABS = ["splat", "overlay", "coverage"];      // the tabs that show the splat
  var MORE_TABS = ["collision", "overlay", "report"];     // kept under "更多"
  var FIRST_CHOICE = ["splat", "mesh", "collision", "overlay", "coverage", "report"];     // what opens first
  function stageName(key) {      // the label the server's schema gives the stage
    var label = window.S3 && window.S3.stageLabel ? window.S3.stageLabel(key) : "";
    return label || key;
  }
  var FILE_LABEL = { splat: "高斯点云", mesh: "网格模型", collision: "碰撞网格" };
  var PREVIEW_POLL_MS = 1000;
  var COVERAGE_POLL_MS = 1000;
  var SMALL_SCREEN = "(max-width: 640px), (max-height: 480px)";   // same idea as the CSS breakpoint

  var current = {
    job: null,            // job name on screen
    signature: null,      // what was last rendered (job, tab, files): polls change nothing
    tab: null,
    files: {},            // key or file name -> output file entry
    run: 0,               // incremented whenever what is being loaded changes
    viewer: null,
    cameras: [],
    report: null,         // parsed report.json of this job, once loaded
    reportFor: null,
    previews: null,
    coverage: null        // {job, size, data}: the coverage map last loaded
  };
  // Choices the user makes stay for the whole visit of the page.
  var choice = {
    quality: null,          // "lite" | "full" | "original"; null: by screen size (never original)
    before: null,           // the quality in use before "original" was picked (to go back to)
    confirmed: null,        // job whose original-quality warning was accepted on a small screen
    style: { collision: "both", overlay: "wire" }
  };
  var QUALITIES = ["lite", "full", "original"];
  var modulePromise = null;

  function $(id) { return document.getElementById(id); }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
  }

  // ------------------------------------------------------------------ small helpers

  function fmtBytes(n) {
    if (typeof n !== "number" || !isFinite(n)) { return ""; }
    if (n < 1024) { return n + " B"; }
    var units = ["KB", "MB", "GB"], value = n / 1024, i = 0;
    while (value >= 1024 && i < units.length - 1) { value /= 1024; i++; }
    return (value >= 100 ? value.toFixed(0) : value.toFixed(1)) + " " + units[i];
  }

  function fmtCount(n) {
    if (typeof n !== "number" || !isFinite(n)) { return ""; }
    return Math.round(n).toLocaleString("zh-CN");
  }

  function fmtWan(n) {      // 300000 -> "30 万"
    if (typeof n !== "number" || !isFinite(n)) { return ""; }
    if (n < 10000) { return fmtCount(n); }
    var wan = (n / 10000).toFixed(n >= 1000000 ? 0 : 1);
    return wan.replace(/\.0$/, "") + " 万";
  }

  function fmtSeconds(s) {
    if (typeof s !== "number" || !isFinite(s) || s < 0) { return ""; }
    if (s < 60) { return (s < 10 ? s.toFixed(1) : s.toFixed(0)) + " 秒"; }
    var whole = Math.round(s), h = Math.floor(whole / 3600), m = Math.floor((whole % 3600) / 60);
    return (h ? h + " 小时 " : "") + m + " 分 " + (whole % 60) + " 秒";
  }

  function fmtNumber(value, digits) {
    return typeof value === "number" && isFinite(value) ? value.toFixed(digits) : "";
  }

  function fmtDate(iso) {
    var d = new Date(iso || "");
    if (isNaN(d.getTime())) { return ""; }
    function pad(n) { return n < 10 ? "0" + n : String(n); }
    return d.getFullYear() + "年" + (d.getMonth() + 1) + "月" + d.getDate() + "日 " + pad(d.getHours()) + ":" + pad(d.getMinutes());
  }

  function isNum(v) { return typeof v === "number" && isFinite(v); }

  function jobUrl(job, rest) { return "/api/jobs/" + encodeURIComponent(job) + rest; }

  function fileUrl(job, file) {
    return jobUrl(job, "/download/" + file.path.split("/").map(encodeURIComponent).join("/"));
  }

  function request(method, url) {
    return fetch(url, { method: method, cache: "no-cache", credentials: "same-origin" }).then(function (res) {
      return res.text().then(function (text) {
        var data = null;
        try { data = text ? JSON.parse(text) : null; } catch (e) { data = null; }
        if (!res.ok) {
          var err = new Error((data && data.message) || ("请求失败（" + res.status + "）"));
          err.status = res.status;
          throw err;
        }
        return { data: data, text: text };
      });
    }, function () { throw new Error("连接不上电脑上的扫描程序"); });
  }

  function sleep(ms) { return new Promise(function (resolve) { setTimeout(resolve, ms); }); }

  // ------------------------------------------------------------------ report

  function addRow(list, label, value) {
    if (value === null || value === undefined || value === "") { return; }
    list.appendChild(el("dt", null, label));
    list.appendChild(el("dd", null, String(value)));
  }

  function sizeText(size) {
    return Array.isArray(size) && size.length === 2 && isNum(size[0]) && isNum(size[1]) ? size[0] + "×" + size[1] : "";
  }

  // PSNR / SSIM / LPIPS wherever the trainer put them.
  function findMetric(node, name, depth) {
    if (!node || typeof node !== "object" || depth > 4) { return null; }
    var keys = Object.keys(node), i;
    for (i = 0; i < keys.length; i++) {
      if (keys[i].toLowerCase() === name && isNum(node[keys[i]])) { return node[keys[i]]; }
    }
    for (i = 0; i < keys.length; i++) {
      var found = findMetric(node[keys[i]], name, depth + 1);
      if (found !== null) { return found; }
    }
    return null;
  }

  function axisName(up) {
    if (!Array.isArray(up) || up.length !== 3 || !up.every(isNum)) { return ""; }
    var names = ["X", "Y", "Z"];
    for (var i = 0; i < 3; i++) {
      if (Math.abs(up[i]) > 0.999) { return (up[i] > 0 ? "+" : "−") + names[i] + " 轴"; }
    }
    return "(" + up.map(function (v) { return v.toFixed(3); }).join(", ") + ")";
  }

  function renderReport(report, job) {
    var list = $("v-report-list");
    list.textContent = "";
    var stages = (report && typeof report.stages === "object" && report.stages) || {};
    var ingest = stages.ingest || {}, sfm = stages.sfm || {}, dataset = stages.dataset || {};
    var train = stages.train || {}, mesh = stages.mesh || {};

    if (report.template) {
      var tpl = window.S3 && window.S3.templatesInfo ? (window.S3.templatesInfo.templates || []).filter(function (x) { return x.id === report.template; })[0] : null;
      addRow(list, "模板", tpl ? tpl.label : report.template);
    }
    addRow(list, "完成时间", fmtDate(report.finished_at));

    if (isNum(ingest.images)) {
      var parts = [];
      if (isNum(ingest.photos) && ingest.photos > 0) { parts.push(fmtCount(ingest.photos) + " 张照片"); }
      if (isNum(ingest.video_frames) && ingest.video_frames > 0) {
        parts.push(fmtCount(ingest.video_frames) + " 帧取自" + (isNum(ingest.videos) ? " " + ingest.videos + " 段" : "") + "视频");
      }
      var original = sizeText(ingest.size) || (isNum(ingest.long_side) && ingest.long_side > 0 ? "长边 " + ingest.long_side + " 像素" : "");
      addRow(list, "输入素材", fmtCount(ingest.images) + " 张图片" + (parts.length ? "（" + parts.join("，") + "）" : "") + (original ? "，原始尺寸 " + original : ""));
    }
    if (isNum(sfm.registered)) {
      var total = isNum(sfm.images) ? sfm.images : ingest.images;
      var pose = "注册 " + fmtCount(sfm.registered) + (isNum(total) ? " / " + fmtCount(total) : "") + " 张";
      if (isNum(total) && total > 0) { pose += "（" + Math.round(100 * sfm.registered / total) + "%）"; }
      if (sfm.method) { pose += "，方法 " + ({ spirula: "Spirula", colmap: "COLMAP" }[sfm.method] || sfm.method); }
      addRow(list, "相机位姿", pose);
    }
    addRow(list, "重投影误差", isNum(sfm.mean_reprojection_error_px) ? sfm.mean_reprojection_error_px.toFixed(3) + " 像素（越小越好，一般低于 1）" : "");
    if (isNum(sfm.points)) {
      addRow(list, "稀疏点", fmtCount(sfm.points) + " 个" + (isNum(sfm.mean_track_length) ? "，平均每个点被 " + sfm.mean_track_length.toFixed(1) + " 张图片看到" : ""));
    }
    if (isNum(dataset.images) || sizeText(dataset.size)) {
      addRow(list, "训练图像", (isNum(dataset.images) ? fmtCount(dataset.images) + " 张" : "") + (sizeText(dataset.size) ? (isNum(dataset.images) ? "，" : "") + sizeText(dataset.size) + " 像素" : ""));
    }
    if (isNum(train.gaussians)) {
      addRow(list, "高斯数量", fmtCount(train.gaussians) + " 个" + (isNum(train.iterations) ? "，训练 " + fmtCount(train.iterations) + " 步" : ""));
    }
    var psnr = findMetric(train.metrics, "psnr", 0), ssim = findMetric(train.metrics, "ssim", 0), lpips = findMetric(train.metrics, "lpips", 0);
    var metrics = [];
    if (psnr !== null) { metrics.push("PSNR " + psnr.toFixed(2)); }
    if (ssim !== null) { metrics.push("SSIM " + ssim.toFixed(3)); }
    if (lpips !== null) { metrics.push("LPIPS " + lpips.toFixed(3)); }
    addRow(list, "留出评测", metrics.length ? metrics.join("，") + "（PSNR、SSIM 越高越好，LPIPS 越低越好）" : "");
    if (isNum(mesh.mesh_faces)) {
      var texture = isNum(mesh.texture_width) && isNum(mesh.texture_height) ? "，贴图 " + mesh.texture_width + "×" + mesh.texture_height : "";
      addRow(list, "网格模型", fmtCount(mesh.mesh_faces) + " 个面" + (isNum(mesh.mesh_vertices) ? "，" + fmtCount(mesh.mesh_vertices) + " 个顶点" : "") + texture);
    }
    if (isNum(mesh.collision_faces)) {
      addRow(list, "碰撞网格", fmtCount(mesh.collision_faces) + " 个面" + (isNum(mesh.collision_vertices) ? "，" + fmtCount(mesh.collision_vertices) + " 个顶点" : ""));
    }

    var sizes = [];
    (job.output_files || []).forEach(function (file) {
      if (FILE_LABEL[file.key]) { sizes.push(FILE_LABEL[file.key] + " " + fmtBytes(file.size)); }
    });
    addRow(list, "文件大小", sizes.join("，"));

    var timings = [], sum = 0;
    Object.keys(stages).forEach(function (key) {
      var seconds = stages[key] && stages[key].stage_seconds;
      if (isNum(seconds)) { timings.push(stageName(key) + " " + fmtSeconds(seconds)); sum += seconds; }
    });
    if (timings.length) { addRow(list, "各阶段用时", timings.join("，") + "；合计 " + fmtSeconds(sum)); }

    var orientation = report.orientation;
    if (orientation && Array.isArray(orientation.up_in_file_coordinates)) {
      addRow(list, "朝向", "场景的上方是文件坐标的 " + axisName(orientation.up_in_file_coordinates) + "。两个 GLB 文件已经带有转正的旋转（Y 轴朝上），网页里的高斯按同一个旋转显示。模型的大小是任意的。");
    } else if (report.orientation === null) {
      addRow(list, "朝向", "没能判断哪边是上方，文件保持原始坐标。");
    }

    var warnings = Array.isArray(report.warnings) ? report.warnings : [];
    var box = $("v-report-warnings");
    box.textContent = "";
    warnings.forEach(function (text) { box.appendChild(el("li", null, String(text))); });
    box.hidden = warnings.length === 0;
    addRow(list, "警告", warnings.length ? warnings.length + " 条，见下方" : "无");
  }

  function identity3(m) {
    if (!Array.isArray(m) || m.length !== 3) { return null; }
    var off = 0;
    for (var r = 0; r < 3; r++) {
      if (!Array.isArray(m[r]) || m[r].length !== 3 || !m[r].every(isNum)) { return null; }
      for (var c = 0; c < 3; c++) { off = Math.max(off, Math.abs(m[r][c] - (r === c ? 1 : 0))); }
    }
    return off < 1e-6;
  }

  function transformSummary(data) {
    var part = data && data.train_from_world;
    if (!part || typeof part !== "object") { return "这个文件的格式无法识别，请看原始内容。"; }
    var scale = isNum(part.scale) ? part.scale : 1;
    var t = Array.isArray(part.translation) && part.translation.length === 3 && part.translation.every(isNum) ? part.translation : [0, 0, 0];
    var rotation = part.rotation || {};
    var same = identity3(rotation.matrix_3x3);
    var moved = Math.max(Math.abs(t[0]), Math.abs(t[1]), Math.abs(t[2])) > 1e-9;
    var text;
    if (same === null) {
      text = "这个文件的格式无法识别，请看原始内容。";
    } else if (same && !moved && Math.abs(scale - 1) < 1e-9) {
      text = "无变换：高斯和网格的坐标就是相机位姿计算得到的原始坐标。";
    } else {
      var steps = [];
      if (!same) {
        var angle = rotation.axis_angle && isNum(rotation.axis_angle.angle_deg) ? rotation.axis_angle.angle_deg : null;
        var axis = rotation.axis_angle && Array.isArray(rotation.axis_angle.axis) && rotation.axis_angle.axis.every(isNum) ? rotation.axis_angle.axis : null;
        steps.push(angle !== null ? "旋转 " + angle.toFixed(2) + "°" + (axis ? "（绕轴 " + axis.map(function (v) { return v.toFixed(3); }).join(", ") + "）" : "") : "旋转");
      }
      if (Math.abs(scale - 1) >= 1e-9) { steps.push("缩放 " + scale.toPrecision(6) + " 倍"); }
      if (moved) { steps.push("平移 (" + t.map(function (v) { return v.toPrecision(5); }).join(", ") + ")"); }
      text = "训练时坐标经过了变换：原始坐标先" + steps.join("、再") + "，得到高斯和网格使用的坐标。";
    }
    var geo = data.georeference;
    if (geo && isNum(geo.origin_lat) && isNum(geo.origin_lon)) {
      text += " 带有地理参考：原点位于北纬 " + geo.origin_lat.toFixed(6) + "°、东经 " + geo.origin_lon.toFixed(6) + "°，x 朝东、y 朝北、z 朝上，单位米。";
    }
    return text;
  }

  function loadJson(job, file) {
    return request("GET", fileUrl(job, file)).then(function (result) {
      if (!result.data || typeof result.data !== "object") { throw new Error("文件内容不是合法的 JSON"); }
      return result;
    });
  }

  function showReport(job, run) {
    var reportFile = current.files.report || current.files["report.json"];
    var transformFile = current.files["scene_transform.json"];
    var status = $("v-report-status");
    $("v-report-list").textContent = "";
    $("v-report-warnings").hidden = true;
    ["v-report-raw-box", "v-transform-title", "v-transform-summary", "v-transform-raw-box"].forEach(function (id) { $(id).hidden = true; });
    status.hidden = false;
    status.textContent = reportFile ? "正在读取报告…" : "这个任务没有 report.json。";
    if (reportFile) {
      loadJson(job.job, reportFile).then(function (result) {
        if (run !== current.run) { return; }
        current.report = result.data;
        current.reportFor = job.job;
        status.hidden = true;
        renderReport(result.data, job);
        $("v-report-raw").textContent = JSON.stringify(result.data, null, 2);
        $("v-report-raw-box").hidden = false;
      }, function (err) {
        if (run === current.run) { status.textContent = "无法读取报告：" + err.message; }
      });
    }
    if (transformFile) {
      loadJson(job.job, transformFile).then(function (result) {
        if (run !== current.run) { return; }
        $("v-transform-summary").textContent = transformSummary(result.data);
        $("v-transform-raw").textContent = result.text;
        ["v-transform-title", "v-transform-summary", "v-transform-raw-box"].forEach(function (id) { $(id).hidden = false; });
      }, function (err) {
        if (run !== current.run) { return; }
        $("v-transform-summary").textContent = "无法读取 scene_transform.json：" + err.message;
        ["v-transform-title", "v-transform-summary"].forEach(function (id) { $(id).hidden = false; });
      });
    }
  }

  // ------------------------------------------------------------------ 3D: status line

  function setStatus(text, loaded, total, failed) {
    var box = $("v-status");
    box.hidden = false;
    box.classList.toggle("is-error", !!failed);
    $("v-status-text").textContent = text;
    var track = $("v-progress");
    track.hidden = !!failed;
    if (total > 0) {
      track.className = "progress";
      $("v-progress-bar").style.width = (100 * Math.min(1, loaded / total)).toFixed(1) + "%";
    } else {
      track.className = "progress is-indeterminate";
      $("v-progress-bar").style.width = "";
    }
    $("v-retry").hidden = true;
  }

  function showFailure(text, canRetry) {
    setStatus(text, 0, 0, true);
    $("v-retry").hidden = !canRetry;
  }

  function disposeViewer() {
    if (current.viewer) {
      current.viewer.dispose();
      current.viewer = null;
    }
    setExpanded(false);
  }

  function setExpanded(on) {
    var pane = $("v-3d");
    if (pane.classList.contains("is-expanded") === on) { return; }
    pane.classList.toggle("is-expanded", on);
    document.documentElement.classList.toggle("viewer-expanded", on);
    $("v-expand").textContent = on ? "退出全屏" : "全屏";
    $("v-expand").setAttribute("aria-pressed", String(on));
  }

  // ------------------------------------------------------------------ 3D: what to load

  function smallScreen() {
    return !!(window.matchMedia && window.matchMedia(SMALL_SCREEN).matches);
  }

  // Which of the three qualities of the splat to show:
  //   lite      preview-lite.spz: the most important 300,000 Gaussians, no view-dependent colour
  //   full      preview.spz: every Gaussian with all its view-dependent colour, compressed
  //   original  splat.ply itself, exactly the file that can be downloaded
  function quality() {
    if (!choice.quality) {
      // The lighter file on a small screen, the full one otherwise; never the original by
      // itself. Decided once, by the size of the window; after that only the switch on the
      // page changes it.
      choice.quality = smallScreen() ? "lite" : "full";
    }
    return choice.quality;
  }

  function loadModule() {
    if (!modulePromise) {
      modulePromise = import("/static/viewer3d.js");
      modulePromise.catch(function () { modulePromise = null; });   // a later try may succeed
    }
    return modulePromise;
  }

  function loadRotation(job) {
    if (current.reportFor === job.job && current.report) { return Promise.resolve(current.report); }
    var file = current.files.report || current.files["report.json"];
    if (!file) { return Promise.resolve(null); }
    return loadJson(job.job, file).then(function (result) {
      current.report = result.data;
      current.reportFor = job.job;
      return result.data;
    }, function () { return null; });
  }

  function loadCameras(job) {
    return request("GET", jobUrl(job.job, "/cameras")).then(function (result) {
      return (result.data && Array.isArray(result.data.cameras)) ? result.data.cameras : [];
    }, function () { return []; });
  }

  function renderQualityLabels() {
    var variants = (current.previews && current.previews.variants) || {};
    var total = current.report && current.report.stages && current.report.stages.train && current.report.stages.train.gaussians;
    function label(node, name, variant, tip) {
      var size = variant && variant.state === "ready" && isNum(variant.size) ? "  " + fmtBytes(variant.size) : "";
      node.textContent = name + size;
      var most = variant && variant.max_gaussians;
      var count = isNum(most) && !(isNum(total) && total <= most) ? "最多 " + fmtWan(most) + "个高斯"
        : (isNum(total) ? fmtWan(total) + "个高斯" : "");
      S3.setTip(node.parentNode, (count ? count + "。" : "") + tip);
    }
    label($("v-quality-lite"), "轻量", variants.lite, "精简了高斯，不含随视角变化的颜色（反光、高光），适合手机。");
    label($("v-quality-full"), "完整", variants.full, "全部高斯，含随视角变化的颜色，数值经过压缩，和原文件差别很难看出来。");
    var source = sourceFile();
    $("v-quality-original").textContent = "原画质" + (source && isNum(source.size) ? "  " + fmtBytes(source.size) : "");
    S3.setTip($("v-quality-original").parentNode, "直接读取 splat.ply，和下载到的文件完全一致，文件大，加载慢。");
  }

  // splat.ply itself: what the server said about it, or failing that the entry in the file list.
  function sourceFile() {
    var source = current.previews && current.previews.source;
    if (source && typeof source.path === "string") { return source; }
    return current.files.splat || null;
  }

  function hideConfirm() {
    $("v-confirm").hidden = true;
    confirmRun = 0;
  }

  // Resolves with what the viewer needs to load splat.ply. On a small screen it first shows
  // how big the file is and waits for a plain yes (once per job); "no" goes back to the
  // quality that was in use before.
  function ensureOriginal(job, run) {
    return request("GET", jobUrl(job.job, "/preview")).then(function (result) {
      if (run !== current.run) { throw new Error("cancelled"); }
      current.previews = result.data;
      renderQualityLabels();
      var source = sourceFile();
      if (!source) { throw new Error("这个任务没有高斯点云文件。"); }
      var ready = { url: fileUrl(job.job, source) + "?s=" + source.size, type: "ply", precise: true };
      if (!smallScreen() || choice.confirmed === job.job) { return ready; }
      return new Promise(function (resolve) {
        var count = isNum(source.gaussians) ? "，" + fmtWan(source.gaussians) + "个高斯" : "";
        $("v-confirm-title").textContent = "原画质要下载 " + fmtBytes(source.size) + count;
        $("v-confirm-text").textContent = "这是 splat.ply 原文件，没有经过压缩。手机和平板的内存通常不够：下载会很慢，页面可能卡住或者被系统关掉。建议在电脑上看原画质；“完整”已经包含随视角变化的颜色，文件小得多。";
        $("v-status").hidden = true;
        $("v-confirm").hidden = false;
        confirmRun = run;
        confirmYes = function () {
          choice.confirmed = job.job;
          setStatus("正在加载三维查看器…", 0, 0);
          resolve(ready);
        };
      });
    });
  }
  var confirmRun = 0, confirmYes = null;

  function answerConfirm(yes) {
    if (!confirmRun || confirmRun !== current.run) { hideConfirm(); return; }
    var proceed = confirmYes;
    hideConfirm();
    if (yes && proceed) { proceed(); return; }
    choice.quality = choice.before && choice.before !== "original" ? choice.before : "lite";
    restart();
  }

  // Resolves with the address of the preview file, asking the server to build it first when
  // the job has none. The server builds one at a time and never keeps a request waiting, so
  // this polls; it stops as soon as the user moves on (run changes).
  function ensurePreview(job, variant, run) {
    var asked = false, started = Date.now();
    function step() {
      if (run !== current.run) { return Promise.reject(new Error("cancelled")); }
      return request("GET", jobUrl(job.job, "/preview")).then(function (result) {
        if (run !== current.run) { throw new Error("cancelled"); }
        current.previews = result.data;
        renderQualityLabels();
        var state = result.data && result.data.variants && result.data.variants[variant];
        if (!state) { throw new Error("服务器没有给出预览状态"); }
        if (state.state === "ready") {
          // "precise": see viewer3d.js. The full preview is worth it, the reduced one is
          // meant to be light.
          return { url: jobUrl(job.job, "/preview/" + variant) + "?s=" + state.size, type: "spz", precise: variant === "full" };
        }
        if (state.state === "no_splat") { throw new Error("这个任务没有高斯点云文件。"); }
        if (state.state === "failed" && asked) { throw new Error(state.error || "无法生成预览。"); }
        var seconds = Math.round((Date.now() - started) / 1000);
        var what = state.output_outdated
          ? "这个任务的网页预览是旧版本（不含随视角变化的颜色），正在重新生成（只需一次，通常十几秒）…"
          : "正在生成网页预览（每个任务只需一次，通常十几秒）…";
        setStatus(what + (seconds >= 3 ? " 已用 " + seconds + " 秒" : ""), 0, 0);
        if (state.state === "missing" || state.state === "failed") {
          asked = true;
          return request("POST", jobUrl(job.job, "/preview/" + variant)).then(step);
        }
        asked = true;
        return sleep(PREVIEW_POLL_MS).then(step);
      });
    }
    return step();
  }

  // Resolves with the coverage map (app/coverage.py), asking the server to compute it first
  // when the job has none. Like the previews: the server never keeps a request waiting, so
  // this polls, and stops as soon as the user moves on.
  function ensureCoverage(job, run) {
    var asked = false, started = Date.now();
    function step() {
      if (run !== current.run) { return Promise.reject(new Error("cancelled")); }
      return request("GET", jobUrl(job.job, "/coverage")).then(function (result) {
        if (run !== current.run) { throw new Error("cancelled"); }
        var state = result.data || {};
        if (state.state === "ready") {
          var have = current.coverage;
          if (have && have.job === job.job && have.size === state.size) { return have.data; }
          setStatus("正在读取拍摄覆盖…", 0, 0);
          return request("GET", jobUrl(job.job, "/coverage/data") + "?s=" + state.size).then(function (got) {
            if (!got.data || !got.data.voxels || !got.data.grid) { throw new Error("拍摄覆盖的数据格式不对"); }
            current.coverage = { job: job.job, size: state.size, data: got.data };
            return got.data;
          });
        }
        if (state.state === "unavailable") {
          var err = new Error(state.message || "这个任务无法计算拍摄覆盖。");
          err.final = true;
          throw err;
        }
        if (state.state === "missing" && state.enabled === false) {
          var off = new Error("「拍摄覆盖」这一步没有打开：在上面的工作流里打开它，再点「运行」。");
          off.final = true;
          throw off;
        }
        if (state.state === "failed" && asked) { throw new Error(state.error || "无法计算拍摄覆盖。"); }
        var seconds = Math.round((Date.now() - started) / 1000);
        setStatus("正在计算拍摄覆盖（每个任务只需一次，通常不到半分钟）…" + (seconds >= 3 ? " 已用 " + seconds + " 秒" : ""), 0, 0);
        if (state.state === "missing" || state.state === "failed") {
          asked = true;
          return request("POST", jobUrl(job.job, "/coverage")).then(step);
        }
        asked = true;
        return sleep(COVERAGE_POLL_MS).then(step);
      });
    }
    return step();
  }

  function percent(part, whole) {
    if (!(whole > 0)) { return "0%"; }
    var p = 100 * part / whole;
    return (p > 0 && p < 1 ? "不到 1" : Math.round(p)) + "%";
  }

  // The panel under the picture: what the colours say, and the weak regions to re-shoot.
  // The shares come from the map (by amount of surface in the part the photos aimed at; see
  // app/coverage.py), so they do not depend on which quality of the splat is loaded.
  function renderCoverage(data) {
    var share = data.surface_share || data.levels || {};
    var judged = (share.weak || 0) + (share.medium || 0) + (share.strong || 0);
    var cameras = Array.isArray(data.cameras) ? data.cameras.length : 0;
    $("v-coverage-summary").textContent = "按 " + fmtCount(cameras) + " 张照片计算。" + (judged > 0
      ? "照片对准的范围里：拍得充分 " + percent(share.strong || 0, judged) + "，一般 " +
        percent(share.medium || 0, judged) + "，不足 " + percent(share.weak || 0, judged) +
        "。范围外的远景是灰色，不做判断。"
      : "");
    var list = $("v-coverage-list");
    list.textContent = "";
    var regions = Array.isArray(data.regions) ? data.regions : [];
    $("v-coverage-none").hidden = regions.length > 0;
    $("v-coverage-title").hidden = regions.length === 0;
    regions.forEach(function (region) {
      if (!region || typeof region.number !== "number") { return; }
      var item = el("li", "coverage-item");
      item.setAttribute("data-region", String(region.number));
      item.appendChild(el("p", "coverage-text", "第 " + region.number + " 处：" + String(region.text || "")));
      var buttons = el("div", "coverage-buttons");
      var go = el("button", "btn small", region.suggest ? "从这里补拍" : "看看这里");
      go.type = "button";
      go.setAttribute("data-go", "suggest");
      buttons.appendChild(go);
      if (region.suggest) {
        var look = el("button", "btn small", "从最近的照片看");
        look.type = "button";
        look.setAttribute("data-go", "photo");
        buttons.appendChild(look);
      }
      item.appendChild(buttons);
      list.appendChild(item);
    });
    markPicked(0);
  }

  function markPicked(number) {
    var items = $("v-coverage-list").querySelectorAll(".coverage-item");
    for (var i = 0; i < items.length; i++) {
      var on = items[i].getAttribute("data-region") === String(number);
      items[i].classList.toggle("is-picked", on);
    }
  }

  // The line under the picture: what is being shown, and what this quality leaves out.
  function splatNote(tier, info) {
    if (!(info.splats > 0)) { return ""; }
    var variants = (current.previews && current.previews.variants) || {};
    var source = sourceFile() || {};
    var text = "正在显示 " + fmtCount(info.splats) + " 个高斯";
    text += tier === "lite" ? "（轻量档，适合手机）。" : tier === "original" ? "（原画质，和 splat.ply 一致）。" : "。";
    if (source.sh_degree === 0) { text += "这个点云本身不含随视角变化的颜色。"; }
    if (tier === "full" && variants.full && variants.full.output_outdated) {
      text += "输出文件夹里的 preview.spz 是旧版本，网页已经另外生成了新的并改用它。";
    }
    return text;
  }

  function show3d(job, tab, run) {
    var needSplat = SPLAT_TABS.indexOf(tab) >= 0;
    var needCollision = tab === "collision" || tab === "overlay";
    var needCoverage = tab === "coverage";
    $("v-quality").hidden = !needSplat;
    $("v-style").hidden = !needCollision;
    $("v-coverage").hidden = true;
    $("v-legend").hidden = !needCoverage;
    $("v-note").textContent = "";
    hideConfirm();
    var tier = quality();
    var radios = document.querySelectorAll('input[name="v-quality"]'), i;
    for (i = 0; i < radios.length; i++) { radios[i].checked = radios[i].value === quality(); }
    radios = document.querySelectorAll('input[name="v-style"]');
    for (i = 0; i < radios.length; i++) { radios[i].checked = radios[i].value === choice.style[tab]; }
    renderQualityLabels();
    setStatus("正在加载三维查看器…", 0, 0);

    var mod = null, report = null, coverage = null;
    loadModule().then(function (loaded) {
      mod = loaded;
      if (run !== current.run) { throw new Error("cancelled"); }
      if (!mod.webgl2Supported()) {
        var err = new Error("这个浏览器不支持 WebGL2，无法显示三维内容。请更新浏览器，或换一台设备查看；文件仍然可以在上面下载。");
        err.final = true;
        throw err;
      }
      return Promise.all([loadRotation(job), loadCameras(job)]);
    }, function () {
      throw new Error("无法加载三维查看器的组件，请刷新页面重试。");
    }).then(function (results) {
      report = results[0];
      current.cameras = results[1];
      renderQualityLabels();
      return needCoverage ? ensureCoverage(job, run) : null;
    }).then(function (map) {
      coverage = map;
      if (!needSplat) { return null; }
      return tier === "original" ? ensureOriginal(job, run) : ensurePreview(job, tier, run);
    }).then(function (splat) {
      if (run !== current.run) { throw new Error("cancelled"); }
      if (!current.viewer) {
        current.viewer = new mod.Viewer3D($("v-stage"), {
          onProgress: function (p) {
            var text = p.label + (p.total > 0 ? " " + fmtBytes(p.loaded) + " / " + fmtBytes(p.total) : (p.loaded > 0 ? " " + fmtBytes(p.loaded) : ""));
            setStatus(text, p.loaded, p.total);
          },
          onReady: function (info) {
            $("v-status").hidden = true;
            if (current.tab === "coverage" && info.coverage && current.coverage) {
              $("v-note").textContent = "颜色表示每一处被多少张照片、从多宽的角度拍到（看得见背景的空洞不会上色）。白点是拍照的位置。" + splatNote(quality(), info);
              renderCoverage(current.coverage.data);
              $("v-coverage").hidden = false;
            } else {
              $("v-note").textContent = splatNote(quality(), info);
            }
          },
          onError: function (message) { showFailure(message, true); }
        });
      }
      $("v-next-camera").hidden = current.cameras.length < 2;
      var orientation = report && report.orientation;
      return current.viewer.show({
        mode: tab,
        splat: splat,
        urls: {
          splat: splat ? splat.url : null,
          mesh: current.files.mesh ? fileUrl(job.job, current.files.mesh) : null,
          collision: current.files.collision ? fileUrl(job.job, current.files.collision) : null
        },
        rotation: orientation && orientation.rotation_to_y_up_xyzw,
        cameras: current.cameras,
        coverage: coverage,
        style: choice.style[tab] || "wire"
      });
    }).catch(function (err) {
      if (run !== current.run || (err && err.message === "cancelled")) { return; }
      showFailure((err && err.message) || "加载失败", !(err && err.final));
    });
  }

  // ------------------------------------------------------------------ tabs

  function activate(job, tab) {
    var run = ++current.run;
    current.tab = tab;
    var links = $("v-tabs").querySelectorAll("a");
    for (var i = 0; i < links.length; i++) {
      var on = links[i].getAttribute("data-view") === tab;
      links[i].classList.toggle("is-active", on);
      if (on) { links[i].setAttribute("aria-current", "page"); } else { links[i].removeAttribute("aria-current"); }
    }
    var more = $("v-more");
    more.querySelector("summary").classList.toggle("is-active", !!more.querySelector("a.is-active"));
    $("v-report").hidden = tab !== "report";
    $("v-3d").hidden = tab === "report";
    if (tab === "report") {
      disposeViewer();        // leaving 3D: give the GPU memory back
      showReport(job, run);
    } else {
      show3d(job, tab, run);
    }
  }

  function update(job, requested) {
    var files = {};
    (job.output_files || []).forEach(function (file) {
      if (file.key) { files[file.key] = file; }
      if (!files[file.name]) { files[file.name] = file; }
    });
    var available = {
      report: !!(files.report || files["report.json"] || files["scene_transform.json"]),
      splat: !!files.splat, mesh: !!files.mesh, collision: !!files.collision,
      overlay: !!(files.splat && files.collision),
      coverage: !!files.splat && !!(files["coverage.json"] || (job.workflow && job.workflow.stages && job.workflow.stages.coverage && job.workflow.stages.coverage.enabled))
    };
    var any = TABS.some(function (tab) { return available[tab]; });
    var visible = any;      // whatever stages have finished, their files can be looked at
    $("d-viewer").hidden = !visible;
    if (!visible) { close(); return; }

    var links = $("v-tabs").querySelectorAll("a");
    for (var i = 0; i < links.length; i++) {
      var name = links[i].getAttribute("data-view");
      links[i].hidden = !available[name];
      links[i].setAttribute("href", "#/job/" + encodeURIComponent(job.job) + "/view/" + name);
    }
    $("v-more").hidden = !TABS.some(function (name) { return MORE_TABS.indexOf(name) >= 0 && available[name]; });
    var tab = requested && available[requested] ? requested : FIRST_CHOICE.filter(function (t) { return available[t]; })[0];
    var signature = job.job + "|" + tab + "|" + (job.output_files || []).map(function (f) { return f.path + ":" + f.size; }).join(",");
    if (signature === current.signature) { return; }
    if (current.job !== job.job) {
      disposeViewer();
      current.report = null;
      current.reportFor = null;
      current.previews = null;
      current.cameras = [];
      current.coverage = null;
    }
    current.job = job.job;
    current.signature = signature;
    current.files = files;
    current.jobView = job;
    activate(job, tab);
  }

  function close() {
    current.run++;
    current.signature = null;
    current.job = null;
    current.tab = null;
    hideConfirm();
    disposeViewer();
  }

  function restart() {      // same tab again with the current choices (quality switch, retry)
    if (current.jobView && current.tab && current.tab !== "report") {
      show3d(current.jobView, current.tab, ++current.run);
    }
  }

  function init() {
    $("v-retry").addEventListener("click", restart);
    $("v-quality").addEventListener("change", function (event) {
      if (event.target && event.target.name === "v-quality" && QUALITIES.indexOf(event.target.value) >= 0) {
        if (event.target.value === "original" && choice.quality !== "original") { choice.before = quality(); }
        choice.quality = event.target.value;
        restart();
      }
    });
    $("v-confirm-yes").addEventListener("click", function () { answerConfirm(true); });
    $("v-confirm-no").addEventListener("click", function () { answerConfirm(false); });
    $("v-style").addEventListener("change", function (event) {
      if (event.target && event.target.name === "v-style" && current.tab) {
        choice.style[current.tab] = event.target.value;
        if (current.viewer) { current.viewer.setStyle(event.target.value); }
      }
    });
    $("v-reset").addEventListener("click", function () { if (current.viewer) { current.viewer.resetView(); } });
    $("v-coverage-list").addEventListener("click", function (event) {
      var button = event.target && event.target.closest ? event.target.closest("button[data-go]") : null;
      var item = button ? button.closest(".coverage-item") : null;
      if (!item || !current.viewer || current.tab !== "coverage") { return; }
      var number = parseInt(item.getAttribute("data-region"), 10);
      current.viewer.pickRegion(number, button.getAttribute("data-go"));
      markPicked(number);
      var stage = $("v-stage");
      if (stage.getBoundingClientRect().top < 0) { stage.scrollIntoView({ block: "start", behavior: "smooth" }); }
    });
    $("v-next-camera").addEventListener("click", function () { if (current.viewer) { current.viewer.nextCamera(); } });
    $("v-expand").addEventListener("click", function () { setExpanded(!$("v-3d").classList.contains("is-expanded")); });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") { setExpanded(false); }
    });
    window.addEventListener("pagehide", disposeViewer);
    window.addEventListener("pageshow", function (event) { if (event.persisted) { restart(); } });
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", init); }
  else { init(); }

  window.scan3dJobView = { update: update, close: close };
})();
