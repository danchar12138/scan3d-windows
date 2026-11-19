/* scan3d UI: helpers shared by the other scripts (window.S3). Plain JavaScript, no dependencies.
 * Everything put on the page from data goes through textContent, never innerHTML. */
(function () {
  "use strict";

  var S3 = window.S3 = window.S3 || {};
  var SVG_NS = "http://www.w3.org/2000/svg";

  S3.$ = function (id) { return document.getElementById(id); };

  S3.el = function (tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
  };

  S3.setText = function (node, text) {
    if (node.textContent !== text) { node.textContent = text; }
  };

  S3.setClass = function (node, cls) {      // also for SVG elements, whose className is not a string
    if (node.getAttribute("class") !== cls) { node.setAttribute("class", cls); }
  };

  S3.sleep = function (ms) { return new Promise(function (resolve) { setTimeout(resolve, ms); }); };

  // An icon from the sprite in index.html: <svg class="ico"><use href="#i-name"/></svg>
  S3.icon = function (name, className) {
    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "ico" + (className ? " " + className : ""));
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    var use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", "#i-" + name);
    svg.appendChild(use);
    return svg;
  };

  S3.setIcon = function (svg, name) {
    var use = svg.firstChild;
    var href = "#i-" + name;
    if (use && use.getAttribute("href") !== href) { use.setAttribute("href", href); }
  };

  // ------------------------------------------------------------------ vocabulary

  // What each stage state looks like. The word and the icon carry the meaning; colour only helps.
  S3.STAGE_STATE = {
    disabled: { word: "未开启", icon: "disabled", help: "这个阶段没有打开，不会运行" },
    pending: { word: "待运行", icon: "pending", help: "已打开，还没有运行" },
    queued: { word: "排队中", icon: "queued", help: "在这次运行的计划里，等前面的阶段" },
    running: { word: "运行中", icon: "run", help: "正在运行" },
    done: { word: "已完成", icon: "done", help: "结果是最新的" },
    stale: { word: "需重算", icon: "stale", help: "有结果，但这之后改过设置或上游重新运行了，结果已经过期" },
    failed: { word: "失败", icon: "failed", help: "上一次运行在这里出错了" }
  };
  S3.STATE_ORDER = ["pending", "queued", "running", "done", "stale", "failed", "disabled"];

  S3.JOB_STATE = {
    created: "待运行", queued: "排队中", running: "运行中",
    done: "已完成", failed: "失败", cancelled: "已取消"
  };

  // Labels of the small facts a stage reports (status.json stages.<id>.summary).
  S3.SUMMARY_LABEL = {
    images: "图片", photos: "照片", videos: "视频", video_frames: "视频帧",
    registered: "注册照片", points: "稀疏点", gaussians: "高斯数量", iterations: "训练步数",
    mesh_faces: "网格面数", collision_faces: "碰撞网格面数", dense_points: "稠密点",
    masked_percent: "抠掉的比例（%）", mask_kind: "抠图方式", regions: "待补拍区域", method: "方法"
  };
  S3.SUMMARY_VALUE = { moving: "抠掉走动的人", subject: "只留主体", spirula: "Spirula", colmap: "COLMAP" };

  // ------------------------------------------------------------------ formatting

  S3.fmtBytes = function (n) {
    if (typeof n !== "number" || !isFinite(n)) { return ""; }
    if (n < 1024) { return n + " B"; }
    var units = ["KB", "MB", "GB", "TB"];
    var value = n / 1024, i = 0;
    while (value >= 1024 && i < units.length - 1) { value /= 1024; i++; }
    return (value >= 100 ? value.toFixed(0) : value.toFixed(1)) + " " + units[i];
  };

  function pad2(n) { return n < 10 ? "0" + n : String(n); }

  // 3725 -> "1:02:05", 125 -> "2:05"
  S3.fmtDuration = function (seconds) {
    if (typeof seconds !== "number" || !isFinite(seconds) || seconds < 0) { return ""; }
    var s = Math.floor(seconds);
    var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return (h > 0 ? h + ":" + pad2(m) : String(m)) + ":" + pad2(s % 60);
  };

  // 45 -> "45 秒", 125 -> "2 分 5 秒"; for the time a finished stage took
  S3.fmtSpent = function (seconds) {
    if (typeof seconds !== "number" || !isFinite(seconds) || seconds < 0) { return ""; }
    if (seconds < 1) { return "不到 1 秒"; }
    if (seconds < 60) { return Math.round(seconds) + " 秒"; }
    var whole = Math.round(seconds), h = Math.floor(whole / 3600), m = Math.floor((whole % 3600) / 60);
    if (h) { return h + " 小时 " + m + " 分"; }
    return m + " 分 " + (whole % 60) + " 秒";
  };

  S3.fmtRemaining = function (seconds) {
    if (!isFinite(seconds) || seconds <= 0) { return ""; }
    if (seconds < 60) { return "不到 1 分钟"; }
    if (seconds < 3600) { return "约 " + Math.round(seconds / 60) + " 分钟"; }
    return "约 " + (seconds / 3600).toFixed(1) + " 小时";
  };

  S3.fmtTime = function (iso) {
    if (!iso) { return ""; }
    var d = new Date(iso);
    if (isNaN(d.getTime())) { return ""; }
    return (d.getMonth() + 1) + "月" + d.getDate() + "日 " + pad2(d.getHours()) + ":" + pad2(d.getMinutes());
  };

  S3.fmtCount = function (n) {
    if (typeof n !== "number" || !isFinite(n)) { return ""; }
    if (Math.abs(n) >= 100000) { return (n / 10000).toFixed(n >= 1000000 ? 0 : 1).replace(/\.0$/, "") + " 万"; }
    return Math.round(n).toLocaleString("zh-CN");
  };

  S3.fmtMb = function (mb) {
    if (typeof mb !== "number" || !isFinite(mb)) { return ""; }
    if (mb >= 1024) { return (mb / 1024).toFixed(1) + " GB"; }
    return Math.round(mb) + " MB";
  };

  S3.firstLine = function (text) { return String(text || "").split("\n")[0]; };

  S3.clamp01 = function (v) { return Math.max(0, Math.min(1, v)); };

  // ------------------------------------------------------------------ API

  var API_TIMEOUT_MS = 15000;

  S3.apiError = function (message, status, data) {
    var err = new Error(message);
    err.status = status || 0;
    err.data = data || null;
    err.code = data && data.error ? data.error : "";
    return err;
  };

  S3.api = function (method, path, body) {
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
          var message = (data && data.message) || ("请求失败（" + res.status + "）");
          if (data && Array.isArray(data.problems) && data.problems.length) {
            message += "\n" + data.problems.join("\n");
          }
          throw S3.apiError(message, res.status, data);
        }
        return data;
      });
    }, function () {
      throw S3.apiError("连接不上电脑上的扫描程序", 0, null);
    }).finally(function () { clearTimeout(timer); });
  };

  S3.jobUrl = function (job, rest) {
    return "/api/jobs/" + encodeURIComponent(job) + (rest || "");
  };

  // First click arms the button, a second click within a few seconds runs the action.
  // One tooltip for the whole page. The browser's own `title` tooltip waits about a second and
  // flashes an empty dark box first, so every hint is set with S3.setTip and drawn here at once.
  S3.setTip = function (node, text) {
    if (text) { node.setAttribute("data-tip", text); } else { node.removeAttribute("data-tip"); }
  };
  (function () {
    var tip = null, target = null, timer = null;
    function hide() {
      clearTimeout(timer); timer = null; target = null;
      if (tip) { tip.classList.remove("is-on"); }
    }
    function place() {
      if (!target || !tip) { return; }
      tip.textContent = target.getAttribute("data-tip") || "";
      if (!tip.textContent) { hide(); return; }
      var r = target.getBoundingClientRect(), w = document.documentElement.clientWidth;
      tip.style.left = "0px"; tip.style.top = "0px";
      tip.classList.add("is-on");
      var t = tip.getBoundingClientRect();
      var left = Math.max(8, Math.min(w - t.width - 8, r.left + r.width / 2 - t.width / 2));
      var top = r.bottom + 8;
      if (top + t.height > window.innerHeight - 8) { top = Math.max(8, r.top - t.height - 8); }
      tip.style.left = Math.round(left) + "px"; tip.style.top = Math.round(top) + "px";
    }
    function show(node, delay) {
      if (!tip) {
        tip = document.createElement("div");
        tip.className = "s3-tip"; tip.setAttribute("role", "tooltip");
        document.body.appendChild(tip);
      }
      if (node === target) { return; }
      hide(); target = node;
      timer = setTimeout(place, delay);
    }
    document.addEventListener("pointerover", function (e) {
      if (e.pointerType && e.pointerType !== "mouse") { return; }
      var n = e.target.closest ? e.target.closest("[data-tip]") : null;
      if (n) { show(n, 120); } else { hide(); }
    });
    document.addEventListener("pointerdown", hide);
    document.addEventListener("scroll", hide, true);
    document.addEventListener("focusin", function (e) {
      var n = e.target.closest ? e.target.closest("[data-tip]") : null;
      if (n && e.target.matches && e.target.matches(":focus-visible")) { show(n, 0); }
    });
    document.addEventListener("focusout", hide);
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") { hide(); } });
  })();

  S3.confirmButton = function (button, armedText, action) {
    var timer = null, label = "";
    function disarm() {
      clearTimeout(timer);
      timer = null;
      button.classList.remove("is-armed");
      if (label) { button.textContent = label; }
    }
    button.addEventListener("click", function () {
      if (timer !== null) { disarm(); action(); return; }
      label = button.textContent;
      button.textContent = armedText;
      button.classList.add("is-armed");
      timer = setTimeout(disarm, 4000);
    });
    button.addEventListener("blur", function () { if (timer !== null) { disarm(); } });
    return { disarm: disarm };
  };

  S3.showError = function (id, text) {
    var node = S3.$(id);
    node.textContent = text;
    node.hidden = !text;
  };

  // The label of a stage as the server's schema gives it.
  S3.stageLabel = function (id) {
    var schema = S3.schema, list = schema && schema.stages || [];
    for (var i = 0; i < list.length; i++) { if (list[i].id === id) { return list[i].label; } }
    return id || "";
  };

  // "A、B 和 C"
  S3.joinLabels = function (ids) {
    var names = ids.map(S3.stageLabel);
    if (names.length <= 1) { return names.join(""); }
    return names.slice(0, -1).join("、") + " 和 " + names[names.length - 1];
  };
})();
