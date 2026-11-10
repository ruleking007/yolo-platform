/* YOLO 训练平台 —— 前端单页应用 */
"use strict";

const $ = (id) => document.getElementById(id);
const api = async (path, opts = {}) => {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
};
const postJSON = (path, body) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}) });
const del = (path) => api(path, { method: "DELETE" });

function toast(msg, type = "") {
  const el = document.createElement("div");
  el.className = "toast " + type;
  el.textContent = msg;
  $("toast-box").appendChild(el);
  setTimeout(() => el.remove(), 4200);
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtSize = (b) => b > 1e9 ? (b / 1e9).toFixed(2) + " GB" : b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : (b / 1e3).toFixed(0) + " KB";
const fmtTime = (t) => t ? new Date(t * 1000).toLocaleString("zh-CN", { hour12: false }) : "-";
const fmtDur = (s) => { if (!s) return "-"; s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60); return h ? `${h}h${m}m` : m ? `${m}m${s % 60}s` : `${s}s`; };
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

/* ---------------- Tab 切换 ---------------- */
const pages = ["dashboard", "datasets", "annotate", "training", "capture", "predict"];
document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  pages.forEach(p => $("page-" + p).classList.toggle("active", p === btn.dataset.tab));
  onEnter(btn.dataset.tab);
}));
function onEnter(tab) {
  if (tab === "dashboard") loadDashboard();
  if (tab === "datasets") loadDatasets();
  if (tab === "annotate") anInit();
  if (tab === "training") {
    loadTaskTable();
    fillDatasetSelect($("tr-ds"));
    loadModels();            // 模型仓库已合并到本页
    fillRepoSelect();
    checkNativeDialog();
    augRender();
  }
  if (tab === "capture") capInit();
  if (tab === "predict") fillModelSelect($("pd-model"));
}

/* ---------------- 总览 ---------------- */
async function loadDashboard() {
  try {
    const s = await api("/api/system");
    $("sysstats").innerHTML =
      `CPU ${s.cpu.percent}% · ${s.cpu.count} 核<br>内存 ${s.memory.used_gb}/${s.memory.total_gb} GB (${s.memory.percent}%)<br>` +
      `torch ${s.torch.version} · ${s.torch.cuda ? "CUDA ✓" : "仅 CPU"}`;
    $("env-info").innerHTML = [
      ["Python", s.python], ["系统", s.platform], ["CPU 核心", s.cpu.count],
      ["内存", `${s.memory.total_gb} GB`], ["PyTorch", s.torch.version],
      ["训练设备", s.torch.cuda ? "CUDA GPU" : "CPU（无 NVIDIA 显卡）"],
      ["数据集", s.datasets], ["训练任务", s.tasks], ["模型", s.models],
    ].map(([k, v]) => `<div><span>${k}：</span><b>${esc(v)}</b></div>`).join("");
    $("dash-cards").innerHTML = `
      <div class="card"><span class="card-t">${icon("folder")}数据集</span><b>${s.datasets}</b></div>
      <div class="card"><span class="card-t">${icon("trending")}训练任务</span><b>${s.tasks}</b></div>
      <div class="card"><span class="card-t">${icon("layers")}模型</span><b>${s.models}</b></div>
      <div class="card"><span class="card-t">${icon("cpu")}训练设备</span><b>${s.torch.cuda ? "GPU" : "CPU"}</b></div>`;
    const tasks = await api("/api/tasks");
    $("dash-recent").innerHTML = tasks.length ? taskTableHTML(tasks.slice(0, 6)) :
      '<p class="muted">还没有训练任务。点上方「快速开始」生成演示数据并训练。</p>';
  } catch (e) { $("sysstats").textContent = "服务未就绪: " + e.message; }
}

async function quickDemo() {
  toast("正在生成演示数据集…");
  try {
    const ds = await postJSON("/api/demo", { count: 120 });
    toast(`演示数据集已生成（${ds.n_images} 张），启动训练…`, "ok");
    const t = await postJSON("/api/tasks", {
      dataset_id: ds.id, name: "demo-形状识别", model: "yolo11n",
      epochs: 20, imgsz: 320, batch: 16, workers: 0 });
    document.querySelector('[data-tab="training"]').click();
    openTaskDetail(t.id);
  } catch (e) { toast("失败: " + e.message, "err"); }
}

/* ---------------- 纯图片导入向导（客户只给图片） ---------------- */
const WIZ = { files: [], dsId: null, running: false, batchSize: 12 };
function wizBind() {
  const zone = $("wi-drop");
  ["dragover", "dragenter"].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove("over"); }));
  zone.addEventListener("drop", e => {
    const fs = e.dataTransfer.files;
    if (fs && fs.length) { WIZ.files = [...fs]; wizShowPicked(); }
  });
  $("wi-file").addEventListener("change", e => { WIZ.files = [...e.target.files]; wizShowPicked(); });
  $("wi-name").addEventListener("keyup", wizShowPicked);
}
function wizShowPicked() {
  const n = WIZ.files.length;
  const total = WIZ.files.reduce((s, f) => s + f.size, 0);
  $("wi-picked").textContent = n ? `已选择 ${n} 个文件（约 ${fmtSize(total)}）` : "";
  $("wi-start").disabled = n === 0 || WIZ.running;
}
function wizLog(msg, cls = "") {
  const el = $("wi-log");
  el.innerHTML += `<div class="${cls}">${esc(msg)}</div>`;
  el.scrollTop = el.scrollHeight;
}
async function wizardUpload() {
  if (WIZ.running) return;
  const name = $("wi-name").value.trim();
  const classes = $("wi-classes").value.split(/[,，]/).map(s => s.trim()).filter(Boolean);
  if (!WIZ.files.length) return toast("请先选择图片", "err");
  if (!name) return toast("请填写数据集名称", "err");

  WIZ.running = true;
  $("wi-start").disabled = true;
  $("wi-bar-wrap").style.display = "";
  $("wi-log").innerHTML = "";
  $("wi-status").textContent = "创建数据集…";
  wizSetBar(0);

  try {
    const ds = await postJSON("/api/datasets/from-images", { name, classes });
    WIZ.dsId = ds.id;
    wizLog(`${icon("check-circle",14)} 已创建数据集「${esc(name)}」（id ${ds.id}）`, "good");

    const all = WIZ.files.filter(f => f.size > 0);
    const batches = [];
    for (let i = 0; i < all.length; i += WIZ.batchSize) batches.push(all.slice(i, i + WIZ.batchSize));
    let sent = 0, added = 0, dup = 0, bad = 0;
    const errors = [];

    for (const [bi, batch] of batches.entries()) {
      const fd = new FormData();
      batch.forEach(f => fd.append("files", f, f.name));
      fd.append("split", "train");
      fd.append("max_side", "1920");
      fd.append("dedupe", "true");
      $("wi-status").innerHTML = `<span class="spin"></span> 上传第 ${bi + 1}/${batches.length} 批（${sent + batch.length}/${all.length}）…`;
      try {
        const r = await api(`/api/datasets/${WIZ.dsId}/ingest`, { method: "POST", body: fd });
        added += r.added; dup += r.skipped_duplicate; bad += r.skipped_invalid;
        r.errors.forEach(e => errors.push(e));
        wizLog(`第 ${bi + 1} 批：入库 ${r.added} 张${r.skipped_duplicate ? `，去重 ${r.skipped_duplicate}` : ""}${r.skipped_invalid ? `，跳过 ${r.skipped_invalid}` : ""}`);
        if (r.skipped_invalid) r.errors.forEach(e => wizLog("  " + icon("alert",14) + " " + e, "bad"));
      } catch (e) {
        wizLog(`${icon("x-circle",14)} 第 ${bi + 1} 批失败：${e.message}`, "bad");
      }
      sent += batch.length;
      wizSetBar(sent / all.length * 0.95);
    }

    $("wi-status").innerHTML = `<span class="spin"></span> 正在划分训练/验证集…`;
    const fin = await postJSON(`/api/datasets/${WIZ.dsId}/finalize`,
      { classes: classes.length ? classes : undefined, split_ratio: +$("wi-split").value || 0.8 });
    wizSetBar(1);

    wizLog(`${icon("check-circle",14)} 完成：入库 ${added} 张（去重 ${dup}，跳过 ${bad}）`, "good");
    wizLog(`${icon("bar-chart",14)} train ${fin.n_train} / val ${fin.n_val}，共 ${fin.n_boxes ?? fin.boxes} 个框`, "good");
    $("wi-status").innerHTML = added
      ? `导入完成：${added} 张图片。下一步请点下方按钮进入在线标注。`
      : "没有可用图片入库，请检查文件格式。";

    const btn = document.createElement("button");
    btn.className = "btn primary";
    btn.innerHTML = icon("pen",14) + " 去标注这 " + added + " 张图片";
    btn.onclick = () => gotoAnnotate(WIZ.dsId);
    $("wi-status").appendChild(btn);

    WIZ.files = []; $("wi-file").value = ""; $("wi-picked").textContent = "";
    loadDatasets();
  } catch (e) {
    wizLog(icon("x-circle",14) + " " + e.message, "bad");
    $("wi-status").textContent = "导入失败：" + e.message;
  } finally {
    WIZ.running = false;
    $("wi-start").disabled = WIZ.files.length === 0;
  }
}
function wizSetBar(p) { $("wi-bar").style.width = (Math.max(0, Math.min(1, p)) * 100).toFixed(1) + "%"; }

/* ---------------- 数据集 ---------------- */
const dz = { over: false };
function bindDropzone() {
  const zone = $("ds-drop");
  ["dragover", "dragenter"].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove("over"); }));
  zone.addEventListener("drop", e => {
    const f = e.dataTransfer.files[0];
    if (f) { $("ds-file").files = e.dataTransfer.files; toast(`已选择 ${f.name}`); }
  });
}
async function uploadDataset() {
  const f = $("ds-file").files[0];
  if (!f) return toast("请先选择文件", "err");
  const fd = new FormData();
  fd.append("file", f);
  fd.append("name", $("ds-name").value);
  fd.append("task", $("ds-task").value);
  fd.append("split_ratio", $("ds-split").value);
  $("upload-progress").innerHTML = `<span class="spin"></span> 正在上传并导入 ${esc(f.name)}（大文件需要一点时间）…`;
  try {
    const r = await api("/api/datasets/upload", { method: "POST", body: fd });
    $("upload-progress").textContent = "";
    toast(`导入成功：${r.n_images} 张图，${r.n_labeled} 张已标注`, "ok");
    loadDatasets();
  } catch (e) {
    $("upload-progress").textContent = "";
    toast("导入失败: " + e.message, "err");
  }
}
async function createDataset() {
  const name = $("newds-name").value.trim();
  const classes = $("newds-classes").value.split(/[,，]/).map(s => s.trim()).filter(Boolean);
  if (!name || !classes.length) return toast("请填写名称和类别（逗号分隔）", "err");
  try {
    const r = await postJSON("/api/datasets", { name, classes });
    toast("数据集已创建", "ok");
    document.querySelector('[data-tab="annotate"]').click();
    setTimeout(() => { $("an-ds").value = r.id; anLoadList(); }, 300);
    loadDatasets();
  } catch (e) { toast(e.message, "err"); }
}
async function loadDatasets() {
  const list = await api("/api/datasets");
  $("ds-list").innerHTML = list.map(d => `
    <div class="ds-card">
      <h4>${esc(d.name)} <span class="muted small">${esc(d.id)}</span></h4>
      <div class="ds-stats">
        <span>${icon("image")} ${d.n_images}</span><span>${icon("pen")} ${d.n_labeled}</span>
        <span>train ${d.n_train}</span><span>val ${d.n_val}</span>
        <span>${icon("tag")} ${d.boxes} 框</span><span>${d.classes.length} 类</span>
      </div>
      <div class="muted small" style="margin-bottom:8px">${esc(d.classes.slice(0, 8).join(", "))}${d.classes.length > 8 ? "…" : ""}</div>
      <div class="row gap wrap">
        <button class="btn small primary" onclick="gotoAnnotate('${d.id}')">${icon("pen",14)}标注</button>
        <button class="btn small" onclick="gotoTrain('${d.id}')">${icon("trending",14)}训练</button>
        <button class="btn small" onclick="appendImages('${d.id}')">${icon("plus",14)}加图</button>
        <button class="btn small" onclick="resplit('${d.id}')">${icon("refresh",14)}重新划分</button>
        <a class="btn small" href="/api/datasets/${d.id}/export">${icon("download",14)}导出</a>
        <button class="btn small" onclick="renameDs('${d.id}')">${icon("pen",14)}改名</button>
        <button class="btn small danger" onclick="deleteDs('${d.id}','${esc(d.name)}')">${icon("trash",14)}删除</button>
      </div>
    </div>`).join("") || '<p class="muted">暂无数据集，先导入或创建。</p>';
}
function gotoAnnotate(id) { document.querySelector('[data-tab="annotate"]').click(); setTimeout(() => { $("an-ds").value = id; anLoadList(); }, 250); }
function gotoTrain(id) { document.querySelector('[data-tab="training"]').click(); setTimeout(() => $("tr-ds").value = id, 250); }

/* 给已有数据集追加图片（客户后续补图） */
function appendImages(dsId) {
  const inp = document.createElement("input");
  inp.type = "file";
  inp.accept = "image/*,.heic,.heif";
  inp.multiple = true;
  inp.onchange = async () => {
    const files = [...inp.files].filter(f => f.size > 0);
    if (!files.length) return;
    toast(`正在追加 ${files.length} 张图片…`);
    let added = 0, dup = 0, bad = 0;
    const B = 12;
    for (let i = 0; i < files.length; i += B) {
      const batch = files.slice(i, i + B);
      const fd = new FormData();
      batch.forEach(f => fd.append("files", f, f.name));
      fd.append("split", "train");
      fd.append("max_side", "1920");
      fd.append("dedupe", "true");
      try {
        const r = await api(`/api/datasets/${dsId}/ingest`, { method: "POST", body: fd });
        added += r.added; dup += r.skipped_duplicate; bad += r.skipped_invalid;
      } catch (e) { toast("批次失败：" + e.message, "err"); }
    }
    await postJSON(`/api/datasets/${dsId}/finalize`, { split_ratio: 0.8 });
    toast(`已追加 ${added} 张（去重 ${dup}，跳过 ${bad}）`, "ok");
    loadDatasets();
  };
  inp.click();
}
async function renameDs(id) {
  const name = prompt("新名称："); if (!name) return;
  await api(`/api/datasets/${id}/rename`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
  loadDatasets();
}
async function resplit(id) {
  const ratio = prompt("训练集占比（0-1）", "0.8"); if (!ratio) return;
  const r = await postJSON(`/api/datasets/${id}/split`, { ratio: parseFloat(ratio) });
  toast(`已重新划分：train ${r.n_train} / val ${r.n_val}`, "ok"); loadDatasets();
}
async function deleteDs(id, name) {
  if (!confirm(`确定删除数据集「${name}」及其全部图片标注？`)) return;
  await del(`/api/datasets/${id}`); toast("已删除"); loadDatasets();
}
async function fillDatasetSelect(sel) {
  const list = await api("/api/datasets");
  const cur = sel.value;
  sel.innerHTML = list.map(d => `<option value="${d.id}">${esc(d.name)} (${d.n_images}图/${d.n_labeled}标注)</option>`).join("");
  if (cur) sel.value = cur;
}

/* ---------------- 标注 ---------------- */
const PALETTE = ["#ff5252", "#4c8dff", "#35c98e", "#ffb454", "#c77dff", "#5ad2f4", "#ff8fab", "#9ccc65", "#f06292", "#ba68c8"];
const an = {
  ds: "", split: "train", items: [], idx: -1, img: null, boxes: [],
  sel: -1, cls: 0, classes: [], drag: null, imgW: 0, imgH: 0, dirty: false, undoStack: [],
  shape: "box", poly: null,
};
async function anInit() {
  const sel = $("an-ds");
  if (!sel.options.length) await fillDatasetSelect(sel);
  if (!an.ds && sel.value) an.ds = sel.value;
  await anLoadList();
}
async function anLoadList() {
  an.ds = $("an-ds").value;
  if (!an.ds) { $("an-thumbs").innerHTML = '<p class="muted small">请先创建/导入数据集</p>'; return; }
  an.split = $("an-split").value;
  const detail = await api(`/api/datasets/${an.ds}`);
  an.classes = detail.classes;
  an.renderClasses();
  anUpdateProgress(detail);

  const empt = detail.n_images === 0;
  $("an-empty").style.display = empt ? "" : "none";
  document.querySelector(".annotate-layout").style.display = empt ? "none" : "";

  const r = await api(`/api/datasets/${an.ds}/images?split=${an.split}&labeled=${$("an-filter").value}&q=${encodeURIComponent($("an-q").value)}&limit=500`);
  an.items = r.items;
  $("an-count").textContent = `${r.total} 张`;
  $("an-thumbs").innerHTML = an.items.map((it, i) => `
    <div class="thumb ${i === an.idx ? "sel" : ""}" onclick="anOpen(${i})" title="${esc(it.name)}">
      <img loading="lazy" src="/api/datasets/${an.ds}/thumb/${it.split}/${encodeURIComponent(it.name)}">
      <div><div class="t-name">${esc(it.name)}</div>
      <div class="t-box ${it.labeled ? "ok" : ""}">${it.n_boxes ? it.n_boxes + " 框" : "未标注"}</div></div>
    </div>`).join("") || '<p class="muted small">当前筛选下没有图片。</p>';
  if (empt) { anClearCanvas(); return; }
  if (an.idx >= 0 && an.idx < an.items.length) anOpen(an.idx);
  else if (an.items.length) anOpen(0);
  else anClearCanvas();
}

function anUpdateProgress(detail) {
  const total = detail.n_images || 0;
  const done = detail.n_labeled || 0;
  const pct = total ? done / total : 0;
  $("an-prog-bar").style.width = (pct * 100).toFixed(1) + "%";
  $("an-prog-text").textContent = total
    ? `已标注 ${done} / ${total} 张（${(pct * 100).toFixed(0)}%）`
    : "还没有图片";
  $("an-prog-sub").textContent = total
    ? `剩余 ${total - done} 张 · 类别 ${detail.classes.length} 个 · 框 ${detail.boxes} 个`
    : "在上方「数据集」页上传图片后即可开始标注";
  $("an-gotrain").style.display = (done > 0) ? "" : "none";
}

/* 标注完成 → 直接带入训练页 */
function anReadyToTrain() {
  const unlabeled = an.items.filter(i => !i.labeled).length;
  if (unlabeled > 0 && !confirm(`还有 ${unlabeled} 张（当前筛选下）未标注，仍要开始训练吗？`)) {
    $("an-filter").value = "no"; anLoadList();
    return;
  }
  gotoTrain(an.ds);
}
an.renderClasses = function () {
  $("an-classes").innerHTML = an.classes.map((c, i) => `
    <div class="class-item ${i === an.cls ? "sel" : ""}" onclick="an.cls=${i};an.renderClasses()">
      <span class="dot" style="background:${PALETTE[i % PALETTE.length]}"></span>
      <span style="flex:1">${i + 1}. ${esc(c)}</span>
      <button class="btn small" style="padding:1px 6px" title="删除类别" onclick="event.stopPropagation();anDelClass(${i})">×</button>
    </div>`).join("");
};
function anAddClass() {
  const name = prompt("新类别名称："); if (!name) return;
  an.classes.push(name.trim());
  api(`/api/datasets/${an.ds}/classes`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ classes: an.classes }) })
    .then(() => an.renderClasses());
}
async function anDelClass(i) {
  if (!confirm(`删除类别「${an.classes[i]}」？已用该类别的标注框也会被移除。`)) return;
  an.classes.splice(i, 1);
  await api(`/api/datasets/${an.ds}/classes`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ classes: an.classes }) });
  an.boxes = an.boxes.filter(b => b.cls !== i).map(b => b.cls > i ? { ...b, cls: b.cls - 1 } : b);
  an.renderClasses(); anDraw();
}
function anClearCanvas() {
  an.img = null; an.boxes = []; an.idx = -1;
  const cv = $("an-canvas"), ctx = cv.getContext("2d");
  cv.width = 800; cv.height = 420;
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = "#555"; ctx.font = "16px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("没有可显示的图片", 400, 210);
  $("an-imgname").textContent = "";
}
async function anOpen(i) {
  // 翻页自动保存：默认开启，避免客户忘记保存丢标注
  if (an.dirty && $("an-autosave").checked && an.idx >= 0) {
    try { await anSave(true); } catch (e) { toast("自动保存失败：" + e.message, "err"); }
  } else if (an.dirty && !confirm("当前图有未保存修改，放弃并切换？")) {
    return;
  }
  an.idx = i; an.dirty = false; an.undoStack = []; anCancelPoly();
  const it = an.items[i]; if (!it) return;
  an.split = it.split;
  $("an-imgname").textContent = `${it.split}/${it.name}`;
  document.querySelectorAll(".thumb").forEach((el, k) => el.classList.toggle("sel", k === i));
  an.img = new Image();
  an.img.onload = async () => {
    const cv = $("an-canvas");
    const maxW = cv.parentElement.clientWidth - 24;
    const scale = Math.min(1, maxW / an.img.naturalWidth, 1600 / an.img.naturalHeight);
    cv.width = Math.round(an.img.naturalWidth * scale);
    cv.height = Math.round(an.img.naturalHeight * scale);
    an.imgW = an.img.naturalWidth; an.imgH = an.img.naturalHeight;
    const r = await api(`/api/datasets/${an.ds}/label/${it.split}/${encodeURIComponent(it.stem)}`);
    an.task = r.task || "detect";
    an.boxes = (r.boxes || []).map(b => ({ ...b }));
    an.sel = -1; anUpdateShapeHint(); anDraw();
  };
  an.img.src = `/api/datasets/${an.ds}/image/${it.split}/${encodeURIComponent(it.name)}?t=${Date.now()}`;
}

/* 按任务类型提示形状的实际保存方式 */
function anUpdateShapeHint() {
  const el = $("an-help");
  if (!el) return;
  const base = "拖拽画框 · 点击选中拖动 · 角点缩放 · 数字键 1-9 切换类别 · B/C/E/P 切换形状 · 多边形连续点击加点，双击/Enter 闭合，Esc 取消，Ctrl+Z 撤销上一个点";
  el.textContent = an.task === "segment"
    ? "分割数据集：圆/椭圆/矩形保存时自动转为多边形轮廓 · " + base
    : "检测数据集：圆/椭圆/多边形按外接框参与训练（原始形状会保留，随时可改）· " + base;
}
/* ---------- 形状几何：归一化 <-> 画布像素 ---------- */
function shapeGeom(b, cv) {
  if (b.shape === "polygon") {
    const pts = [];
    for (let i = 0; i < b.points.length; i += 2)
      pts.push([b.points[i] * cv.width, b.points[i + 1] * cv.height]);
    return { kind: "polygon", pts };
  }
  if (b.shape === "circle" || b.shape === "ellipse")
    return { kind: b.shape, cx: b.cx * cv.width, cy: b.cy * cv.height,
             rx: b.rx * cv.width, ry: b.ry * cv.height };
  return { kind: "box", x: (b.cx - b.w / 2) * cv.width, y: (b.cy - b.h / 2) * cv.height,
           w: b.w * cv.width, h: b.h * cv.height };
}

/* 可拖拽把手：box 四角 / 圆与椭圆四轴点 / 多边形顶点 + 边中点(插入) */
function shapeHandles(b, cv) {
  const g = shapeGeom(b, cv);
  if (g.kind === "box")
    return [{ x: g.x, y: g.y, id: "tl" }, { x: g.x + g.w, y: g.y, id: "tr" },
            { x: g.x, y: g.y + g.h, id: "bl" }, { x: g.x + g.w, y: g.y + g.h, id: "br" }];
  if (g.kind === "circle" || g.kind === "ellipse")
    return [{ x: g.cx + g.rx, y: g.cy, id: "r" }, { x: g.cx - g.rx, y: g.cy, id: "l" },
            { x: g.cx, y: g.cy - g.ry, id: "t" }, { x: g.cx, y: g.cy + g.ry, id: "b" }];
  const hs = g.pts.map(([x, y], i) => ({ x, y, id: "v" + i }));
  for (let i = 0; i < g.pts.length; i++) {
    const [x1, y1] = g.pts[i], [x2, y2] = g.pts[(i + 1) % g.pts.length];
    hs.push({ x: (x1 + x2) / 2, y: (y1 + y2) / 2, id: "m" + i });
  }
  return hs;
}

function pointInPoly(p, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if (((yi > p.y) !== (yj > p.y)) && (p.x < (xj - xi) * (p.y - yi) / (yj - yi) + xi))
      inside = !inside;
  }
  return inside;
}

function anDraw() {
  const cv = $("an-canvas"); if (!an.img) return;  const ctx = cv.getContext("2d");
  ctx.drawImage(an.img, 0, 0, cv.width, cv.height);

  an.boxes.forEach((b, i) => {
    const color = PALETTE[(b.cls || 0) % PALETTE.length];
    const g = shapeGeom(b, cv);
    ctx.lineWidth = i === an.sel ? 3 : 1.6;
    ctx.strokeStyle = color; ctx.fillStyle = color;
    let lx, ly;
    if (g.kind === "box") {
      ctx.strokeRect(g.x, g.y, g.w, g.h);
      lx = g.x; ly = g.y;
    } else if (g.kind === "circle" || g.kind === "ellipse") {
      ctx.beginPath();
      ctx.ellipse(g.cx, g.cy, Math.max(1, g.rx), Math.max(1, g.ry), 0, 0, Math.PI * 2);
      ctx.stroke();
      if (i === an.sel) {
        ctx.save(); ctx.lineWidth = 1; ctx.beginPath();
        ctx.moveTo(g.cx - 7, g.cy); ctx.lineTo(g.cx + 7, g.cy);
        ctx.moveTo(g.cx, g.cy - 7); ctx.lineTo(g.cx, g.cy + 7); ctx.stroke(); ctx.restore();
      }
      lx = g.cx - g.rx; ly = g.cy - g.ry;
    } else {
      ctx.beginPath();
      g.pts.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.closePath(); ctx.stroke();
      ctx.globalAlpha = .12; ctx.fill(); ctx.globalAlpha = 1;
      lx = Math.min(...g.pts.map(q => q[0])); ly = Math.min(...g.pts.map(q => q[1]));
    }
    const label = `${an.classes[b.cls] ?? b.cls}` + (b.conf ? ` ${(+b.conf).toFixed(2)}` : "");
    ctx.font = "12px sans-serif";
    const tw = ctx.measureText(label).width + 8;
    ctx.fillStyle = color; ctx.fillRect(lx, Math.max(0, ly - 16), tw, 16);
    ctx.fillStyle = "#08101e"; ctx.fillText(label, lx + 4, Math.max(12, ly - 4));
    if (i === an.sel) {
      shapeHandles(b, cv).forEach(h => {
        ctx.fillStyle = h.id.startsWith("m") ? "rgba(255,255,255,.5)" : "#fff";
        ctx.fillRect(h.x - 3, h.y - 3, 6, 6);
      });
    }
  });

  // 绘制中的预览
  const pc = PALETTE[an.cls % PALETTE.length];
  ctx.setLineDash([5, 4]); ctx.strokeStyle = pc; ctx.lineWidth = 1.6;
  if (an.drag && an.drag.mode === "new") {
    const d = an.drag;
    if (d.shape === "box")
      ctx.strokeRect(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0));
    else if (d.shape === "circle") {
      const r = Math.hypot(d.x1 - d.x0, d.y1 - d.y0);
      ctx.beginPath(); ctx.arc(d.x0, d.y0, r, 0, 7); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(d.x0, d.y0); ctx.lineTo(d.x1, d.y1); ctx.stroke();
    } else if (d.shape === "ellipse") {
      ctx.beginPath();
      ctx.ellipse((d.x0 + d.x1) / 2, (d.y0 + d.y1) / 2, Math.abs(d.x1 - d.x0) / 2, Math.abs(d.y1 - d.y0) / 2, 0, 0, 7);
      ctx.stroke();
      ctx.strokeRect(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0));
    }
  }
  if (an.poly && an.poly.points.length) {
    const pts = an.poly.points;
    ctx.beginPath();
    pts.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    if (an.poly.cursor) ctx.lineTo(an.poly.cursor[0], an.poly.cursor[1]);
    ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = pc;
    pts.forEach(([x, y], k) => { ctx.beginPath(); ctx.arc(x, y, k === 0 ? 5.5 : 3, 0, 7); ctx.fill(); });
  }
  ctx.setLineDash([]);
}
/* 覆盖层：跟随鼠标的 labelme 式全画布细十字（黑白双线，深浅图都可见） */
function drawOverlay(p) {
  const cv = $("an-canvas"), ov = $("an-overlay");
  if (!ov || !cv) return;
  if (ov.width !== cv.width || ov.height !== cv.height) {
    ov.width = cv.width; ov.height = cv.height;
  }
  const ctx = ov.getContext("2d");
  ctx.clearRect(0, 0, ov.width, ov.height);
  if (!an.img || !p) return;
  const x = Math.round(p.x) + .5, y = Math.round(p.y) + .5;
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(255,255,255,.75)";
  ctx.beginPath();
  ctx.moveTo(x, 0); ctx.lineTo(x, ov.height);
  ctx.moveTo(0, y); ctx.lineTo(ov.width, y);
  ctx.stroke();
  ctx.strokeStyle = "rgba(0,0,0,.55)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x, 0); ctx.lineTo(x, ov.height);
  ctx.moveTo(0, y); ctx.lineTo(ov.width, y);
  ctx.stroke();
  ctx.setLineDash([]);
}

function canvasPos(e) {
  const cv = $("an-canvas"), r = cv.getBoundingClientRect();
  return { x: (e.clientX - r.left) * cv.width / r.width, y: (e.clientY - r.top) * cv.height / r.height };
}
function hitBody(b, p, cv) {
  const g = shapeGeom(b, cv);
  if (g.kind === "box")
    return p.x > g.x && p.x < g.x + g.w && p.y > g.y && p.y < g.y + g.h;
  if (g.kind === "circle" || g.kind === "ellipse") {
    const dx = (p.x - g.cx) / Math.max(2, g.rx), dy = (p.y - g.cy) / Math.max(2, g.ry);
    return dx * dx + dy * dy <= 1.1;
  }
  return pointInPoly(p, g.pts);
}

function hitBox(p) {
  const cv = $("an-canvas");
  // 1. 选中形状的把手优先
  if (an.sel >= 0 && an.boxes[an.sel]) {
    for (const h of shapeHandles(an.boxes[an.sel], cv))
      if (Math.hypot(p.x - h.x, p.y - h.y) < 8) return { i: an.sel, handle: h.id };
  }
  // 2. 命中形状主体（从上往下）
  for (let i = an.boxes.length - 1; i >= 0; i--)
    if (hitBody(an.boxes[i], p, cv)) return { i, handle: "body" };
  return null;
}

function anSetShape(sh) {
  an.shape = sh;
  document.querySelectorAll(".shape-tools .btn").forEach(b =>
    b.classList.toggle("active", b.dataset.shape === sh));
  if (an.poly && sh !== "polygon") anCancelPoly();
  $("an-canvas").style.cursor = shapeCursor();
}

/* labelme 风格细十字光标：1px 黑线 + 白色描边（深浅背景都清晰），热点在中心。
   系统 crosshair 是 OS 画的粗十字，没法调粗细，所以用 SVG data-URL 自绘。 */
const CURSOR_CROSS = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='21' height='21'%3E%3Cline x1='10' y1='0' x2='10' y2='21' stroke='white' stroke-width='2.5'/%3E%3Cline x1='0' y1='10' x2='21' y2='10' stroke='white' stroke-width='2.5'/%3E%3Cline x1='10' y1='0' x2='10' y2='21' stroke='black' stroke-width='1'/%3E%3Cline x1='0' y1='10' x2='21' y2='10' stroke='black' stroke-width='1'/%3E%3C/svg%3E") 10 10, crosshair`;
const CURSOR_POLY = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='21' height='21'%3E%3Cline x1='10' y1='0' x2='10' y2='21' stroke='white' stroke-width='2.5'/%3E%3Cline x1='0' y1='10' x2='21' y2='10' stroke='white' stroke-width='2.5'/%3E%3Cline x1='10' y1='0' x2='10' y2='21' stroke='black' stroke-width='1'/%3E%3Cline x1='0' y1='10' x2='21' y2='10' stroke='black' stroke-width='1'/%3E%3Ccircle cx='10' cy='10' r='2.5' fill='none' stroke='white' stroke-width='2'/%3E%3Ccircle cx='10' cy='10' r='2.5' fill='none' stroke='black' stroke-width='1'/%3E%3C/svg%3E") 10 10, crosshair`;

function shapeCursor() { return an.shape === "polygon" ? CURSOR_POLY : CURSOR_CROSS; }

function handleCursor(id) {
  if (id.startsWith("v") || id.startsWith("m")) return CURSOR_POLY;   // 多边形顶点/边中点
  switch (id) {
    case "tl": case "br": return "nwse-resize";
    case "tr": case "bl": return "nesw-resize";
    case "l": case "r": return "ew-resize";
    case "t": case "b": return "ns-resize";
  }
  return CURSOR_CROSS;
}

/* 悬停反馈：把手→方向箭头，形状内部→move，空白→当前工具细十字 */
function hoverCursor(p) {
  const cv = $("an-canvas");
  const hit = hitBox(p);
  let cur = shapeCursor();
  if (hit) cur = hit.handle === "body" ? "move" : handleCursor(hit.handle);
  if (cv.style.cursor !== cur) cv.style.cursor = cur;
}
function anCancelPoly() {
  if (an.poly) { an.poly = null; anDraw(); }
}
function anFinishPoly() {
  if (!an.poly || an.poly.points.length < 3) { anCancelPoly(); return; }
  const cv = $("an-canvas");
  // 去掉距离过近的相邻重复点（双击/误点产生）
  const pts = [];
  an.poly.points.forEach(([x, y]) => {
    const last = pts[pts.length - 1];
    if (!last || Math.hypot(x - last[0], y - last[1]) > 3) pts.push([x, y]);
  });
  if (pts.length >= 3 && Math.hypot(pts[0][0] - pts[pts.length - 1][0],
                                    pts[0][1] - pts[pts.length - 1][1]) < 3) pts.pop();
  if (pts.length < 3) { anCancelPoly(); toast("多边形至少需要 3 个顶点"); return; }
  const flat = [];
  pts.forEach(([x, y]) => { flat.push(x / cv.width, y / cv.height); });
  pushUndo();
  an.boxes.push({ shape: "polygon", cls: an.poly.cls, points: flat });
  an.sel = an.boxes.length - 1; an.dirty = true; an.poly = null;
  anDraw();
}

function bindCanvas() {
  const cv = $("an-canvas");
  cv.addEventListener("mousedown", e => {
    if (!an.img) return;
    if (e.button !== 0) return;
    const p = canvasPos(e);

    // 多边形绘制中：点击加点
    if (an.poly) {
      const first = an.poly.points[0];
      if (an.poly.points.length >= 3 && Math.hypot(p.x - first[0], p.y - first[1]) < 9) {
        anFinishPoly(); return;
      }
      an.poly.points.push([p.x, p.y]); an.poly.cursor = p; anDraw(); return;
    }

    const hit = hitBox(p);
    if (hit && hit.handle !== "body") {
      pushUndo();
      const b = an.boxes[hit.i];
      // 多边形边中点 → 插入顶点
      let vid = hit.handle;
      if (b.shape === "polygon" && hit.handle.startsWith("m")) {
        const idx = +hit.handle.slice(1);
        const [x1, y1] = [b.points[idx * 2] * cv.width, b.points[idx * 2 + 1] * cv.height];
        const [x2, y2] = [b.points[((idx + 1) % (b.points.length / 2)) * 2] * cv.width,
                          b.points[((idx + 1) % (b.points.length / 2)) * 2 + 1] * cv.height];
        b.points.splice((idx + 1) * 2, 0, (x1 + x2) / 2 / cv.width, (y1 + y2) / 2 / cv.height);
        vid = "v" + (idx + 1);
      }
      an.drag = { mode: "handle", vid, b: JSON.parse(JSON.stringify(b)), i: hit.i };
      an.sel = hit.i;
    } else if (hit) {
      an.sel = hit.i;
      pushUndo();
      // 深拷贝：polygon 的 points 数组不能共享引用，否则拖动会累积偏移
      an.drag = { mode: "move", start: p, b: JSON.parse(JSON.stringify(an.boxes[hit.i])) };
    } else if (an.shape === "polygon") {
      an.poly = { points: [[p.x, p.y]], cursor: p, cls: an.cls };
      anDraw(); return;
    } else {
      an.sel = -1;
      an.drag = { mode: "new", shape: an.shape, x0: p.x, y0: p.y, x1: p.x, y1: p.y };
    }
    an.renderClasses(); anDraw();
  });

  cv.addEventListener("mousemove", e => {
    const p = canvasPos(e);
    an.mouse = p;
    drawOverlay(p);                       // 细十字辅助线只画在覆盖层，不重绘主画布
    if (an.poly) { an.poly.cursor = p; anDraw(); return; }
    if (!an.drag) { hoverCursor(p); return; }
    const W = cv.width, H = cv.height;
    const d = an.drag;
    if (d.mode === "new") { d.x1 = p.x; d.y1 = p.y; }
    else if (d.mode === "move") {
      const b = an.boxes[an.sel];
      const dx = (p.x - d.start.x) / W, dy = (p.y - d.start.y) / H;
      if (b.shape === "polygon") {
        const o = d.b.points;
        for (let i = 0; i < o.length; i += 2) {
          b.points[i] = clamp01(o[i] + dx);
          b.points[i + 1] = clamp01(o[i + 1] + dy);
        }
      } else {
        b.cx = clamp01(d.b.cx + dx);
        b.cy = clamp01(d.b.cy + dy);
      }
    } else if (d.mode === "handle") {
      const b = an.boxes[an.sel], o = d.b;
      if (b.shape === "polygon") {
        const vi = +d.vid.slice(1);
        b.points[vi * 2] = clamp01(p.x / W);
        b.points[vi * 2 + 1] = clamp01(p.y / H);
      } else if (b.shape === "circle") {
        const cxp = o.cx * W, cyp = o.cy * H;
        const r = (d.vid === "l" || d.vid === "r") ? Math.abs(p.x - cxp) : Math.abs(p.y - cyp);
        b.rx = Math.max(.002, r / W); b.ry = Math.max(.002, r / H);
      } else if (b.shape === "ellipse") {
        if (d.vid === "l" || d.vid === "r") b.rx = Math.max(.002, Math.abs(p.x - o.cx * W) / W);
        else b.ry = Math.max(.002, Math.abs(p.y - o.cy * H) / H);
      } else {  // box：角点 id 同时含横/纵方向
        const nx = p.x / W, ny = p.y / H;
        const x1 = o.cx - o.w / 2, y1 = o.cy - o.h / 2, x2 = o.cx + o.w / 2, y2 = o.cy + o.h / 2;
        if (d.vid.includes("l")) { const v = Math.min(nx, x2 - .002); b.cx = (v + x2) / 2; b.w = x2 - v; }
        if (d.vid.includes("r")) { const v = Math.max(nx, x1 + .002); b.cx = (x1 + v) / 2; b.w = v - x1; }
        if (d.vid.includes("t")) { const v = Math.min(ny, y2 - .002); b.cy = (v + y2) / 2; b.h = y2 - v; }
        if (d.vid.includes("b")) { const v = Math.max(ny, y1 + .002); b.cy = (y1 + v) / 2; b.h = v - y1; }
      }
    }
    anDraw();
  });

  window.addEventListener("mouseup", () => {
    if (!an.drag) return;
    const d = an.drag, W = cv.width, H = cv.height;
    if (d.mode === "new") {
      const w = Math.abs(d.x1 - d.x0), h = Math.abs(d.y1 - d.y0);
      if (d.shape === "box" && w > 6 && h > 6) {
        pushUndo();
        an.boxes.push({ shape: "box", cls: an.cls,
          cx: ((d.x0 + d.x1) / 2) / W, cy: ((d.y0 + d.y1) / 2) / H, w: w / W, h: h / H });
        an.sel = an.boxes.length - 1; an.dirty = true;
      } else if (d.shape === "circle") {
        const r = Math.hypot(d.x1 - d.x0, d.y1 - d.y0);
        if (r > 5) { pushUndo();
          an.boxes.push({ shape: "circle", cls: an.cls, cx: d.x0 / W, cy: d.y0 / H, rx: r / W, ry: r / H });
          an.sel = an.boxes.length - 1; an.dirty = true; }
      } else if (d.shape === "ellipse" && w > 6 && h > 6) {
        pushUndo();
        an.boxes.push({ shape: "ellipse", cls: an.cls,
          cx: ((d.x0 + d.x1) / 2) / W, cy: ((d.y0 + d.y1) / 2) / H, rx: w / 2 / W, ry: h / 2 / H });
        an.sel = an.boxes.length - 1; an.dirty = true;
      }
    }
    an.drag = null; anDraw();
  });

  cv.addEventListener("dblclick", e => {
    if (an.poly) { e.preventDefault(); anFinishPoly(); }
  });
  cv.addEventListener("mouseleave", () => { an.mouse = null; drawOverlay(null); });
}
const clamp01 = v => Math.max(0, Math.min(1, v));

/* 画布区支持拖拽图片直接导入 */
function bindCanvasDrop() {
  const wrap = document.querySelector(".an-canvas-wrap");
  if (!wrap) return;
  ["dragover", "dragenter"].forEach(ev => wrap.addEventListener(ev, e => {
    e.preventDefault(); wrap.classList.add("drop-active");
  }));
  ["dragleave", "drop"].forEach(ev => wrap.addEventListener(ev, e => {
    e.preventDefault(); wrap.classList.remove("drop-active");
  }));
  wrap.addEventListener("drop", e => {
    const fs = e.dataTransfer.files;
    if (fs && fs.length) {
      if (!an.ds) return toast("请先选择数据集", "err");
      anUploadImages(fs);
    }
  });
}
function pushUndo() { an.undoStack.push(JSON.stringify(an.boxes)); if (an.undoStack.length > 30) an.undoStack.shift(); }
function anUndo() {
  const last = an.undoStack.pop();
  if (last !== undefined) { an.boxes = JSON.parse(last); an.sel = -1; an.dirty = true; anDraw(); }
}
function anDeleteBox() {
  if (an.sel < 0) return toast("先点击选中一个框", "");
  pushUndo(); an.boxes.splice(an.sel, 1); an.sel = -1; an.dirty = true; anDraw();
}
async function anSave(silent = false) {
  if (!an.img || an.idx < 0) return;
  const it = an.items[an.idx];
  const boxes = an.boxes.map(b => {
    if (b.shape === "polygon") return { shape: "polygon", cls: b.cls, points: b.points };
    if (b.shape === "circle" || b.shape === "ellipse")
      return { shape: b.shape, cls: b.cls, cx: b.cx, cy: b.cy, rx: b.rx, ry: b.ry };
    return { shape: "box", cls: b.cls, cx: b.cx, cy: b.cy, w: b.w, h: b.h };
  });
  await api(`/api/datasets/${an.ds}/label/${it.split}/${encodeURIComponent(it.stem)}`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ boxes }) });
  it.n_boxes = boxes.length; it.labeled = boxes.length > 0;
  an.dirty = false;
  const t = document.querySelectorAll(".thumb")[an.idx];
  if (t) {
    const el = t.querySelector(".t-box");
    el.textContent = boxes.length ? boxes.length + " 框" : "未标注";
    el.className = "t-box " + (boxes.length ? "ok" : "");
  }
  if (!silent) toast(`已保存 ${it.name}（${boxes.length} 框）`, "ok");
  // 轻量刷新进度条
  try { anUpdateProgress(await api(`/api/datasets/${an.ds}`)); } catch {}
}
function anPrev() { if (an.idx > 0) anOpen(an.idx - 1); else toast("已经是第一张"); }
function anNext() {
  if (an.idx < an.items.length - 1) anOpen(an.idx + 1);
  else anSaveThenNext();
}
async function anSaveThenNext() {
  if (an.dirty) await anSave(true);
  toast(`已到最后一张（共 ${an.items.length} 张），可切换筛选继续或去训练`, "ok");
  anUpdateProgress(await api(`/api/datasets/${an.ds}`));
}
function anAddImages() { $("an-file").click(); }

/* 用系统文件管理器批量选图（能一次选整个文件夹里的多张） */
async function anAddImagesNative() {
  if (!an.ds) return toast("请先选择数据集", "err");
  const picked = [];
  // 允许分多次选，每次多选；用户点取消即结束
  for (let round = 0; round < 5; round++) {
    const p = await pickNativeFile("images", null, true);
    if (!p) break;
    const r = window.__lastNativePaths || [p];
    picked.push(...r);
    if (!confirm(`已选 ${picked.length} 张，继续选更多？`)) break;
  }
  if (!picked.length) return;
  await ingestPaths(picked);
}

async function ingestPaths(paths) {
  toast(`正在导入 ${paths.length} 张图片…`);
  $("an-prog-text").textContent = "导入图片中…";
  try {
    const r = await postJSON(`/api/datasets/${an.ds}/ingest-paths`, {
      paths, split: "train", max_side: 1920, dedupe: true });
    await postJSON(`/api/datasets/${an.ds}/finalize`, { split_ratio: 0.8 });
    toast(`已导入 ${r.added} 张（去重 ${r.skipped_duplicate}，跳过 ${r.skipped_invalid}）`, "ok");
    if (r.skipped_paths && r.skipped_paths.length) {
      toast(`部分文件被跳过：${r.skipped_paths.slice(0, 3).join("；")}`, "err");
    }
    anLoadList();
  } catch (e) {
    toast("导入失败：" + e.message, "err");
  }
}
async function anUploadImages(files) {
  if (!files.length || !an.ds) return toast("请先选择数据集", "err");
  toast(`正在导入 ${files.length} 张图片…`);
  const all = [...files].filter(f => f.size > 0);
  let added = 0, dup = 0, bad = 0;
  const B = 12;
  for (let i = 0; i < all.length; i += B) {
    const fd = new FormData();
    all.slice(i, i + B).forEach(f => fd.append("files", f, f.name));
    fd.append("split", "train");
    fd.append("max_side", "1920");
    fd.append("dedupe", "true");
    try {
      const r = await api(`/api/datasets/${an.ds}/ingest`, { method: "POST", body: fd });
      added += r.added; dup += r.skipped_duplicate; bad += r.skipped_invalid;
    } catch (e) { toast("批次失败：" + e.message, "err"); }
  }
  await postJSON(`/api/datasets/${an.ds}/finalize`, { split_ratio: 0.8 });
  toast(`已导入 ${added} 张（去重 ${dup}，跳过 ${bad}）`, "ok");
  an.ds && anLoadList();
}
async function aiAssist() {
  if (an.idx < 0) return toast("先打开一张图片", "err");
  const models = await api("/api/models");
  if (!models.length) return toast("模型仓库为空，请先训练一个模型", "err");
  const pick = prompt("用哪个模型预标注？输入序号：\n" + models.map((m, i) => `${i + 1}. ${m.name}`).join("\n"), "1");
  const m = models[parseInt(pick) - 1];
  if (!m) return;
  toast("AI 推理中…");
  try {
    const it = an.items[an.idx];
    const r = await postJSON("/api/predict/dataset-image", {
      model_id: m.id, dataset_id: an.ds, split: it.split, name: it.name, conf: 0.3 });
    pushUndo();
    an.boxes = r.boxes.map(b => ({ shape: "box", ...b }));
    an.dirty = true; an.sel = -1; anDraw();
    toast(`AI 生成 ${r.n} 个框，请人工校正后保存`, "ok");
  } catch (e) { toast(e.message, "err"); }
}
document.addEventListener("keydown", e => {
  if (!document.getElementById("page-annotate").classList.contains("active")) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  const k = e.key.toLowerCase();
  // 多边形绘制中的专用键优先
  if (an.poly) {
    if (e.key === "Enter") { e.preventDefault(); anFinishPoly(); return; }
    if (e.key === "Escape") { e.preventDefault(); anCancelPoly(); toast("已取消多边形"); return; }
    if (k === "z" && (e.ctrlKey || e.metaKey)) {   // 撤销上一个点
      e.preventDefault();
      an.poly.points.pop();
      if (an.poly.points.length < 1) anCancelPoly(); else anDraw();
      return;
    }
  }
  if (k === "a" || e.key === "ArrowLeft") anPrev();
  else if (k === "d" || e.key === "ArrowRight") anNext();
  else if (k === "s") { e.preventDefault(); anSave(); }
  else if (k === "delete") anDeleteBox();
  else if (k === "z" && (e.ctrlKey || e.metaKey)) anUndo();
  else if (k === "b") anSetShape("box");
  else if (k === "c") anSetShape("circle");
  else if (k === "e") anSetShape("ellipse");
  else if (k === "p") anSetShape("polygon");
  else if (e.key === "Escape") { an.sel = -1; anDraw(); }
  else if (/^[1-9]$/.test(k) && an.classes.length >= +k) { an.cls = +k - 1; an.renderClasses(); if (an.sel >= 0) { an.boxes[an.sel].cls = an.cls; if (an.poly) an.poly.cls = an.cls; anDraw(); } }
});

/* ---------------- 数据增强面板 ---------------- */
/* default = ultralytics 官方默认值；min/max/step 控制滑块 */
const AUG_DEFS = [
  { k: "mosaic",      label: "Mosaic 拼接",   min: 0, max: 1, step: 0.05, def: 1.0,  tip: "4 图拼接增强，小目标/遮挡利器" },
  { k: "close_mosaic",label: "末期关 Mosaic", min: 0, max: 50, step: 1,   def: 10,   tip: "最后 N 个 epoch 关闭 mosaic 精调（0=不关）", int: true },
  { k: "mixup",       label: "Mixup 混合",    min: 0, max: 1, step: 0.05, def: 0.0,  tip: "两图 alpha 混合，建议 ≤0.3" },
  { k: "copy_paste",  label: "CopyPaste",     min: 0, max: 1, step: 0.05, def: 0.0,  tip: "实例复制粘贴（分割任务翻倍生效）" },
  { k: "degrees",     label: "旋转角度 °",    min: 0, max: 30, step: 0.5, def: 0.0,  tip: "±角度随机旋转" },
  { k: "translate",   label: "平移",          min: 0, max: 0.5, step: 0.02, def: 0.1, tip: "随机平移比例" },
  { k: "scale",       label: "缩放",          min: 0, max: 0.9, step: 0.05, def: 0.5, tip: "随机缩放幅度" },
  { k: "shear",       label: "剪切 °",        min: 0, max: 15, step: 0.5,  def: 0.0,  tip: "平面剪切变形" },
  { k: "perspective", label: "透视",          min: 0, max: 0.005, step: 0.0005, def: 0.0, tip: "透视畸变（>0.001 易失真）" },
  { k: "flipud",      label: "上下翻转",      min: 0, max: 1, step: 0.05, def: 0.0,  tip: "竖直物体慎开" },
  { k: "fliplr",      label: "左右翻转",      min: 0, max: 1, step: 0.05, def: 0.5,  tip: "水平物体慎开（如文字）" },
  { k: "erasing",     label: "随机擦除",      min: 0, max: 0.9, step: 0.05, def: 0.4, tip: "随机遮挡矩形（仅分类头）" },
  { k: "hsv_h",       label: "色调 HSV-H",    min: 0, max: 0.1, step: 0.005, def: 0.015, tip: "色调抖动" },
  { k: "hsv_s",       label: "饱和度 HSV-S",  min: 0, max: 0.9, step: 0.05, def: 0.7, tip: "饱和度抖动" },
  { k: "hsv_v",       label: "明度 HSV-V",    min: 0, max: 0.9, step: 0.05, def: 0.4, tip: "明度抖动" },
];

const AUG_PRESETS = {
  default: {},
  few:     { mosaic: 1, close_mosaic: 12, mixup: 0.15, copy_paste: 0.15, degrees: 5, scale: 0.5, translate: 0.1, fliplr: 0.5, hsv_s: 0.7, hsv_v: 0.4 },
  weak:    { mosaic: 0.5, close_mosaic: 5, mixup: 0, copy_paste: 0, degrees: 0, scale: 0.3, fliplr: 0.5 },
  off:     { mosaic: 0, close_mosaic: 0, mixup: 0, copy_paste: 0, degrees: 0, translate: 0, scale: 0, shear: 0, perspective: 0, flipud: 0, fliplr: 0, erasing: 0, hsv_h: 0, hsv_s: 0, hsv_v: 0 },
};

function augRender() {
  const grid = $("aug-grid");
  if (!grid || grid.children.length) return;
  grid.innerHTML = AUG_DEFS.map(d => `
    <div class="aug-item" data-k="${d.k}">
      <b>${d.label}</b>
      <input type="number" id="aug-${d.k}" min="${d.min}" max="${d.max}" step="${d.step}" value="${d.def}"
             oninput="augChanged('${d.k}')" onblur="augNormalize('${d.k}')"
             title="${d.tip}｜默认 ${d.def}，范围 ${d.min}~${d.max}">
      <button class="aug-reset" onclick="augResetOne('${d.k}')" title="恢复默认（${d.def}）">${icon("refresh",13)}</button>
    </div>`).join("");
  augSummary();
}

function augValue(d) {
  const el = $("aug-" + d.k);
  if (!el || el.value.trim() === "") return d.def;   // 留空 = 用默认
  let v = d.int ? Math.round(+el.value) : +el.value;
  if (Number.isNaN(v)) return d.def;
  return Math.min(d.max, Math.max(d.min, v));
}

function augChanged(k) {
  const d = AUG_DEFS.find(x => x.k === k);
  const el = $("aug-" + k);
  if (!el) return;
  const item = el.closest(".aug-item");
  if (item) item.classList.toggle("changed", el.value.trim() !== "" && augValue(d) !== d.def);
  augSummary();
}

function augNormalize(k) {
  const d = AUG_DEFS.find(x => x.k === k);
  const el = $("aug-" + k);
  if (!el) return;
  el.value = augValue(d);          // 失焦时钳制到合法范围
  augChanged(k);
}

function augResetOne(k) {
  const d = AUG_DEFS.find(x => x.k === k);
  const el = $("aug-" + k);
  if (!el) return;
  el.value = d.def;
  augChanged(k);
}

function augSummary() {
  const changed = AUG_DEFS.filter(d => {
    const el = $("aug-" + d.k);
    return el && el.value.trim() !== "" && augValue(d) !== d.def;
  });
  const s = $("aug-summary");
  if (s) s.textContent = changed.length
    ? `已修改 ${changed.length} 项：${changed.map(d => d.k + "=" + augValue(d)).join(", ")}`
    : "全部使用官方默认值";
}

function applyPreset(name) {
  const p = AUG_PRESETS[name] || {};
  AUG_DEFS.forEach(d => {
    const el = $("aug-" + d.k);
    if (!el) return;
    el.value = d.k in p ? p[d.k] : d.def;
    augChanged(d.k);
  });
  toast(`已套用预设：${{ default: "官方默认", few: "小样本·强增强", weak: "弱增强", off: "关闭增强" }[name]}`, "ok");
}

function augCollect() {
  const out = {};
  AUG_DEFS.forEach(d => {
    const el = $("aug-" + d.k);
    if (!el || el.value.trim() === "") return;      // 留空不发送
    const v = augValue(d);
    if (v !== d.def) out[d.k] = v;                  // 只覆盖用户改动过的
  });
  return out;
}

/* ---------------- 模型改进 / 损失函数（视频三大方向落地） ---------------- */
const STRAT_DEFS = [
  { k: "multi_scale",   label: "多尺度训练",   min: 0, max: 0.5, step: 0.05, def: 0.0, tip: "每 batch 随机缩放 imgsz±比例，增强尺度鲁棒性（0=关）" },
  { k: "warmup_epochs", label: "Warmup 轮数",  min: 0, max: 10, step: 0.5, def: 3.0, tip: "前期线性升温学习率，稳定注意力等新层收敛" },
  { k: "patience",      label: "早停 patience", min: 0, max: 500, step: 5, def: 100, int: true, tip: "N 轮无提升自动早停（调小省 CPU 时间）" },
  { k: "cos_lr",        label: "余弦退火",     min: 0, max: 1, step: 1, def: 0, int: true, tip: "1=余弦学习率曲线替代线性" },
];

let IMPROVE_OPTS = { arch: [], iou_loss: [] };

async function loadImprovements() {
  IMPROVE_OPTS = await api("/api/train/improvements");
  $("arch-chips").innerHTML = IMPROVE_OPTS.arch.map(o => `
    <label class="chip" title="${o.desc}">
      <input type="checkbox" value="${o.key}" onchange="archChanged()">
      <span>${o.name}</span>
    </label>`).join("");
  $("tr-iou-loss").innerHTML = IMPROVE_OPTS.iou_loss.map(o =>
    `<option value="${o.key}">${o.name}</option>`).join("");
  $("tr-iou-loss").onchange = () => {
    const o = IMPROVE_OPTS.iou_loss.find(x => x.key === $("tr-iou-loss").value);
    $("iou-loss-desc").textContent = o ? o.desc : "";
  };
  $("tr-iou-loss").onchange();
  $("strat-grid").innerHTML = STRAT_DEFS.map(d => `
    <div class="aug-item" data-k="${d.k}">
      <b>${d.label}</b>
      <input type="number" id="strat-${d.k}" min="${d.min}" max="${d.max}" step="${d.step}" value="${d.def}"
             oninput="stratChanged('${d.k}')" onblur="stratNormalize('${d.k}')"
             title="${d.tip}｜默认 ${d.def}，范围 ${d.min}~${d.max}">
    </div>`).join("");
  $("tr-model").addEventListener("change", archChanged);
  archChanged();
}

function stratChanged(k) {
  const d = STRAT_DEFS.find(x => x.k === k);
  const el = $("strat-" + k);
  if (!d || !el) return;
  const v = el.value.trim() === "" ? d.def : parseFloat(el.value);
  el.closest(".aug-item").classList.toggle("changed", v !== d.def);
}
function stratNormalize(k) {
  const d = STRAT_DEFS.find(x => x.k === k);
  const el = $("strat-" + k);
  if (!d || !el) return;
  let v = parseFloat(el.value);
  if (isNaN(v)) v = d.def;
  v = Math.min(d.max, Math.max(d.min, v));
  el.value = d.int ? Math.round(v) : +v.toFixed(4);
  stratChanged(k);
}

function archSelected() {
  return [...document.querySelectorAll("#arch-chips input:checked")].map(cb => cb.value);
}

function archChanged() {
  const picked = archSelected();
  const mode = $("tr-init") ? $("tr-init").value : "builtin";
  const base = $("tr-model") ? $("tr-model").value : "";
  const okBase = /^yolo(11|8)[nsm]$/.test(base);
  const usable = mode === "builtin" && okBase;
  document.querySelectorAll("#arch-chips input").forEach(cb => {
    cb.disabled = !usable;
    cb.closest(".chip").classList.toggle("off", !usable);
  });
  const parts = [];
  if (mode !== "builtin") parts.push("结构改进仅支持内置基础模型（当前来源为仓库/本机权重，勾选将被忽略）");
  else if (!okBase) parts.push(`「${base}」不在结构改进范围（支持 yolo11 / yolov8 的 n/s/m）`);
  if (picked.includes("p2")) parts.push("P2 头显存与每轮耗时约 ×2~3，CPU 训练建议同时降低 imgsz 或 batch");
  if (picked.length && usable) parts.push(`已选 ${picked.join(" + ")}：训练时自动迁移预训练权重，新增层随机初始化`);
  $("arch-summary").textContent = parts.join("　·　");
}

function stratCollect() {
  const out = {};
  STRAT_DEFS.forEach(d => {
    const el = $("strat-" + d.k);
    if (!el || el.value.trim() === "") return;
    const v = parseFloat(el.value);
    if (isNaN(v) || v === d.def) return;
    out[d.k] = d.k === "cos_lr" ? !!v : (d.int ? Math.round(v) : v);
  });
  return out;
}

/* ---------------- 训练 ---------------- */
async function startTraining() {
  const ds = $("tr-ds").value;
  if (!ds) return toast("请选择数据集", "err");

  const mode = $("tr-init").value;
  const body = {
    dataset_id: ds, name: $("tr-name").value || undefined,
    epochs: +$("tr-epochs").value,
    imgsz: +$("tr-imgsz").value, batch: +$("tr-batch").value,
    device: $("tr-device").value, workers: +$("tr-workers").value,
    ...augCollect(), ...stratCollect(),
  };
  if (mode === "builtin") {
    body.model = $("tr-model").value;
    const arch = archSelected();
    if (arch.length) body.arch = arch;
  } else if (mode === "repo") {
    const mid = $("tr-repo-model").value;
    if (!mid) return toast("模型仓库为空，请先训练一个模型或从本机添加", "err");
    body.init_model_id = mid;
  } else {
    const p = $("tr-weights").value.trim();
    if (!p) return toast("请填写本机权重文件路径", "err");
    body.init_weights = p;
  }
  const iou = $("tr-iou-loss").value;
  if (iou) body.iou_loss = iou;
  const lr0 = $("tr-lr0").value; if (lr0) body.lr0 = +lr0;
  try {
    const r = await postJSON("/api/tasks", body);
    if (r.warning) toast(r.warning, "err");
    else toast("训练已启动" + (r.init_weights ? "（已使用自定义基础权重）" : ""), "ok");
    openTaskDetail(r.id);
    loadTaskTable();
  } catch (e) { toast(e.message, "err"); }
}
function taskTableHTML(tasks) {
  return `<table class="tbl"><thead><tr><th>名称</th><th>模型</th><th>状态</th><th>进度</th><th>mAP50</th><th>耗时</th><th>操作</th></tr></thead>
  <tbody>${tasks.map(t => taskRow(t)).join("")}</tbody></table>`;
}
function taskRow(t) {
  const st = { queued: "排队中", running: "训练中", completed: "完成", failed: "失败", stopped: "已停止", interrupted: "已中断" }[t.status] || t.status;
  const dur = t.started_at ? (t.finished_at || Date.now() / 1000) - t.started_at : 0;
  const map50 = t.metrics && t.metrics.mAP50 != null ? (+t.metrics.mAP50).toFixed(3) : "-";
  return `<tr>
    <td><a href="#" onclick="openTaskDetail('${t.id}');return false">${esc(t.name)}</a></td>
    <td>${esc(t.model)}</td>
    <td><span class="badge ${t.status}">${st}</span></td>
    <td><div class="progress"><i style="width:${(t.progress * 100).toFixed(1)}%"></i></div><span class="muted small">${(t.progress * 100).toFixed(0)}%</span></td>
    <td>${map50}</td><td>${fmtDur(dur)}</td>
    <td class="row gap" style="flex-wrap:nowrap">
      ${["queued", "running"].includes(t.status) ? `<button class="btn small" onclick="stopTask('${t.id}')">${icon("stop",14)}停止</button>` : ""}
      ${["failed", "stopped", "interrupted"].includes(t.status) ? `<button class="btn small" onclick="resumeTask('${t.id}')">${icon("play",14)}续训</button>` : ""}
      ${t.status === "completed" ? `<button class="btn small" onclick="scrollToModels()">${icon("layers",14)}模型</button>` : ""}
      <button class="btn small danger" onclick="deleteTask('${t.id}')">${icon("trash",14)}删除</button>
    </td></tr>`;
}
function scrollToModels() {
  loadModels();
  $("model-table").scrollIntoView({ behavior: "smooth", block: "center" });
}
let taskTimer = null;
async function loadTaskTable() {
  const tasks = await api("/api/tasks");
  const tb = $("task-table").querySelector("tbody");
  tb.innerHTML = tasks.map(t => taskRow(t)).join("") || '<tr><td colspan="7" class="muted">暂无任务</td></tr>';
  const active = tasks.some(t => ["queued", "running"].includes(t.status));
  if (active && !taskTimer) taskTimer = setInterval(refreshTasks, 2500);
  if (!active && taskTimer) { clearInterval(taskTimer); taskTimer = null; }
}
async function refreshTasks() {
  await loadTaskTable();
  if (currentTask) pollDetail();
}
async function stopTask(id) { await postJSON(`/api/tasks/${id}/stop`); toast("已发送停止"); loadTaskTable(); }
async function resumeTask(id) {
  try { const r = await postJSON(`/api/tasks/${id}/resume`); toast("续训任务已启动", "ok"); openTaskDetail(r.id); }
  catch (e) { toast(e.message, "err"); }
}
async function deleteTask(id) {
  if (!confirm("删除任务及其训练产物？")) return;
  await del(`/api/tasks/${id}`); toast("已删除");
  if (currentTask === id) { currentTask = null; $("task-detail").style.display = "none"; }
  loadTaskTable();
}

/* ---- 任务详情：曲线 / 日志 ---- */
let currentTask = null, logOffset = 0, detailTimer = null, metricsCache = [];
async function openTaskDetail(id) {
  currentTask = id; logOffset = 0;
  $("task-detail").style.display = "block";
  const t = await api(`/api/tasks/${id}`);
  $("td-title").textContent = `· ${t.name}（${t.model}）` +
    ((t.args && t.args.arch && t.args.arch.length) ? ` [${t.args.arch.join("+")}]` : "") +
    ((t.args && t.args.iou_loss) ? ` [${t.args.iou_loss.toUpperCase()}]` : "");
  metricsCache = await api(`/api/tasks/${id}/metrics`);
  switchChart(currentChart);
  if (detailTimer) clearInterval(detailTimer);
  detailTimer = setInterval(pollDetail, 2500);
  $("task-detail").scrollIntoView({ behavior: "smooth" });
}
async function pollDetail() {
  if (!currentTask) return;
  try {
    metricsCache = await api(`/api/tasks/${currentTask}/metrics`);
    if (currentChart === "loss") drawLoss();
    if (currentChart === "map") drawMap();
    if (currentChart === "log") await pullLog();
    if (currentChart === "artifacts") await pullArtifacts();
  } catch {}
}
let currentChart = "loss";
function switchChart(which) {
  currentChart = which;
  document.querySelectorAll("[data-chart]").forEach(b => b.classList.toggle("active", b.dataset.chart === which));
  $("chart-area").style.display = ["loss", "map"].includes(which) ? "" : "none";
  $("pr-area").style.display = which === "pr" ? "" : "none";
  $("artifacts-area").style.display = which === "artifacts" ? "" : "none";
  $("log-area").style.display = which === "log" ? "" : "none";
  if (which === "loss") drawLoss();
  if (which === "map") drawMap();
  if (which === "pr") drawPR();
  if (which === "artifacts") pullArtifacts();
  if (which === "log") pullLog(true);
}
async function pullLog(reset = false) {
  if (!currentTask) return;
  if (reset) { logOffset = 0; $("log-area").textContent = ""; }
  const r = await api(`/api/tasks/${currentTask}/log?offset=${logOffset}`);
  if (r.text) {
    const el = $("log-area");
    el.textContent += r.text;
    el.scrollTop = el.scrollHeight;
    logOffset = r.offset;
    if (el.textContent.length > 400000) el.textContent = el.textContent.slice(-200000);
  }
}
async function pullArtifacts() {
  if (!currentTask) return;
  const list = await api(`/api/tasks/${currentTask}/artifacts`);
  $("artifacts-area").innerHTML = list.length ? list.map(a =>
    `<img src="${a.url}" alt="${esc(a.name)}" title="${esc(a.name)}" onclick="window.open('${a.url}')">`).join("")
    : '<p class="muted">训练结束（或每轮验证后）会生成 results.png、混淆矩阵等图表。</p>';
}
/* 简易 canvas 折线图 */
function drawChart(canvas, series, opts) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, P = { l: 52, r: 16, t: 26, b: 30 };
  ctx.clearRect(0, 0, W, H);
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  series.forEach(s => s.points.forEach(([x, y]) => {
    if (!isFinite(y)) return;
    xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
    ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
  }));
  if (!isFinite(xmin)) { ctx.fillStyle = "#667"; ctx.font = "14px sans-serif"; ctx.fillText(opts.empty || "暂无数据，等待训练产出…", P.l, H / 2); return; }
  if (opts.zeroY) ymin = Math.min(ymin, 0);
  const pad = (ymax - ymin) * 0.08 || 0.01; ymax += pad; ymin = opts.zeroY ? 0 : ymin - pad;
  const X = x => P.l + (x - xmin) / ((xmax - xmin) || 1) * (W - P.l - P.r);
  const Y = y => H - P.b - (y - ymin) / ((ymax - ymin) || 1) * (H - P.t - P.b);
  ctx.strokeStyle = "#26324b"; ctx.fillStyle = "#7c8aa5"; ctx.font = "11px sans-serif"; ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const yv = ymin + (ymax - ymin) * i / 5, yp = Y(yv);
    ctx.beginPath(); ctx.moveTo(P.l, yp); ctx.lineTo(W - P.r, yp); ctx.stroke();
    ctx.fillText(yv.toFixed(3), 8, yp + 4);
  }
  const xt = Math.max(1, Math.round((xmax - xmin) / 8));
  for (let x = Math.ceil(xmin); x <= xmax; x += xt) ctx.fillText(x, X(x) - 6, H - 10);
  series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    s.points.forEach(([x, y]) => { if (!isFinite(y)) return; started ? ctx.lineTo(X(x), Y(y)) : (ctx.moveTo(X(x), Y(y)), started = true); });
    ctx.stroke();
    ctx.fillStyle = s.color;
    s.points.forEach(([x, y]) => { if (isFinite(y)) { ctx.beginPath(); ctx.arc(X(x), Y(y), 2.4, 0, 7); ctx.fill(); } });
  });
  // 图例
  let lx = P.l;
  ctx.font = "12px sans-serif";
  series.forEach(s => {
    ctx.fillStyle = s.color; ctx.fillRect(lx, 8, 14, 4);
    ctx.fillStyle = "#cdd7ea"; ctx.fillText(s.label, lx + 20, 14);
    lx += 26 + ctx.measureText(s.label).width;
  });
}
function drawLoss() {
  const get = (m, keys) => { for (const k of keys) if (m && m[k] != null) return m[k]; return null; };
  const trainBox = metricsCache.map(r => [r.epoch, get(r.loss, ["train/box_loss"]) ?? get(r.metrics, ["train/box_loss"]) ?? NaN]);
  const trainCls = metricsCache.map(r => [r.epoch, get(r.loss, ["train/cls_loss"]) ?? NaN]);
  const valBox = metricsCache.map(r => [r.epoch, get(r.metrics, ["val/box_loss"]) ?? NaN]);
  const valCls = metricsCache.map(r => [r.epoch, get(r.metrics, ["val/cls_loss"]) ?? NaN]);
  drawChart($("metric-canvas"), [
    { label: "train box", color: "#4c8dff", points: trainBox },
    { label: "train cls", color: "#5ad2f4", points: trainCls },
    { label: "val box", color: "#35c98e", points: valBox },
    { label: "val cls", color: "#ffb454", points: valCls },
  ], { empty: "等待第一个 epoch 完成…" });
}
function drawMap() {
  const g = k => metricsCache.map(r => [r.epoch, r.metrics?.[k] ?? NaN]);
  drawChart($("metric-canvas"), [
    { label: "mAP50", color: "#4c8dff", points: g("metrics/mAP50(B)") },
    { label: "mAP50-95", color: "#35c98e", points: g("metrics/mAP50-95(B)") },
    { label: "precision", color: "#ffb454", points: g("metrics/precision(B)") },
    { label: "recall", color: "#ff8fab", points: g("metrics/recall(B)") },
  ], { zeroY: true, empty: "等待验证指标…" });
}
async function drawPR() {
  /* PR 曲线：从 artifacts 里的 PNG 直接展示（ultralytics 自动绘制） */
  const list = await api(`/api/tasks/${currentTask}/artifacts`);
  const pr = list.filter(a => /P_curve|R_curve|results\.png/i.test(a.name));
  const cv = $("pr-canvas"), ctx = cv.getContext("2d");
  ctx.fillStyle = "#10151f"; ctx.fillRect(0, 0, cv.width, cv.height);
  if (!pr.length) { ctx.fillStyle = "#667"; ctx.font = "14px sans-serif"; ctx.fillText("训练完成后可查看 PR 曲线（当前显示 results.png 替代）", 40, 190); return; }
  const img = new Image();
  img.onload = () => {
    cv.width = 1000; cv.height = Math.round(1000 * img.height / img.width);
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
  };
  img.src = pr[0].url;
}

/* ---------------- 系统原生文件选择框 ---------------- */
let nativeDialogAvailable = null;

async function checkNativeDialog() {
  try {
    const r = await api("/api/native-dialog/available");
    nativeDialogAvailable = r.available;
    const hint = $("ml-native-hint");
    if (hint) {
      hint.textContent = r.available
        ? "点击按钮会弹出 Windows 文件管理器窗口（选择后自动回填路径）"
        : `当前环境无法弹系统对话框（${r.reason}），请用「网页内浏览目录」或手填路径`;
      if (!r.available) hint.style.color = "var(--warn)";
    }
    return r.available;
  } catch (e) {
    nativeDialogAvailable = false;
    return false;
  }
}

/**
 * 弹出系统文件管理器选文件，把结果写回指定 input。
 * 浏览器拿不到真实路径，所以由本地服务端调起原生对话框。
 */
async function pickNativeFile(purpose = "weights", inputId = "ml-path", multiple = false) {
  if (nativeDialogAvailable === null) await checkNativeDialog();
  if (nativeDialogAvailable === false) {
    toast("当前环境无法弹出系统文件对话框，请改用网页浏览或手填路径", "err");
    window.__lastNativePaths = [];
    return "";
  }

  toast("已调起系统文件管理器，请在弹出的窗口中选择…");
  try {
    const initDir = (inputId && $(inputId) && $(inputId).value) ? null : undefined;
    const r = await postJSON("/api/native-dialog/pick", {
      kind: "file", purpose, multiple, initialdir: initDir,
    });
    window.__lastNativePaths = r.paths || [];
    if (r.cancelled) { toast("已取消选择"); return ""; }
    if (!r.ok) { toast(r.error || "选择失败", "err"); return ""; }
    const paths = r.paths || [];
    if (!paths.length) return "";
    const p = paths[0];
    if (inputId && $(inputId)) $(inputId).value = p;
    toast(multiple && paths.length > 1 ? `已选择 ${paths.length} 个文件` : `已选择：${p}`, "ok");
    if (purpose === "weights" && inputId === "ml-path") mlInspect();
    return p;
  } catch (e) {
    toast("调用文件对话框失败：" + e.message, "err");
    window.__lastNativePaths = [];
    return "";
  }
}

/* ---------------- 模型仓库 ---------------- */
let modelsCache = [];

const SOURCE_LABEL = { trained: "训练产出", uploaded: "本机导入", pretrained: "预训练", demo: "演示" };
const TASK_LABEL = { detect: "检测", segment: "分割", classify: "分类", pose: "姿态", obb: "旋转框" };

async function loadModels() {
  const [models, choices] = await Promise.all([api("/api/models"), api("/api/models/choices")]);
  modelsCache = models;

  $("pretrained-list").innerHTML = choices.pretrained.map(p => `
    <div class="card" style="min-width:180px">
      <div>${p.id}</div>
      <div class="muted small">${p.downloaded ? "已下载 " + fmtSize(p.size) : "未下载"}</div>
      ${p.downloaded ? `<span class="ok-dot">${icon("check",12)}</span>` : `<button class="btn small" onclick="dlPretrained('${p.id}')">${icon("download",14)}下载</button>`}
    </div>`).join("");

  renderModels();
  fillModelSelect($("pd-model"));
  fillRepoSelect();
}

function renderModels() {
  const q = ($("md-q").value || "").toLowerCase();
  const src = $("md-filter").value;
  const list = modelsCache.filter(m =>
    (!src || m.source === src) &&
    (!q || m.name.toLowerCase().includes(q) ||
      (m.classes || []).some(c => String(c).toLowerCase().includes(q))));

  $("model-table").querySelector("tbody").innerHTML = list.map(m => `
    <tr>
      <td>
        <div><b>${esc(m.name)}</b></div>
        <div class="muted small">${m.exists ? "" : '<span style="color:var(--danger)">${icon("alert",13)} 文件已丢失</span>'}${m.note ? esc(m.note).slice(0, 60) : ""}</div>
      </td>
      <td><span class="badge queued">${TASK_LABEL[m.task] || m.task || "检测"}</span></td>
      <td>${m.classes.length ? m.classes.slice(0, 4).map(c => esc(c)).join(", ") + (m.classes.length > 4 ? ` …(共${m.classes.length})` : "") : '<span class="muted">未记录</span>'}</td>
      <td>${m.metrics && m.metrics["mAP50-95"] != null ? (+m.metrics["mAP50-95"]).toFixed(3) : "-"}</td>
      <td>${fmtSize(m.size)}</td>
      <td><span class="muted small">${SOURCE_LABEL[m.source] || m.source}</span></td>
      <td class="row gap" style="flex-wrap:nowrap">
        <button class="btn small primary" onclick="useForPredict('${m.id}')">${icon("search",14)}验证</button>
        <button class="btn small" onclick="trainFromModel('${m.id}')" title="以此为初始权重继续训练">${icon("trending",14)}再训练</button>
        <a class="btn small" href="/api/models/${m.id}/download">${icon("download",14)}下载</a>
        <button class="btn small" onclick="openExport('${m.id}')">${icon("archive",14)}导出</button>
        <button class="btn small danger" onclick="deleteModel('${m.id}')">${icon("trash",14)}删除</button>
      </td>
    </tr>`).join("") ||
    `<tr><td colspan="7" class="muted">${modelsCache.length ? "没有匹配的模型" : "仓库为空：训练完成会自动入库，也可从本机添加"}</td></tr>`;
}

async function dlPretrained(id) {
  toast(`正在下载 ${id}.pt …`);
  try { await postJSON("/api/models/pretrained", { model: id }); toast("下载完成", "ok"); loadModels(); }
  catch (e) { toast(e.message, "err"); }
}
async function deleteModel(id) {
  const m = modelsCache.find(x => x.id === id);
  if (!confirm(`删除模型「${m ? m.name : id}」？${m && m.in_repo ? "\n（权重文件也会被删除）" : "\n（仅移除登记，原文件保留）"}`)) return;
  await del(`/api/models/${id}`); toast("已删除"); loadModels();
}

/* ---- 从本机添加模型 ---- */
async function mlInspect() {
  const path = $("ml-path").value.trim();
  if (!path) return toast("请先填文件路径，或点「浏览本机」", "err");
  $("ml-result").innerHTML = '<span class="spin"></span> 正在解析权重…';
  try {
    const r = await postJSON("/api/models/inspect", { path });
    const cls = (r.classes || []).slice(0, 8).join(", ");
    $("ml-result").innerHTML = `
      <div class="inspect-box">
        <div><b>${esc(r.name)}</b> <span class="muted">${esc(r.ext)} · ${fmtSize(r.size)}</span></div>
        <div>任务：<b>${TASK_LABEL[r.task] || r.task || "-"}</b> · 类别数：<b>${(r.classes || []).length}</b>${r.params ? ` · 参数：<b>${(r.params / 1e6).toFixed(2)}M</b>` : ""}${r.epochs ? ` · 训练轮次：${r.epochs}` : ""}</div>
        ${cls ? `<div>类别：${esc(cls)}${(r.classes || []).length > 8 ? " …" : ""}</div>` : '<div class="muted">未在权重中读到类别信息（可在添加后手动补）</div>'}
        ${r.error ? `<div style="color:var(--danger)">${esc(r.error)}</div>` : ""}
        <div class="muted">${esc(r.path)}</div>
      </div>`;
  } catch (e) {
    $("ml-result").innerHTML = `<span style="color:var(--danger)">${icon("x-circle",14)} ${esc(e.message)}</span>`;
  }
}

async function mlImport() {
  const path = $("ml-path").value.trim();
  if (!path) return toast("请先填权重文件路径", "err");
  const mode = document.querySelector('input[name="ml-mode"]:checked').value;
  $("ml-result").innerHTML = '<span class="spin"></span> 正在添加到仓库…';
  try {
    const r = await postJSON("/api/models/import-local", { path, mode });
    $("ml-result").innerHTML = `<span style="color:var(--accent2)">${icon("check-circle",14)} 已添加到仓库：${esc(r.path)}（${fmtSize(r.size)}，${(r.classes || []).length} 个类别）</span>`;
    toast("模型已加入仓库", "ok");
    loadModels();
  } catch (e) {
    $("ml-result").innerHTML = `<span style="color:var(--danger)">${icon("x-circle",14)} ${esc(e.message)}</span>`;
  }
}

let mlBrowserCwd = "";
async function mlBrowse(path = "") {
  try {
    const r = await api(`/api/models/browse?path=${encodeURIComponent(path)}`);
    mlBrowserCwd = r.cwd;
    const rows = [];
    if (r.parent) rows.push(`<div class="br-item dir" onclick="mlBrowse('${r.parent.replace(/\\/g, "\\\\")}')">${icon("arrow-up",14)} ..</div>`);
    if (r.home && r.cwd !== r.home) rows.push(`<div class="br-item dir" onclick="mlBrowse('${r.home.replace(/\\/g, "\\\\")}')">${icon("home",14)} 主目录</div>`);
    rows.push(...r.items.map(it => it.type === "dir"
      ? `<div class="br-item dir" onclick="mlBrowse('${it.path.replace(/\\/g, "\\\\")}')">${icon("folder",14)} ${esc(it.name)}</div>`
      : `<div class="br-item file" onclick="mlChoose('${it.path.replace(/\\/g, "\\\\")}')">${icon("layers",14)} ${esc(it.name)} <span class="muted small">${fmtSize(it.size)}</span></div>`));
    $("ml-result").innerHTML = `
      <div class="browser">
        <div class="browser-path">${esc(r.cwd)}</div>
        <div class="browser-list">${rows.join("") || '<div class="muted small" style="padding:8px">此目录下没有权重文件</div>'}</div>
      </div>`;
  } catch (e) { toast("浏览失败：" + e.message, "err"); }
}
function mlChoose(p) {
  $("ml-path").value = p;
  $("ml-result").innerHTML = "";
  mlInspect();
}

/* ---- 训练页：从模型仓库初始化 ---- */
function trInitChange() {
  const mode = $("tr-init").value;
  $("tr-model-wrap").style.display = mode === "builtin" ? "" : "none";
  $("tr-repo-wrap").style.display = mode === "repo" ? "" : "none";
  $("tr-path-wrap").style.display = mode === "lexical" ? "" : "none";
  if (mode === "repo") fillRepoSelect();
  const hint = {
    builtin: "CPU 参考：yolo11n · 100 图 · imgsz 480 ≈ 每 epoch 10~30 秒",
    repo: "以仓库里已有模型为起点继续训练（适合增量数据/继续收敛）",
    lexical: "填本机 .pt 绝对路径作为初始权重",
  }[mode];
  $("tr-hint").textContent = hint;
  if (typeof archChanged === "function") archChanged();
}
async function fillRepoSelect() {
  const sel = $("tr-repo-model");
  const cur = sel.value;
  const ms = await api("/api/models");
  sel.innerHTML = ms.map(m => `<option value="${m.id}">${esc(m.name)}${m.metrics && m.metrics["mAP50-95"] != null ? ` (mAP ${(+m.metrics["mAP50-95"]).toFixed(3)})` : ""}</option>`).join("")
    || '<option value="">（仓库为空）</option>';
  if (cur && [...sel.options].some(o => o.value === cur)) sel.value = cur;
}
function trainFromModel(id) {
  $("tr-init").value = "repo";
  trInitChange();
  $("tr-repo-model").value = id;
  $("tr-init").scrollIntoView({ behavior: "smooth", block: "center" });
  toast("已设为初始权重，填好参数后点「开始训练」", "ok");
}

const dlgExport = $("dlg-export");
let exportModelId = null;
async function openExport(id) {
  exportModelId = id;
  const c = await api("/api/models/choices");
  $("ex-format").innerHTML = c.exports.map(f => `<option value="${f}">${f}</option>`).join("");
  $("ex-result").textContent = "";
  dlgExport.showModal();
}
$("ex-go").addEventListener("click", async () => {
  $("ex-result").innerHTML = '<span class="spin"></span> 导出中（onnx 首次需要几秒～几十秒）…';
  try {
    const r = await postJSON(`/api/models/${exportModelId}/export`, { format: $("ex-format").value, imgsz: +$("ex-imgsz").value });
    $("ex-result").innerHTML = `${icon("check-circle",14)} 已导出 ${esc(r.name)}（${fmtSize(r.size)}）
      <a class="btn small" href="/api/models/${exportModelId}/export-file">下载</a>`;
  } catch (e) { $("ex-result").innerHTML = icon("x-circle",14) + " " + esc(e.message); }
});
function useForPredict(id) {
  document.querySelector('[data-tab="predict"]').click();
  setTimeout(() => { $("pd-model").value = id; pdOnModelChange(); }, 300);
}
function openPredictFor() { document.querySelector('[data-tab="predict"]').click(); }

async function fillModelSelect(sel) {
  const models = await api("/api/models");
  const usable = models.filter(m => m.exists);
  sel.innerHTML = usable.map(m =>
    `<option value="${m.id}">${esc(m.name)}${m.classes.length ? ` · ${m.classes.length}类` : ""}</option>`).join("")
    || '<option value="">（仓库为空）</option>';
  pdOnModelChange();
}

/* ---------------- 屏幕采集插件 ---------------- */
let capTimer = null, capJobs = [];

async function capInit() {
  capModeChange();
  await fillDatasetSelect($("cap-ds"));
  // 默认时间：5 分钟后
  const d = new Date(Date.now() + 5 * 60000);
  d.setSeconds(0, 0);
  $("cap-at-once").value = toLocalInput(d);
  try {
    const c = await api("/api/capture/capability");
    if (!c.available) {
      $("cap-warn").innerHTML = icon("alert",14) + " 当前环境无法截屏：" + esc(c.reason);
      $("cap-warn-panel").style.display = "";
    } else if (c.screen[0]) {
      $("cap-warn-panel").style.display = "none";
      $("cap-status").textContent = `屏幕 ${c.screen[0]}×${c.screen[1]}`;
    }
  } catch (e) {
    $("cap-warn").innerHTML = icon("alert",14) + " 采集能力检查失败：" + esc(e.message);
    $("cap-warn-panel").style.display = "";
  }
  await capLoadJobs();
  if (!capTimer) capTimer = setInterval(capLoadJobs, 3000);
}

function toLocalInput(d) {
  const p = n => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}

function capModeChange() {
  const once = $("cap-mode").value === "once";
  $("cap-at-once-wrap").style.display = once ? "" : "none";
  $("cap-at-daily-wrap").style.display = once ? "none" : "";
}

async function capCreate(runNow = false) {
  const ds = $("cap-ds").value;
  if (!ds) return toast("请先准备一个数据集（可在数据集页创建）", "err");
  const mode = $("cap-mode").value;
  const at = mode === "once" ? $("cap-at-once").value : $("cap-at-daily").value;
  if (!at) return toast("请填写启动时间", "err");
  const body = {
    dataset_id: ds, name: $("cap-name").value || undefined, mode, at,
    duration_min: +$("cap-dur").value, interval_sec: +$("cap-interval").value,
    active_window: +$("cap-window").value, move_threshold: +$("cap-th").value,
    max_images: +$("cap-max").value,
  };
  try {
    const j = await postJSON("/api/capture/jobs", body);
    if (runNow) await postJSON(`/api/capture/jobs/${j.id}/run-now`);
    toast(runNow ? "任务已创建并立即开始采集" : "排程任务已创建", "ok");
    capLoadJobs();
  } catch (e) { toast(e.message, "err"); }
}

async function capTest() {
  $("cap-status").innerHTML = '<span class="spin"></span> 试截中…';
  try {
    const r = await postJSON("/api/capture/test", {});
    if (r.duplicate) {
      $("cap-status").textContent = `试截成功：画面与已有图片重复（去重生效，未新增）`;
      toast("截屏正常，画面重复被去重", "ok");
    } else {
      $("cap-status").textContent = `试截成功：已存入测试数据集`;
      toast("截屏成功，已存入「屏幕采集-测试」", "ok");
    }
    if (r.image_token) {
      $("cap-recent").innerHTML = `<img src="/api/predict/image/${r.image_token}" style="max-width:100%;border-radius:8px">
        <div class="muted small">${icon("arrow-up",12)} 刚截到的画面</div>`;
    }
  } catch (e) {
    $("cap-status").textContent = "";
    toast("试截失败：" + e.message, "err");
  }
}

const CAP_STATUS = { scheduled: "已排程", running: "采集中", paused: "已暂停",
                     done: "已完成", failed: "失败", interrupted: "已中断" };

async function capLoadJobs() {
  try {
    capJobs = await api("/api/capture/jobs");
    const tb = $("cap-table").querySelector("tbody");
    tb.innerHTML = capJobs.map(j => {
      const ds = j.dataset_id;
      const sched = j.mode === "daily" ? `每天 ${j.at_time}` : fmtTime(new Date(j.at_time).getTime() / 1000);
      let next = "-";
      if (j.status === "scheduled" && j.secs_to_run != null) next = fmtDur(j.secs_to_run) + " 后";
      else if (j.status === "running") next = "运行中";
      return `<tr>
        <td>${esc(j.name)}</td>
        <td class="muted small">${esc(ds)}</td>
        <td>${esc(sched)} · ${j.duration_min}分钟 · 每${j.interval_sec}s</td>
        <td><span class="badge ${j.status === "running" ? "running" : j.status === "scheduled" ? "queued" : j.status === "done" ? "completed" : j.status === "failed" || j.status === "interrupted" ? "failed" : "stopped"}">${CAP_STATUS[j.status] || j.status}</span>
          ${j.last_error ? `<div class="muted small" style="color:var(--danger)">${esc(j.last_error)}</div>` : ""}</td>
        <td>${j.captured || 0} / ${j.skipped_dup || 0} / ${j.skipped_idle || 0}</td>
        <td>${next}${j.last_capture ? `<div class="muted small">上次 ${fmtTime(j.last_capture)}</div>` : ""}</td>
        <td class="row gap" style="flex-wrap:nowrap">
          ${j.status === "running" ? `<button class="btn small" onclick="capStop('${j.id}')">${icon("stop",14)}停止</button>` : ""}
          ${j.status !== "running" ? `<button class="btn small primary" onclick="capRunNow('${j.id}')">${icon("camera",14)}立即采</button>` : ""}
          ${j.status !== "running" ? `<button class="btn small" onclick="capResume('${j.id}')" title="按排程时间等待">${icon("clock",14)}排程</button>` : ""}
          <button class="btn small" onclick="capViewDS('${j.dataset_id}')">${icon("pen",14)}去标注</button>
          <button class="btn small danger" onclick="capDelete('${j.id}')">${icon("trash",14)}删除</button>
        </td></tr>`;
    }).join("") || '<tr><td colspan="7" class="muted">暂无采集任务</td></tr>';

    const pick = $("cap-pick");
    const cur = pick.value;
    pick.innerHTML = capJobs.map(j => `<option value="${j.id}">${esc(j.name)}（${CAP_STATUS[j.status] || j.status}）</option>`).join("") || '<option value="">（无任务）</option>';
    if (cur && [...pick.options].some(o => o.value === cur)) pick.value = cur;
  } catch (e) { /* 服务未就绪时静默 */ }
}

async function capRunNow(id) {
  try { await postJSON(`/api/capture/jobs/${id}/run-now`); toast("已开始采集", "ok"); capLoadJobs(); }
  catch (e) { toast(e.message, "err"); }
}
async function capStop(id) { await postJSON(`/api/capture/jobs/${id}/stop`); toast("已停止"); capLoadJobs(); }
async function capResume(id) { await postJSON(`/api/capture/jobs/${id}/resume`); toast("已按排程等待"); capLoadJobs(); }
async function capDelete(id) {
  if (!confirm("删除采集任务？已采集的图片会保留在数据集里。")) return;
  await del(`/api/capture/jobs/${id}`); toast("已删除"); capLoadJobs();
}
function capViewDS(dsId) { gotoAnnotate(dsId); }

async function capRecent() {
  const id = $("cap-pick").value;
  if (!id) { $("cap-recent").innerHTML = ""; return; }
  const list = await api(`/api/capture/jobs/${id}/recent`);
  const job = capJobs.find(j => j.id === id);
  $("cap-recent").innerHTML = list.length ? list.map(c => `
    <div class="batch-cell">
      <img loading="lazy" src="/api/datasets/${job.dataset_id}/thumb/${c.split}/${encodeURIComponent(c.name)}">
      <div class="batch-meta"><span class="muted small">${fmtTime(c.mtime)}</span></div>
    </div>`).join("") : '<p class="muted small">该任务还没有采集到图片（或图片名不带 cap_ 前缀）</p>';
}

/* ---------------- 推理验证（统一走模型仓库） ---------------- */
function pdOnModelChange() {
  const m = modelsCache.find(x => x.id === $("pd-model").value)
    || null;
  if (!m) { $("pd-model-meta").textContent = ""; return; }
  const parts = [];
  if (m.task) parts.push(TASK_LABEL[m.task] || m.task);
  if (m.classes.length) parts.push(`${m.classes.length} 类：${m.classes.slice(0, 5).join("/")}${m.classes.length > 5 ? "…" : ""}`);
  if (m.metrics && m.metrics["mAP50-95"] != null) parts.push(`mAP50-95 ${(+m.metrics["mAP50-95"]).toFixed(3)}`);
  parts.push(SOURCE_LABEL[m.source] || m.source);
  $("pd-model-meta").textContent = parts.join(" · ");
}

async function pdRun(file) {
  if (!file) return;
  const model = $("pd-model").value;
  if (!model) return toast("模型仓库为空，先去训练或添加模型", "err");
  $("pd-result").innerHTML = '<span class="spin"></span> 推理中…';
  $("pd-list").innerHTML = ""; $("pd-count").textContent = "";
  $("pd-batch-panel").style.display = "none";
  const fd = new FormData();
  fd.append("model_id", model); fd.append("file", file);
  fd.append("conf", $("pd-conf").value);
  fd.append("imgsz", $("pd-imgsz").value);
  try {
    const r = await api("/api/predict", { method: "POST", body: fd });
    renderPredictResult(r);
  } catch (e) {
    $("pd-result").innerHTML = `<span style="color:var(--danger)">${esc(e.message)}</span>`;
  }
}

/* 从本机磁盘（系统对话框）选图片推理 */
async function pdPickFromDisk() {
  const model = $("pd-model").value;
  if (!model) return toast("请先选择模型", "err");
  const p = await pickNativeFile("images", null, false);
  if (!p) return;
  $("pd-result").innerHTML = '<span class="spin"></span> 推理中…';
  $("pd-batch-panel").style.display = "none";
  try {
    const r = await postJSON("/api/predict/by-path", {
      model_id: model, path: p,
      conf: +$("pd-conf").value, imgsz: +$("pd-imgsz").value });
    renderPredictResult(r);
  } catch (e) {
    $("pd-result").innerHTML = `<span style="color:var(--danger)">${esc(e.message)}</span>`;
  }
}

function renderPredictResult(r) {
  const d = r.result;
  $("pd-result").innerHTML = (r.image_token
    ? `<img src="/api/predict/image/${r.image_token}">` : "<p class='muted'>无检测目标，未生成标注图</p>")
    + `<p class="muted small">${d.width}×${d.height} · 推理耗时 ${d.speed} ms · 检出 ${d.detections.length} 个目标</p>`
    + (d.source ? `<p class="muted small">文件：${esc(d.source)}</p>` : "");
  $("pd-count").textContent = `${d.detections.length} 个`;
  $("pd-list").innerHTML = d.detections.map(x =>
    `<div class="det-row"><span><span class="swatch" style="background:${PALETTE[x.cls % PALETTE.length]}"></span>${esc(x.cls_name)}</span><b>${(+x.conf).toFixed(2)}</b></div>`).join("")
    || '<p class="muted">无检测结果</p>';
}

/* 用数据集里的图片验证（快速抽样） */
async function pdPickDatasetImage() {
  const model = $("pd-model").value;
  if (!model) return toast("请先选择模型", "err");
  const dsList = await api("/api/datasets");
  if (!dsList.length) return toast("还没有数据集", "err");
  const pick = prompt("选择数据集（输入序号）：\n" +
    dsList.map((d, i) => `${i + 1}. ${d.name} (${d.n_images}图)`).join("\n"), "1");
  const ds = dsList[parseInt(pick) - 1];
  if (!ds) return;
  const imgs = await api(`/api/datasets/${ds.id}/images?limit=500`);
  if (!imgs.items.length) return toast("该数据集没有图片", "err");
  const idx = prompt(`选择第几张（1~${imgs.items.length}）：`, "1");
  const it = imgs.items[parseInt(idx) - 1];
  if (!it) return;
  $("pd-result").innerHTML = '<span class="spin"></span> 推理中…';
  try {
    const r = await postJSON("/api/predict/dataset-image", {
      model_id: model, dataset_id: ds.id, split: it.split, name: it.name, conf: +$("pd-conf").value });
    $("pd-result").innerHTML = `<img src="/api/datasets/${ds.id}/image/${it.split}/${encodeURIComponent(it.name)}">
      <p class="muted small">数据集图片 ${esc(it.name)} · 检出 ${r.n} 个目标 · ${r.speed_ms} ms（未叠加框，可对照原标注）</p>`;
    $("pd-count").textContent = `${r.n} 个`;
    $("pd-list").innerHTML = r.boxes.map(b => {
      const name = (modelsCache.find(m => m.id === model)?.classes[b.cls]) ?? b.cls;
      return `<div class="det-row"><span><span class="swatch" style="background:${PALETTE[b.cls % PALETTE.length]}"></span>${esc(name)}</span><b>${b.conf ? (+b.conf).toFixed(2) : "-"}</b></div>`;
    }).join("") || '<p class="muted">无检测结果</p>';
  } catch (e) { $("pd-result").innerHTML = `<span style="color:var(--danger)">${esc(e.message)}</span>`; }
}

/* 批量验证：在整个数据集上跑一遍，统计检出情况 */
async function pdTestAll() {
  const model = $("pd-model").value;
  if (!model) return toast("请先选择模型", "err");
  const dsList = await api("/api/datasets");
  if (!dsList.length) return toast("还没有数据集", "err");
  const pick = prompt("在哪个数据集上批量验证？输入序号：\n" +
    dsList.map((d, i) => `${i + 1}. ${d.name} (${d.n_images}图)`).join("\n"), "1");
  const ds = dsList[parseInt(pick) - 1];
  if (!ds) return;
  const n = parseInt(prompt("验证前多少张？（最多 40 张，CPU 推理较慢）", "20") || "20");
  const limit = Math.max(1, Math.min(40, n));

  $("pd-batch-panel").style.display = "";
  $("pd-batch-sub").textContent = `模型「${modelsCache.find(m => m.id === model)?.name || ""}」 · ${ds.name} · 前 ${limit} 张`;
  $("pd-batch-body").innerHTML = '<span class="spin"></span> 批量推理中…';

  const imgs = await api(`/api/datasets/${ds.id}/images?limit=${limit}`);
  const cells = [];
  batchCache = [];
  let totalDet = 0, emptyCount = 0, ms = 0;
  for (const it of imgs.items.slice(0, limit)) {
    try {
      const r = await postJSON("/api/predict/dataset-image", {
        model_id: model, dataset_id: ds.id, split: it.split, name: it.name, conf: +$("pd-conf").value });
      totalDet += r.n; ms += r.speed_ms;
      if (r.n === 0) emptyCount++;
      const idx = batchCache.length;
      batchCache.push({ ds: ds.id, dsName: ds.name, model, split: it.split, name: it.name,
                        stem: it.stem, boxes: r.boxes, n: r.n, speed: r.speed_ms, gtN: it.n_boxes });
      const thumb = `/api/datasets/${ds.id}/thumb/${it.split}/${encodeURIComponent(it.name)}`;
      cells.push(`<div class="batch-cell ${r.n ? "" : "empty"}" onclick="showBatchDetail(${idx})" title="点击查看大图：检出与原标注对照">
        <img src="${thumb}">
        <div class="batch-meta"><b>${r.n}</b> 个目标${it.n_boxes ? ` / 标注 ${it.n_boxes}` : ""}<br><span class="muted small">${esc(it.name)}</span></div>
      </div>`);
      $("pd-batch-body").innerHTML = cells.join("");
    } catch (e) {
      cells.push(`<div class="batch-cell err"><div class="batch-meta" style="color:var(--danger)">${esc(e.message)}</div></div>`);
    }
  }
  $("pd-batch-sub").textContent =
    `模型「${modelsCache.find(m => m.id === model)?.name || ""}」 · ${ds.name} · ${cells.length} 张 · 共检出 ${totalDet} 个目标 · 无检出 ${emptyCount} 张 · 平均 ${(ms / Math.max(1, cells.length)).toFixed(0)} ms/张 · 点击缩略图看详情`;
}

/* ---------------- 批量验证详情（点击单元格弹出大图对照） ---------------- */
let batchCache = [];
const bd = { idx: -1, img: null };
let bdSeq = 0;
const SHAPE_CN = { box: "矩形", circle: "圆", ellipse: "椭圆", polygon: "多边形" };

async function showBatchDetail(i) {
  const c = batchCache[i];
  if (!c) return;
  bd.idx = i;
  const seq = ++bdSeq;
  const dlg = $("dlg-bdetail");
  if (!dlg.open) dlg.showModal();
  $("bd-title").textContent = c.name;
  $("bd-nav-info").textContent = `${i + 1} / ${batchCache.length}`;
  $("bd-body").innerHTML = '<p class="muted"><span class="spin"></span> 加载详情…</p>';
  if (c.gt === undefined) {
    try {
      const [det, lab] = await Promise.all([
        api(`/api/datasets/${c.ds}`),
        api(`/api/datasets/${c.ds}/label/${c.split}/${encodeURIComponent(c.stem)}`),
      ]);
      c.gtClasses = det.classes || [];
      c.gt = lab.boxes || [];
    } catch (e) { c.gt = []; c.gtClasses = []; }
    if (seq !== bdSeq) return; // 已切到别的图，丢弃过期结果
  }
  bdRender(c);
}

function bdNav(d) {
  const i = bd.idx + d;
  if (i < 0 || i >= batchCache.length) return;
  showBatchDetail(i);
}

function bdRender(c) {
  const mclasses = modelsCache.find(m => m.id === c.model)?.classes || [];
  const predRows = c.boxes.map(b =>
    `<div class="det-row"><span><span class="swatch" style="background:${PALETTE[(b.cls || 0) % PALETTE.length]}"></span>${esc(mclasses[b.cls] ?? b.cls)}</span><b>${(+b.conf).toFixed(2)}</b></div>`).join("")
    || '<p class="muted small">无检出</p>';
  const gtRows = c.gt.map(b =>
    `<div class="det-row"><span><span class="swatch" style="background:${PALETTE[(b.cls || 0) % PALETTE.length]}"></span>${esc(c.gtClasses[b.cls] ?? b.cls)}</span><b class="muted small">${SHAPE_CN[b.shape] || "矩形"}</b></div>`).join("")
    || '<p class="muted small">该图没有原标注</p>';
  $("bd-body").innerHTML = `
    <div class="bd-canvas-wrap"><canvas id="bd-canvas"></canvas></div>
    <div class="bd-side">
      <h4>模型检出 · ${c.n} 个 <span class="muted small">${(c.speed || 0).toFixed(0)} ms</span></h4>
      <div>${predRows}</div>
      <h4 style="margin-top:12px">原标注 · ${c.gt.length} 个</h4>
      <div>${gtRows}</div>
    </div>`;
  $("bd-legend").innerHTML =
    `<span class="lg lg-solid"></span>模型检出　<span class="lg lg-dash"></span>原标注　<span class="muted small">类别色一致时，虚线=标注、实线=预测，可直观对照漏检/误检</span>`;
  const img = new Image();
  const seq2 = bdSeq;
  img.onload = () => { if (seq2 !== bdSeq) return; bd.img = img; bdDraw(); };
  img.onerror = () => { if (seq2 !== bdSeq) return; $("bd-body").innerHTML = '<p style="color:var(--danger)">图片加载失败</p>'; };
  img.src = `/api/datasets/${c.ds}/image/${c.split}/${encodeURIComponent(c.name)}`;
}

function bdAnchor(g) {
  if (g.kind === "box") return [g.x, g.y];
  if (g.kind === "circle" || g.kind === "ellipse") return [g.cx - g.rx, g.cy - g.ry];
  return [Math.min(...g.pts.map(q => q[0])), Math.min(...g.pts.map(q => q[1]))];
}

function bdStrokeShape(ctx, b, cv, color, lw) {
  const g = shapeGeom(b, cv);
  ctx.lineWidth = lw; ctx.strokeStyle = color;
  if (g.kind === "box") {
    ctx.strokeRect(g.x, g.y, g.w, g.h);
  } else if (g.kind === "circle" || g.kind === "ellipse") {
    ctx.beginPath(); ctx.ellipse(g.cx, g.cy, Math.max(1, g.rx), Math.max(1, g.ry), 0, 0, Math.PI * 2); ctx.stroke();
  } else {
    ctx.beginPath();
    g.pts.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.closePath(); ctx.stroke();
  }
  return g;
}

function bdLabel(ctx, text, x, y, color, s, solid) {
  ctx.font = `${Math.round(12 * s)}px sans-serif`;
  const th = Math.round(17 * s);
  const tw = ctx.measureText(text).width + 8 * s;
  const ly = Math.max(0, y - th);
  if (solid) {
    ctx.fillStyle = color; ctx.fillRect(x, ly, tw, th);
    ctx.fillStyle = "#08101e";
  } else {
    ctx.fillStyle = "rgba(8,14,26,.72)"; ctx.fillRect(x, ly, tw, th);
    ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.strokeRect(x + .5, ly + .5, tw - 1, th - 1);
    ctx.fillStyle = "#d7e3ff";
  }
  ctx.fillText(text, x + 4 * s, ly + Math.round(12.5 * s));
}

function bdDraw() {
  const c = batchCache[bd.idx];
  const cv = $("bd-canvas");
  if (!c || !bd.img || !cv) return;
  cv.width = bd.img.naturalWidth; cv.height = bd.img.naturalHeight;
  const ctx = cv.getContext("2d");
  ctx.drawImage(bd.img, 0, 0, cv.width, cv.height);
  const s = Math.max(1, Math.min(3, cv.width / 640));
  const mclasses = modelsCache.find(m => m.id === c.model)?.classes || [];
  // 原标注：虚线 + 描边标签
  ctx.setLineDash([6 * s, 4 * s]);
  c.gt.forEach(b => {
    const color = PALETTE[(b.cls || 0) % PALETTE.length];
    const [ax, ay] = bdAnchor(bdStrokeShape(ctx, b, cv, color, 1.6 * s));
    bdLabel(ctx, `标注 ${c.gtClasses[b.cls] ?? b.cls}`, ax, ay, color, s, false);
  });
  // 模型检出：实线 + 实底标签
  ctx.setLineDash([]);
  c.boxes.forEach(b => {
    const color = PALETTE[(b.cls || 0) % PALETTE.length];
    const g = shapeGeom({ shape: "box", cx: b.cx, cy: b.cy, w: b.w, h: b.h }, cv);
    ctx.lineWidth = 2.2 * s; ctx.strokeStyle = color;
    ctx.strokeRect(g.x, g.y, g.w, g.h);
    bdLabel(ctx, `检出 ${mclasses[b.cls] ?? b.cls} ${(+b.conf).toFixed(2)}`, g.x, g.y, color, s, true);
  });
}

/* 详情弹窗内 ←/→ 键翻页 */
document.getElementById("dlg-bdetail").addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") { e.preventDefault(); bdNav(-1); }
  if (e.key === "ArrowRight") { e.preventDefault(); bdNav(1); }
});

/* ---------------- 模型选择框填充 ---------------- */
async function fillModelChoices() {
  const c = await api("/api/models/choices");
  $("tr-model").innerHTML = c.models.map(m => `<option value="${m.id}">${m.label}</option>`).join("");
}

/* ---------------- 启动 ---------------- */
bindDropzone(); bindCanvas(); wizBind(); bindCanvasDrop();
fillModelChoices();
loadImprovements().catch(() => {});
loadDashboard();
setInterval(() => {
  if (document.getElementById("page-dashboard").classList.contains("active")) loadDashboard();
}, 8000);
