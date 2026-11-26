import { PortraitSpinePlayer, effectiveCanvasMotionRatio } from '/player/player-widget.js';

const stageChecks = {
  files_complete: '文件完整', loop_seam: '8 秒无缝循环',
  original_coverage: '原图完整覆盖', rigid_no_stretch: '刚体无变形',
};
const core = new Set(['torso', 'body', 'pelvis', 'head', 'face']);
const lower = new Set(['leg', 'upper_leg', 'lower_leg', 'foot']);
const props = new Set(['weapon', 'prop']);
const eyes = [
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12Z"/><circle cx="12" cy="12" r="2.7"/></svg>',
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m3 3 18 18M10 6.2a11 11 0 0 1 12 5.8 14 14 0 0 1-3.1 3.5M6.4 7.8A16 16 0 0 0 2 12s3.5 6 10 6a10 10 0 0 0 4.3-.9"/></svg>',
];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function description(row) {
  if (row.motion_type === 'static') return '站立固定';
  if (!row.motion_adjustable) return '跟随姿势与支撑约束';
  if (['torso', 'body', 'pelvis'].includes(row.role)) return row.label + '呼吸幅度';
  if (row.motion_type === 'soft') return row.label + '飘动幅度';
  return row.label + '摆动幅度';
}

export function createCanvasWorkbench({ request, onJob, onStatus }) {
  const get = name => document.getElementById(name);
  const tabs = ['Motion', 'Order', 'Review'];
  const stage = get('canvasLiveStage');
  let player, observer, revision, job, controller, initial;
  let rows = [], scales = new Map(), baked = new Map(), overall = 1, bakedOverall = 1;
  let tab = 'Motion', busy = false, ready = false, lowerOpen = false;
  let reference = '', editingMessage = '';

  const serialize = () => JSON.stringify([rows.map(row => [row.name, row.enabled]), [...scales], overall]);
  const isDirty = () => serialize() !== reference;
  const isLocked = () => busy || !ready || job?.status === 'running' || Boolean(job?.locked);
  const requiresActivation = () => [...scales].some(([driver, value]) => baked.get(driver) === 0 && effectiveCanvasMotionRatio(value,overall) > 0);
  const message = text => { editingMessage = text; get('canvasWorkbenchMessage').textContent = text; };

  function frameState() {
    if (!player) return;
    player.setLiveSlotOrder(rows.map(row => row.name).reverse());
    player.setLiveHiddenSlots(new Set(rows.filter(row => !row.enabled).map(row => row.name)));
    player.setLiveMotionScales(new Map([...scales].map(([id,value]) => [id,effectiveCanvasMotionRatio(value, overall)])), baked);
  }

  function drawActionState() {
    const dirty = isDirty();
    const locked = isLocked();
    get('btnCanvasTweaks').disabled = locked || !dirty;
    get('btnCanvasTweaks').classList.toggle('cw-dirty', dirty && !locked);
    get('btnCanvasSignoff').disabled = locked;
    get('canvasReviewExport').disabled = locked || !get('canvasWatched').checked;
    get('canvasGlobal').disabled = locked;
    get('canvasReset').disabled = locked || !initial;
    for (const control of get('canvasControlScroll').querySelectorAll('[data-local-control]')) {
      control.disabled = locked || control.dataset.inactive === 'true';
    }
    get('canvasLiveBadge').textContent = job?.locked ? '已锁定 🔒' : busy || job?.status === 'running' ? '保存烘焙中' : ready ? '实时预览' : '载入中';
    if (job?.locked && !editingMessage) get('canvasWorkbenchMessage').textContent = '该角色已确认锁定防误改；如需继续调参请先点上方「解锁」。';
    onStatus({ busy: busy || job?.status === 'running', ready });
  }

  function selectTab(name, focus = false) {
    tab = name;
    for (const id of tabs) {
      const selected = id === name;
      const button = get('canvas' + id + 'Tab');
      button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
      get('canvas' + id + 'Panel').hidden = !selected;
      if (selected && focus) button.focus();
    }
    get('canvasControlScroll').scrollTop = 0;
  }

  function highlight(row) {
    if (player?.highlightSlot !== (row?.name ?? null)) player?.setHighlightSlot(row?.name ?? null);
    const label = get('canvasFocusLabel');
    label.textContent = row?.label || '';
    label.hidden = !row;
  }

  function availabilityChanged() {
    get('canvasWatched').checked = false;
    frameState();
    message(isDirty() ? '实时预览已更新 · 尚未保存' : '当前与已保存版本一致');
    drawActionState();
  }

  function updateEyes() {
    for (const button of get('canvasControlScroll').querySelectorAll('[data-eye]')) {
      const row = rows.find(item => item.name === button.dataset.eye);
      const action = (row.enabled ? '隐藏' : '显示') + row.label;
      button.innerHTML = eyes[row.enabled ? 0 : 1];
      button.setAttribute('aria-label', action); button.title = action;
      button.setAttribute('aria-pressed', String(row.enabled));
    }
  }

  function updateRanges() {
    for (const range of get('canvasLayerRows').querySelectorAll('[data-driver]')) {
      const value = Math.round(100 * (scales.get(range.dataset.driver) ?? 1));
      range.value = String(value);
      range.nextElementSibling.textContent = value ? value + '%' : '静止';
      range.setAttribute('aria-valuetext', value ? value + '%' : '0%，静止');
    }
    const value = Math.round(overall * 100);
    get('canvasGlobal').value = String(value);
    get('canvasGlobalValue').textContent = value + '%';
    get('canvasZeroNotice').hidden = bakedOverall > 0;
  }

  function makeRow(row, mode, index = 0) {
    const part = element('article', 'cw-part');
    part.dataset.slot = row.name;
    const thumb = element('img', 'cw-thumb'); thumb.src = row.thumb_url; thumb.alt = ''; thumb.loading = 'lazy';
    const copy = element('div');
    copy.append(element('strong', '', row.label));
    if (mode === 'motion') {
      copy.append(element('small', '', description(row)));
      if (row.driver_module !== row.module) copy.append(element('small', '', '与持握部位联动'));
    }
    const actions = element('div', 'cw-move-buttons');
    const eye = element('button', 'cw-icon'); eye.type = 'button'; eye.dataset.eye = row.name; eye.dataset.localControl = '';
    eye.onclick = () => { row.enabled = !row.enabled; updateEyes(); availabilityChanged(); };
    actions.append(eye);
    if (mode === 'order') {
      for (const [step, name, arrow] of [[-1, '置前', '↑'], [1, '置后', '↓']]) {
        const move = element('button', 'cw-icon', arrow); move.type = 'button'; move.dataset.localControl = '';
        move.dataset.inactive = String(index + step < 0 || index + step >= rows.length);
        move.setAttribute('aria-label', name + '：' + row.label); move.title = name;
        move.onclick = () => {
          const destination = index + step;
          if (destination < 0 || destination >= rows.length) return;
          [rows[index], rows[destination]] = [rows[destination], rows[index]];
          drawOrder(); updateEyes(); availabilityChanged();
          Array.from(get('canvasOrderRows').querySelectorAll('button')).find(button => button.title === name
            && button.closest('[data-slot]')?.dataset.slot === row.name && !button.disabled)?.focus();
        };
        actions.append(move);
      }
    }
    part.append(thumb, copy, actions);
    if (mode === 'motion' && row.motion_adjustable) {
      const input = element('input'); input.type = 'range'; input.min = '0'; input.max = '150'; input.step = '1';
      input.dataset.driver = row.driver_module; input.dataset.localControl = '';
      input.setAttribute('aria-label', description(row));
      input.oninput = () => { scales.set(row.driver_module, Number(input.value) / 100); highlight(row); updateRanges(); availabilityChanged(); };
      const line = element('div', 'cw-range'); line.append(input, element('output')); part.append(line);
      if (row.baked_motion_ratio === 0) {
        const note = element('small', 'cw-range', '当前为静止态，调高后点一次「保存实时微调」即可激活。');
        part.append(note);
      }
    }
    part.addEventListener('pointerenter', () => highlight(row));
    part.addEventListener('pointerleave', () => { if (!part.contains(document.activeElement)) highlight(null); });
    part.addEventListener('focusin', () => highlight(row));
    part.addEventListener('focusout', event => { if (!part.contains(event.relatedTarget)) highlight(null); });
    return part;
  }

  function drawMotion() {
    const container = get('canvasLayerRows'); container.replaceChildren();
    const groups = [
      ['核心动态', rows.filter(row => core.has(row.role)).sort((a, b) => Number(a.role !== 'torso') - Number(b.role !== 'torso'))],
      ['武器与道具', rows.filter(row => props.has(row.role)).sort((a, b) => Number(a.role !== 'weapon') - Number(b.role !== 'weapon'))],
      ['头发 / 围巾 / 衣摆飘动', rows.filter(row => row.motion_type === 'soft' && !core.has(row.role) && !props.has(row.role) && !lower.has(row.role))],
      ['四肢与其他部位', rows.filter(row => row.motion_type !== 'soft' && !core.has(row.role) && !props.has(row.role) && !lower.has(row.role))],
    ];
    for (const [title, parts] of groups) {
      if (!parts.length) continue;
      const section = element('section', 'cw-group'); section.append(element('h3', '', title));
      section.append(...parts.map(row => makeRow(row, 'motion'))); container.append(section);
    }
    const legs = rows.filter(row => lower.has(row.role));
    if (legs.length) {
      const details = element('details', 'cw-lower'); details.open = lowerOpen;
      details.ontoggle = () => { lowerOpen = details.open; };
      details.append(element('summary', '', '下肢与脚部支撑'), ...legs.map(row => makeRow(row, 'motion')));
      container.append(details);
    }
  }

  function drawOrder() { get('canvasOrderRows').replaceChildren(...rows.map((row, index) => makeRow(row, 'order', index))); }

  function drawChecks() {
    for (const listId of ['canvasHardChecks', 'canvasReviewChecks']) {
      get(listId).replaceChildren(...Object.entries(stageChecks).map(([key, text]) => {
        const pass = job.canvas_checks?.[key];
        return element('li', pass ? '' : 'cw-check-fail', (pass ? '✓ ' : '！') + text);
      }));
    }
    const warnings = [...(job.canvas_diagnostics?.cleanup?.warnings ?? []), ...(job.canvas_diagnostics?.exposure_stress?.warnings ?? [])];
    const pressure = job.canvas_diagnostics?.exposure_stress;
    const result = get('canvasStressResult');
    if (pressure?.levels?.['1.5x']) {
      const [base,high] = ['1.0x','1.5x'].map(key=>pressure.levels[key].peak_exposure_px);
      result.textContent = '全身露底压力：100% '+base+' px · 150% '+high+' px'+(pressure.status==='pass'?' · 通过':' · 待修');
    } else { result.textContent = '当前版本未生成压力体检，请重新保存预览。'; }
    const labels = new Map(rows.map(row => [row.name, row.label]));
    const banner = get('canvasLayerWarningStatus'); banner.hidden = !warnings.length;
    banner.textContent = warnings.length ? warnings.length + ' 条分层提醒 · 建议查看接缝特写' : '';
    get('canvasLayerWarnings').replaceChildren(...warnings.map(warning => {
      const name = labels.get(warning.layer) ?? '局部';
      const text = warning.code === 'max_slider_exposure'
        ? Math.round(warning.gain*100)+'% 动作压力下发现露底，最大单处 '+warning.largest_region_px+' px'
        : warning.code === 'local_layer_pixel_bleed'
        ? '贴图串层：' + warning.layers.map(layer => labels.get(layer) ?? '相邻部位').join('与')
        : warning.code === 'separated_soft_islands' ? name + '有分离的布片，请核对遮挡与摆动' : name + '需检查贴图边缘';
      return element('li', '', text);
    }));
    const links = { canvasSeamLink: job.canvas_seam_preview, canvasLayerSheet: job.canvas_all_layers_url, canvasOpenStandalone: job.canvas_preview_html,
      canvasOpenRig: job.rig_url, canvasOpenDiagnostics: job.canvas_diagnostics_url };
    for (const [id, url] of Object.entries(links)) {
      get(id).hidden = !url;
      if (url) get(id).href = url;
    }
    get('canvasSeamButton').hidden = !job.canvas_seam_preview;
    if (job.canvas_seam_preview && get('canvasSeamImage').getAttribute('src') !== job.canvas_seam_preview) {
      get('canvasSeamImage').src = job.canvas_seam_preview;
    }
    const downloads = [];
    for (const bundle of Object.values(job.exports || {})) {
      for (const file of bundle.files || []) {
        const title = file.filename.endsWith('.zip') ? '下载正式包' : file.filename.endsWith('.html') ? '下载单文件 HTML'
          : file.filename.includes('light') ? '下载浅底视频' : '下载深底视频';
        const link = element('a', 'cw-button', title); link.href = file.url; link.download = file.filename;
        downloads.push(link);
      }
    }
    get('canvasDownloads').replaceChildren(...downloads);
  }

  function getPlayer() {
    if (!player) {
      player = new PortraitSpinePlayer({canvas: stage, webglSurface: stage.parentElement,
        onTimeUpdate(time, duration) {
          const output = get('canvasTimeText');
          const text = time.toFixed(1) + ' / ' + duration.toFixed(1) + ' 秒';
          if (output.textContent === text) return;
          get('canvasTime').max = duration; get('canvasTime').value = time;
          output.textContent = text;
        },
        onStateChange(state) {
          get('canvasPlay').textContent = state.playing ? '暂停' : '播放';
          get('canvasBones').checked = state.showBones;
          stage.dataset.bones = state.showBones ? 'on' : 'off';
          stage.dataset.backend = state.backend;
        },
      });
      observer = new ResizeObserver(() => { if (!get('canvasWorkbench').hidden) player.resize(); });
      observer.observe(stage);
    }
    return player;
  }

  async function render(next) {
    const nextRevision = next.job_id + ':' + next.rig_version;
    const sameJob = job?.job_id === next.job_id;
    job = next;
    get('canvasWorkbench').hidden = false;
    get('canvasJobTitle').textContent = next.character + ' · 待机预览';
    if (revision === nextRevision) { drawChecks(); drawActionState(); return; }
    revision = nextRevision;
    if (!sameJob) { initial = null; lowerOpen = false; selectTab('Motion'); }
    const current = getPlayer();
    const previousTime = sameJob ? current.currentTime : 0;
    const autoplay = !sameJob ? !matchMedia('(prefers-reduced-motion: reduce)').matches : current.playing;
    current.pause();
    ready = false;
    if (!sameJob) current.setShowBones(false);
    rows = next.canvas_layers.map(row => ({...row})).sort((a,b) => b.drawOrder - a.drawOrder);
    baked = new Map(rows.filter(row => row.motion_adjustable).map(row => [row.driver_module, row.baked_motion_ratio ?? row.baked_motion_scale]));
    scales = new Map(rows.filter(row => row.motion_adjustable).map(row => [row.driver_module, row.motion_ratio ?? row.baked_motion_scale]));
    bakedOverall = next.canvas_global_motion_scale ?? 1; overall = bakedOverall;
    reference = serialize();
    if (!initial) initial = { rows: rows.map(row => ({...row})), scales: new Map(scales), overall };
    get('canvasWatched').checked = false;
    highlight(null); drawMotion(); drawOrder(); updateEyes(); updateRanges(); drawActionState(); drawChecks();
    controller?.abort();
    const pending = new AbortController(); controller = pending;
    try {
      const url = next.canvas_live_rig_url ?? next.rig_url;
      await current.loadFromUrl(url, { signal:pending.signal, healthReportUrl:null, autoPlay:false,
        live:{slotOrder:rows.map(row => row.name).reverse(), hiddenSlots:new Set(rows.filter(row=>!row.enabled).map(row=>row.name))} });
      if (pending.signal.aborted) return;
      current.setTime(previousTime);
      if (autoplay) current.play();
      ready = true;
      if (!busy) message('拖动滑块即可看到变化，不必等待保存。');
    } catch (error) {
      if (pending.signal.aborted) return;
      message('预览载入失败：' + error.message);
    } finally { if (!pending.signal.aborted) drawActionState(); }
  }

  async function post(action, data) {
    const next = await request('/api/jobs/' + encodeURIComponent(job.job_id) + '/' + action,
      {method:'POST',body:JSON.stringify({...data, expected_rig_version:job.rig_version})});
    await onJob(next);
  }

  async function save(exportAfter = false) {
    if (isLocked()) return;
    if (exportAfter && !get('canvasWatched').checked) {
      selectTab('Review'); get('canvasWatched').focus(); message('请先观看完整循环并勾选验收。'); return;
    }
    const activate = requiresActivation();
    busy = true; drawActionState();
    try {
      if (isDirty()) {
        message('保存烘焙中，仍可观看左侧预览。');
        const layers = Object.fromEntries(rows.map((row, index) => [row.name, {drawOrder:rows.length - 1 - index, enabled:row.enabled}]));
        const modules = Object.fromEntries([...scales].map(([driver, value]) => [driver, {motion_ratio:value}]));
        await post('layer-tweaks', {layers, modules, global_motion_scale:overall});
      }
      if (exportAfter && activate) { selectTab('Review'); message('原本静止的部位已激活，请重新看完并确认这版。'); return; }
      if (exportAfter) {
        message('正在生成正式包…');
        await post('user-signoff', {decision:'pass', reviewer:'Studio 使用者'});
        selectTab('Review'); message('已验收，可在检查页下载。');
      } else { message('已保存。请核对本版的实际动作。'); }
    } catch (error) { message(error.message); }
    finally { busy = false; drawActionState(); }
  }

  for (const id of tabs) {
    const button = get('canvas' + id + 'Tab');
    button.onclick = () => selectTab(id);
    button.onkeydown = event => {
      const index = tabs.indexOf(tab);
      const next = {ArrowRight:(index+1)%tabs.length, ArrowLeft:(index+tabs.length-1)%tabs.length,
        Home:0, End:tabs.length-1}[event.key];
      if (next !== undefined) { event.preventDefault(); selectTab(tabs[next], true); }
    };
  }
  get('canvasPlay').onclick = () => player?.togglePlay();
  get('canvasRestart').onclick = () => player?.setTime(0);
  get('canvasTime').oninput = event => player?.setTime(event.target.value);
  get('canvasSpeed').onchange = event => player?.setSpeed(event.target.value);
  get('canvasBackground').onchange = event => player?.setBackground(event.target.value);
  get('canvasBones').onchange = event => player?.setShowBones(event.target.checked);
  get('canvasGlobal').oninput = event => { overall = Number(event.target.value) / 100; updateRanges(); availabilityChanged(); };
  get('canvasReset').onclick = () => {
    if (!initial) return;
    const latest = new Map(rows.map(row => [row.name,row]));
    rows = initial.rows.map(row => ({...latest.get(row.name), enabled:row.enabled}));
    scales = new Map(initial.scales); overall = initial.overall;
    drawMotion(); drawOrder(); updateEyes(); updateRanges(); availabilityChanged();
  };
  get('btnCanvasTweaks').onclick = () => save();
  get('btnCanvasSignoff').onclick = get('canvasReviewExport').onclick = () => save(true);
  get('canvasWatched').onchange = drawActionState;
  get('canvasSeamButton').onclick = () => {
    selectTab('Review'); get('canvasSeamLink').focus({preventScroll:true});
    get('canvasSeamLink').scrollIntoView({block:'nearest'});
  };

  return {
    render,
    hide() {
      if (!job) return;
      controller?.abort(); player?.dispose(); observer?.disconnect(); player = null; observer = null;
      job = null; revision = null; ready = false;
      get('canvasWorkbench').hidden = true;
    },
  };
}
