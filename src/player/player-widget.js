/**
 * Reusable zero-dependency Spine 4.2 Portrait Player UI controller.
 * Coordinates Spine42Skeleton + SpineCanvasRenderer + playback transport + health report display.
 */

import { Spine42Skeleton } from './spine42-runtime.js';
import { SpineCanvasRenderer } from './canvas-renderer.js';

export function effectiveCanvasMotionRatio(part, overall = 1) {
  return Math.min(1.5, Math.max(0, part) * Math.max(0, overall));
}

export const HARD_GATE_SPECS = [
  { key: 'rest_diff', label: 'Rest Diff', desc: 'Setup pose vs parts composite max diff' },
  { key: 'setup_identity', label: 'Setup Identity', desc: 'Frame 0 / 8.00s loop boundary match' },
  { key: 'no_holes', label: 'No Holes', desc: 'No interior alpha tears or seam gaps' },
  { key: 'anchor_stability', label: 'Anchor Stability', desc: 'Grounded foot / anchor lock stability' },
];

export const SOFT_HINT_SPECS = [
  { key: 'head', label: 'Head', unit: 'px' },
  { key: 'chest', label: 'Chest', unit: 'px' },
  { key: 'hips', label: 'Hips', unit: 'px' },
  { key: 'hair_tip', label: 'Hair Tip', unit: 'px' },
];

/**
 * Normalize various health-report.json shapes into a consistent display structure.
 *
 * @param {object|null|undefined} raw
 */
export function normalizeHealthReport(raw) {
  if (!raw || typeof raw !== 'object') {
    return {
      loaded: false,
      overallStatus: 'na',
      gates: HARD_GATE_SPECS.map((g) => ({
        ...g,
        status: 'na',
        valueText: '--',
        detail: '',
      })),
      hints: SOFT_HINT_SPECS.map((h) => ({
        ...h,
        status: 'na',
        valueText: '--',
        detail: '',
      })),
    };
  }

  const gateSources = [
    raw.hard_gates,
    raw.hardGates,
    raw.gates,
    raw.checks,
    raw,
  ].filter((x) => x && typeof x === 'object');

  const hintSources = [
    raw.softHints?.landmarks,
    raw.soft_hints?.landmarks,
    raw.soft_hints,
    raw.softHints,
    raw.displacement_hints,
    raw.displacementHints,
    raw.displacements,
    raw.hints,
    raw.motion,
    raw,
  ].filter((x) => x && typeof x === 'object');

  const pick = (sources, key) => {
    for (const src of sources) {
      if (src[key] !== undefined && src[key] !== null) return src[key];
    }
    return undefined;
  };

  const parseStatus = (entry) => {
    if (entry === undefined || entry === null) return 'na';
    if (typeof entry === 'boolean') return entry ? 'pass' : 'fail';
    if (typeof entry === 'string') {
      const low = entry.toLowerCase();
      if (low === 'pass' || low === 'passed' || low === 'ok' || low === 'green') return 'pass';
      if (low === 'warn' || low === 'warning' || low === 'warn_low' || low === 'warn_high' || low === 'yellow') return 'warn';
      if (low === 'fail' || low === 'failed' || low === 'error' || low === 'red') return 'fail';
      if (low === 'not_present' || low === 'na' || low === 'n/a') return 'na';
      return 'na';
    }
    if (typeof entry === 'object') {
      if (typeof entry.pass === 'boolean') return entry.pass ? 'pass' : 'fail';
      if (typeof entry.passed === 'boolean') return entry.passed ? 'pass' : 'fail';
      if (typeof entry.ok === 'boolean') return entry.ok ? 'pass' : 'fail';
      if (typeof entry.status === 'string') return parseStatus(entry.status);
    }
    return 'na';
  };

  const fmtNum = (n) => (Number.isInteger(n) ? String(n) : n.toFixed(2));

  const extractGateMetrics = (key, entry) => {
    if (!entry || typeof entry !== 'object') return null;
    const m = entry.measured && typeof entry.measured === 'object' ? entry.measured : null;
    const th = entry.threshold && typeof entry.threshold === 'object' ? entry.threshold : null;
    if (!m) return null;

    if (key === 'rest_diff' && typeof m.maxRGBA === 'number') {
      const lim = th && typeof th.maxRGBA === 'number' ? th.maxRGBA : null;
      return {
        value: m.maxRGBA,
        valueText: lim !== null ? `${fmtNum(m.maxRGBA)} / <= ${fmtNum(lim)} RGBA` : `${fmtNum(m.maxRGBA)} RGBA`,
      };
    }
    if (key === 'setup_identity' && typeof m.setupVsPartsMaxRGBA === 'number') {
      const f0 = typeof m.frame0VsSetup === 'number' ? m.frame0VsSetup : 0;
      const fend = typeof m.frameLoopEndVsSetup === 'number' ? m.frameLoopEndVsSetup : 0;
      const parts = m.setupVsPartsMaxRGBA;
      const partsLim = th && typeof th.setupVsPartsMaxRGBA === 'number' ? th.setupVsPartsMaxRGBA : null;
      return {
        value: Math.max(f0, fend, parts),
        valueText:
          partsLim !== null
            ? `f0:${fmtNum(f0)} end:${fmtNum(fend)} parts:${fmtNum(parts)} / <= ${fmtNum(partsLim)}`
            : `f0:${fmtNum(f0)} end:${fmtNum(fend)} parts:${fmtNum(parts)}`,
      };
    }
    if (key === 'no_holes' && typeof m.enclosedHolesMax === 'number') {
      const lim = th && typeof th.enclosedHolesMax === 'number' ? th.enclosedHolesMax : null;
      return {
        value: m.enclosedHolesMax,
        valueText: lim !== null ? `${fmtNum(m.enclosedHolesMax)} / <= ${fmtNum(lim)}` : fmtNum(m.enclosedHolesMax),
      };
    }
    if (key === 'anchor_stability' && typeof m.maxDriftPx === 'number') {
      const lim = th && typeof th.maxDriftPx === 'number' ? th.maxDriftPx : null;
      return {
        value: m.maxDriftPx,
        valueText: lim !== null ? `${fmtNum(m.maxDriftPx)} px / <= ${fmtNum(lim)} px` : `${fmtNum(m.maxDriftPx)} px`,
      };
    }
    return null;
  };

  const formatVal = (entry, defaultUnit = '') => {
    if (entry === undefined || entry === null) return { value: null, valueText: '--' };
    if (typeof entry === 'number') {
      return {
        value: entry,
        valueText: `${fmtNum(entry)}${defaultUnit ? ' ' + defaultUnit : ''}`,
      };
    }
    if (typeof entry === 'boolean') return { value: null, valueText: entry ? 'PASS' : 'FAIL' };
    if (typeof entry === 'string') return { value: null, valueText: entry };
    if (typeof entry === 'object') {
      const v =
        entry.ptpPx ??
        entry.value ??
        entry.max ??
        entry.max_px ??
        entry.px ??
        entry.amplitude ??
        entry.drift ??
        entry.diff ??
        entry.count;
      const unit = entry.unit ?? defaultUnit;
      const threshold = entry.threshold ?? entry.limit ?? entry.max_allowed;
      if (typeof v === 'number') {
        const vStr = `${fmtNum(v)}${unit ? ' ' + unit : ''}`;
        if (typeof threshold === 'number') {
          return {
            value: v,
            valueText: `${vStr} / <= ${threshold}${unit ? ' ' + unit : ''}`,
          };
        }
        return { value: v, valueText: vStr };
      }
      if (typeof v === 'string') return { value: null, valueText: v };
      if (typeof entry.summary === 'string') return { value: null, valueText: entry.summary };
    }
    return { value: null, valueText: '--' };
  };

  const gates = HARD_GATE_SPECS.map((spec) => {
    const entry = pick(gateSources, spec.key);
    const status = parseStatus(entry);
    const gateMetric = extractGateMetrics(spec.key, entry) ?? formatVal(entry);
    const detail =
      entry && typeof entry === 'object' && typeof entry.message === 'string'
        ? entry.message
        : spec.desc;
    return {
      ...spec,
      status,
      value: gateMetric.value,
      valueText: gateMetric.valueText,
      measured: entry && typeof entry === 'object' && entry.measured ? entry.measured : null,
      threshold: entry && typeof entry === 'object' && entry.threshold !== undefined ? entry.threshold : null,
      detail,
    };
  });

  const hints = SOFT_HINT_SPECS.map((spec) => {
    const entry = pick(hintSources, spec.key);
    const rawStatusStr =
      entry && typeof entry === 'object' && typeof entry.status === 'string'
        ? entry.status.toLowerCase()
        : null;
    let status = parseStatus(entry);
    if (status === 'na' && entry !== undefined && rawStatusStr !== 'not_present' && rawStatusStr !== 'na' && rawStatusStr !== 'n/a') {
      status = 'info';
    }
    const { value, valueText } = formatVal(entry, spec.unit);
    let detail = '';
    if (entry && typeof entry === 'object') {
      if (typeof entry.note === 'string') {
        detail = entry.note;
      } else if (typeof entry.target === 'string') {
        detail = `target ${entry.target}`;
      } else if (Array.isArray(entry.refRangePx) && entry.refRangePx.length === 2) {
        const rangeStr = `ref ${fmtNum(entry.refRangePx[0])}-${fmtNum(entry.refRangePx[1])} ${spec.unit}`;
        detail = entry.bone ? `bone ${entry.bone} | ${rangeStr}` : rangeStr;
      }
    }
    return {
      ...spec,
      status,
      value,
      valueText,
      refRangePx: entry && typeof entry === 'object' && Array.isArray(entry.refRangePx) ? entry.refRangePx : null,
      detail,
    };
  });

  let overallStatus = 'pass';
  if (gates.some((g) => g.status === 'fail')) overallStatus = 'fail';
  else if (gates.every((g) => g.status === 'na')) overallStatus = 'na';
  else if (gates.some((g) => g.status === 'warn')) overallStatus = 'warn';

  return {
    loaded: true,
    overallStatus,
    gates,
    hints,
  };
}

function loadImageUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
    img.src = url;
  });
}

function resolveRelativeUrl(baseUrl, relPath) {
  try {
    return new URL(relPath, baseUrl).href;
  } catch (_err) {
    const cleanBase = baseUrl.replace(/[^/]*$/, '');
    return cleanBase + relPath.replace(/^\.\//, '');
  }
}

export class PortraitSpinePlayer {
  /**
   * @param {{
   *   canvas: HTMLCanvasElement,
   *   preferWebGL2?: boolean,
   *   defaultDuration?: number,
   *   onStateChange?: (state: object) => void,
   *   onTimeUpdate?: (time: number, duration: number) => void,
   *   onRigLoaded?: (info: object) => void,
   *   onError?: (err: Error) => void,
   * }} options
   */
  constructor(options) {
    if (!options?.canvas) {
      throw new Error('PortraitSpinePlayer requires options.canvas');
    }
    this.canvas = options.canvas;
    this.renderer = new SpineCanvasRenderer(this.canvas, {
      preferWebGL2: options.preferWebGL2 ?? true,
      webglSurface: options.webglSurface,
    });

    this.defaultDuration = options.defaultDuration ?? 8.0;
    this.onStateChange = options.onStateChange ?? (() => {});
    this.onTimeUpdate = options.onTimeUpdate ?? (() => {});
    this.onRigLoaded = options.onRigLoaded ?? (() => {});
    this.onError = options.onError ?? (() => {});

    /** @type {Spine42Skeleton|null} */
    this.skeleton = null;
    this.rigName = '';
    this.animationName = 'idle';
    this.duration = this.defaultDuration;
    this.currentTime = 0.0;
    this.playing = false;
    this.speed = 1.0;
    this.background = 'dark';
    this.compareFrame0 = false;
    this.showParts = false;
    this.showBones = false;
    this.liveMotionScales = new Map();
    this.liveSlotOrder = null;
    this.liveHiddenSlots = null;
    this.highlightSlot = null;
    this.healthReport = normalizeHealthReport(null);

    this._rafId = null;
    this._lastFrameMs = null;
    this._objectUrls = [];

    this._tick = this._tick.bind(this);
  }

  getState() {
    return {
      rigName: this.rigName,
      loaded: this.skeleton !== null,
      animationName: this.animationName,
      duration: this.duration,
      currentTime: this.currentTime,
      playing: this.playing,
      speed: this.speed,
      background: this.background,
      compareFrame0: this.compareFrame0,
      showParts: this.showParts,
      showBones: this.showBones,
      backend: this.renderer.backend,
      healthReport: this.healthReport,
    };
  }

  _emitState() {
    this.onStateChange(this.getState());
  }

  _revokeObjectUrls() {
    if (typeof URL !== 'undefined' && URL.revokeObjectURL) {
      for (const u of this._objectUrls) {
        URL.revokeObjectURL(u);
      }
    }
    this._objectUrls = [];
  }

  /**
   * Load rig directly from in-memory JSON, image map, and optional health report.
   *
   * @param {{
   *   spineJson: object|string,
   *   images: Map<string, HTMLImageElement>|Record<string, HTMLImageElement>,
   *   healthReport?: object|null,
   *   name?: string,
   *   animationName?: string,
   *   autoPlay?: boolean
   * }} bundle
   */
  loadRigData(bundle) {
    const sk = new Spine42Skeleton(bundle.spineJson);
    this.skeleton = sk;
    this.renderer.setTextures(bundle.images);

    const anims = sk.getAnimationNames();
    this.animationName =
      bundle.animationName && anims.includes(bundle.animationName)
        ? bundle.animationName
        : anims.includes('idle')
          ? 'idle'
          : (anims[0] ?? '');

    const detectedDur = this.animationName ? sk.getAnimationDuration(this.animationName) : 0;
    this.duration = detectedDur > 0 ? detectedDur : this.defaultDuration;
    this.currentTime = 0.0;
    this.rigName = bundle.name ?? 'Untitled Rig';
    this.healthReport = normalizeHealthReport(bundle.healthReport ?? null);

    if (bundle.live) {
      this.liveMotionScales = new Map();
      this.liveSlotOrder = bundle.live.slotOrder ?? null;
      this.liveHiddenSlots = bundle.live.hiddenSlots ?? null;
      this.highlightSlot = null;
    }
    this.resize();
    this.renderCurrentFrame();

    const info = {
      name: this.rigName,
      spineVersion: sk.skeleton.spine ?? '4.2',
      width: sk.getCanvasSpec().width,
      height: sk.getCanvasSpec().height,
      bones: sk.bones.length,
      ik: sk.ik.length,
      slots: sk.slots.length,
      animations: anims,
      animationName: this.animationName,
      duration: this.duration,
      backend: this.renderer.backend,
      healthReport: this.healthReport,
    };
    this.onRigLoaded(info);
    this._emitState();

    if (bundle.autoPlay !== false && this.animationName) {
      this.play();
    }
    return info;
  }

  /**
   * Load a rig bundle from HTTP/local URLs: `<name>.spine.json` + `images/*.png` + optional `health-report.json`.
   *
   * @param {string} spineJsonUrl
   * @param {{ imagesBaseUrl?: string, healthReportUrl?: string|null, autoPlay?: boolean }} [options]
   */
  async loadFromUrl(spineJsonUrl, options = {}) {
    this._revokeObjectUrls();
    const absSpineUrl =
      typeof window !== 'undefined'
        ? new URL(spineJsonUrl, window.location.href).href
        : spineJsonUrl;

    const resp = await fetch(absSpineUrl, {signal:options.signal});
    if (!resp.ok) {
      throw new Error(`Failed to fetch Spine JSON (${resp.status}): ${spineJsonUrl}`);
    }
    const spineJson = await resp.json();

    const imgDir = spineJson.skeleton?.images ?? './images/';
    const imagesBase = options.imagesBaseUrl
      ? resolveRelativeUrl(absSpineUrl, options.imagesBaseUrl)
      : resolveRelativeUrl(absSpineUrl, imgDir.endsWith('/') ? imgDir : imgDir + '/');

    const needed = new Set();
    for (const slot of spineJson.slots ?? []) {
      if (slot.attachment) needed.add(slot.attachment);
    }

    const images = new Map();
    await Promise.all(
      Array.from(needed).map(async (attName) => {
        const imgUrl = resolveRelativeUrl(imagesBase, `${attName}.png`);
        const img = await loadImageUrl(imgUrl);
        images.set(attName, img);
      })
    );

    let healthReport = null;
    const healthUrl =
      options.healthReportUrl !== undefined
        ? options.healthReportUrl
        : resolveRelativeUrl(absSpineUrl, './health-report.json');
    if (healthUrl) {
      try {
        const hrResp = await fetch(healthUrl);
        if (hrResp.ok) {
          healthReport = await hrResp.json();
        }
      } catch (_err) {
        healthReport = null;
      }
    }

    const fileName = absSpineUrl.split('/').pop() ?? 'rig';
    const name = fileName.replace(/\.spine\.json$/i, '').replace(/\.json$/i, '');

    if (options.signal?.aborted) return null;
    return this.loadRigData({
      spineJson,
      images,
      healthReport,
      name,
      autoPlay: options.autoPlay ?? true,
      live: options.live,
    });
  }

  /**
   * Load a rig bundle from local File objects (e.g. `<input type="file" webkitdirectory>` or drag-and-drop).
   *
   * @param {FileList|File[]} fileList
   * @param {{ autoPlay?: boolean }} [options]
   */
  async loadFromFiles(fileList, options = {}) {
    this._revokeObjectUrls();
    const files = Array.from(fileList ?? []);
    if (files.length === 0) {
      throw new Error('No files provided');
    }

    const spineFile =
      files.find((f) => /\.spine\.json$/i.test(f.name)) ??
      files.find((f) => /\.json$/i.test(f.name) && !/health-report|rig-build-report|parts\.json/i.test(f.name));
    if (!spineFile) {
      throw new Error('Could not find a *.spine.json file in the selected files');
    }

    const spineText = await spineFile.text();
    const spineJson = JSON.parse(spineText);

    const healthFile = files.find((f) => /health-report\.json$/i.test(f.name));
    let healthReport = null;
    if (healthFile) {
      try {
        healthReport = JSON.parse(await healthFile.text());
      } catch (_err) {
        healthReport = null;
      }
    }

    const pngFiles = files.filter((f) => /\.png$/i.test(f.name));
    const images = new Map();
    await Promise.all(
      pngFiles.map(async (f) => {
        const url = URL.createObjectURL(f);
        this._objectUrls.push(url);
        const img = await loadImageUrl(url);
        const baseName = f.name.replace(/\.png$/i, '');
        images.set(baseName, img);
        images.set(f.name, img);
      })
    );

    const name = spineFile.name.replace(/\.spine\.json$/i, '').replace(/\.json$/i, '');
    return this.loadRigData({
      spineJson,
      images,
      healthReport,
      name,
      autoPlay: options.autoPlay ?? true,
    });
  }

  resize() {
    this.renderer.resize();
    this.renderCurrentFrame();
  }

  play() {
    if (!this.skeleton || this.playing) return;
    this.playing = true;
    this._lastFrameMs = null;
    this._emitState();
    this._rafId = requestAnimationFrame(this._tick);
  }

  pause() {
    if (!this.playing) return;
    this.playing = false;
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
    this._lastFrameMs = null;
    this._emitState();
  }

  togglePlay() {
    if (this.playing) this.pause();
    else this.play();
  }

  setSpeed(speed) {
    const numeric = Number(speed);
    if (!Number.isFinite(numeric) || numeric <= 0) return;
    this.speed = numeric;
    this._emitState();
  }

  setTime(t) {
    const dur = this.duration > 0 ? this.duration : this.defaultDuration;
    this.currentTime = Math.min(dur, Math.max(0, Number(t) || 0));
    this.renderCurrentFrame();
    this.onTimeUpdate(this.currentTime, dur);
  }

  setBackground(mode) {
    if (mode !== 'dark' && mode !== 'light' && mode !== 'checker') return;
    this.background = mode;
    this.renderCurrentFrame();
    this._emitState();
  }

  setCompareFrame0(enabled) {
    this.compareFrame0 = !!enabled;
    this.renderCurrentFrame();
    this._emitState();
  }

  setShowParts(enabled) {
    this.showParts = !!enabled;
    this.renderCurrentFrame();
    this._emitState();
  }

  setShowBones(enabled) {
    this.showBones = !!enabled;
    this.renderCurrentFrame();
    this._emitState();
  }

  setLiveMotionScales(moduleScaleMap, bakedScaleMap, globalScale = 1) {
    const desired = moduleScaleMap instanceof Map ? moduleScaleMap : new Map(Object.entries(moduleScaleMap || {}));
    const baked = bakedScaleMap instanceof Map ? bakedScaleMap : new Map(Object.entries(bakedScaleMap || {}));
    const softIds = [...desired.keys()].sort((left, right) => right.length - left.length);
    const perBone = new Map();
    const timelines = this.skeleton?.json.animations?.[this.animationName]?.bones || {};
    for (const boneName of Object.keys(timelines)) {
      const id = boneName.startsWith('module_') ? boneName.slice(7)
        : softIds.find(candidate => boneName.startsWith('bend_' + candidate)
          && /^\d+$/.test(boneName.slice(5 + candidate.length)));
      const base = baked.get(id) ?? 1;
      const value = desired.get(id) ?? base;
      perBone.set(boneName, base > 0 ? value / base * globalScale : 1);
    }
    this.liveMotionScales = perBone;
    this._refreshEditedFrame();
  }

  setLiveSlotOrder(slotNames) {
    this.liveSlotOrder = slotNames;
    this._refreshEditedFrame();
  }

  setLiveHiddenSlots(hiddenSet) {
    this.liveHiddenSlots = hiddenSet;
    this._refreshEditedFrame();
  }

  setHighlightSlot(slotNameOrNull) {
    this.highlightSlot = slotNameOrNull;
    this._refreshEditedFrame();
  }

  _refreshEditedFrame() {
    if (!this.playing) this.renderCurrentFrame();
  }

  renderCurrentFrame() {
    if (!this.skeleton) {
      this.renderer.render(null, { background: this.background });
      return;
    }
    if (this.compareFrame0 || !this.animationName) {
      this.skeleton.setupPose();
      this.skeleton.updateWorld();
    } else {
      this.skeleton.apply(this.animationName, this.currentTime, {
        loop: true,
        duration: this.duration,
        boneMotionScales: this.liveMotionScales,
      });
    }
    this.renderer.render(this.skeleton, {
      background: this.background,
      compareFrame0: this.compareFrame0,
      showParts: this.showParts,
      showBones: this.showBones,
      slotOrder: this.liveSlotOrder,
      hiddenSlots: this.liveHiddenSlots,
      highlightSlot: this.highlightSlot,
    });
  }

  _tick(nowMs) {
    if (!this.playing) return;
    if (this._lastFrameMs !== null) {
      const dtSec = Math.min(0.1, Math.max(0, (nowMs - this._lastFrameMs) / 1000));
      const dur = this.duration > 0 ? this.duration : this.defaultDuration;
      this.currentTime = (this.currentTime + dtSec * this.speed) % dur;
      this.renderCurrentFrame();
      this.onTimeUpdate(this.currentTime, dur);
    }
    this._lastFrameMs = nowMs;
    this._rafId = requestAnimationFrame(this._tick);
  }

  dispose() {
    this.pause();
    this._revokeObjectUrls();
    this.renderer.dispose();
    this.skeleton = null;
  }
}
