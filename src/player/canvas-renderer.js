/**
 * Spine 4.2 Canvas Renderer (WebGL2 premultiplied-alpha textured mesh renderer
 * with automatic Canvas 2D affine-triangle fallback + crisp 2D wireframe/part overlays).
 *
 * Zero external dependencies.
 */

const VERT_SRC = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUv;
uniform vec2 uCanvasSize;
out vec2 vUv;
void main() {
  vec2 ndc = vec2(
    (aPos.x / uCanvasSize.x) * 2.0 - 1.0,
    1.0 - (aPos.y / uCanvasSize.y) * 2.0
  );
  gl_Position = vec4(ndc, 0.0, 1.0);
  vUv = aUv;
}
`;

const FRAG_SRC = `#version 300 es
precision highp float;
uniform sampler2D uTex;
in vec2 vUv;
out vec4 outColor;
void main() {
  outColor = texture(uTex, vUv);
}
`;

/**
 * Compute 2D convex hull of a flat Float64Array [x0, y0, x1, y1, ...] using Andrew's monotone chain.
 * @param {Float64Array|number[]} flatVerts
 * @returns {Array<[number, number]>}
 */
export function computeConvexHull(flatVerts) {
  const n = flatVerts.length >> 1;
  if (n <= 2) {
    const out = [];
    for (let i = 0; i < n; i++) out.push([flatVerts[2 * i], flatVerts[2 * i + 1]]);
    return out;
  }
  const pts = new Array(n);
  for (let i = 0; i < n; i++) {
    pts[i] = [flatVerts[2 * i], flatVerts[2 * i + 1]];
  }
  pts.sort((a, b) => (a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]));

  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

  const lower = [];
  for (let i = 0; i < n; i++) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], pts[i]) <= 0) {
      lower.pop();
    }
    lower.push(pts[i]);
  }
  const upper = [];
  for (let i = n - 1; i >= 0; i--) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], pts[i]) <= 0) {
      upper.pop();
    }
    upper.push(pts[i]);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

/**
 * Deterministic HSL color for slot index.
 * @param {number} index
 * @param {number} alpha
 * @returns {string}
 */
function slotColor(index, alpha) {
  const hue = (index * 137.508) % 360;
  return `hsla(${hue.toFixed(1)}, 78%, 58%, ${alpha})`;
}

export class SpineCanvasRenderer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{ preferWebGL2?: boolean }} [options]
   */
  constructor(canvas, options = {}) {
    if (!canvas) {
      throw new Error('SpineCanvasRenderer requires a target HTMLCanvasElement');
    }
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.preferWebGL2 = options.preferWebGL2 ?? true;

    /** @type {Map<string, HTMLImageElement|ImageBitmap|HTMLCanvasElement>} */
    this.images = new Map();

    /** @type {HTMLCanvasElement|null} */
    this._glCanvas = null;
    /** @type {WebGL2RenderingContext|null} */
    this._gl = null;
    this._glProgram = null;
    this._glVao = null;
    this._glVbo = null;
    this._glIbo = null;
    this._glUniformCanvasSize = null;
    this._glUniformTex = null;
    /** @type {Map<string, WebGLTexture>} */
    this._glTextures = new Map();
    this._checkerPattern = null;

    this.backend = 'canvas2d';
    if (this.preferWebGL2) {
      this._initWebGL2();
    }
  }

  _initWebGL2() {
    if (typeof document === 'undefined') return;
    try {
      const glCanvas = document.createElement('canvas');
      const gl = glCanvas.getContext('webgl2', {
        alpha: true,
        premultipliedAlpha: true,
        antialias: true,
        preserveDrawingBuffer: true,
      });
      if (!gl) return;

      const vs = this._compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
      const fs = this._compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
      if (!vs || !fs) return;

      const prog = gl.createProgram();
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      gl.deleteShader(vs);
      gl.deleteShader(fs);

      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        gl.deleteProgram(prog);
        return;
      }

      const vao = gl.createVertexArray();
      const vbo = gl.createBuffer();
      const ibo = gl.createBuffer();

      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bindVertexArray(null);

      this._glCanvas = glCanvas;
      this._gl = gl;
      this._glProgram = prog;
      this._glVao = vao;
      this._glVbo = vbo;
      this._glIbo = ibo;
      this._glUniformCanvasSize = gl.getUniformLocation(prog, 'uCanvasSize');
      this._glUniformTex = gl.getUniformLocation(prog, 'uTex');
      this.backend = 'webgl2';
    } catch (_err) {
      this.backend = 'canvas2d';
      this._gl = null;
      this._glCanvas = null;
    }
  }

  _compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  /**
   * Register loaded part images keyed by attachment name (without .png), filename, or slot name.
   *
   * @param {Map<string, HTMLImageElement|ImageBitmap|HTMLCanvasElement>|Record<string, HTMLImageElement>} imageMap
   */
  setTextures(imageMap) {
    this._clearGlTextures();
    this.images.clear();

    const entries = imageMap instanceof Map ? imageMap.entries() : Object.entries(imageMap ?? {});
    for (const [key, img] of entries) {
      if (!img) continue;
      this.images.set(key, img);
      if (key.endsWith('.png')) {
        this.images.set(key.slice(0, -4), img);
      }
    }
  }

  _resolveImage(geom) {
    const attPath = geom.attachment?.path;
    if (attPath && this.images.has(attPath)) return { key: attPath, img: this.images.get(attPath) };
    if (geom.attachmentName && this.images.has(geom.attachmentName)) {
      return { key: geom.attachmentName, img: this.images.get(geom.attachmentName) };
    }
    if (this.images.has(geom.slotName)) {
      return { key: geom.slotName, img: this.images.get(geom.slotName) };
    }
    return null;
  }

  _getOrCreateGlTexture(key, img) {
    const gl = this._gl;
    if (!gl) return null;
    if (this._glTextures.has(key)) {
      return this._glTextures.get(key);
    }
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.bindTexture(gl.TEXTURE_2D, null);
    this._glTextures.set(key, tex);
    return tex;
  }

  _clearGlTextures() {
    if (!this._gl) return;
    for (const tex of this._glTextures.values()) {
      this._gl.deleteTexture(tex);
    }
    this._glTextures.clear();
  }

  /**
   * Resize the backing store of the display canvas to match CSS dimensions * devicePixelRatio.
   * @param {number} [cssWidth]
   * @param {number} [cssHeight]
   */
  resize(cssWidth, cssHeight) {
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const w = Math.max(1, Math.round((cssWidth ?? this.canvas.clientWidth ?? 640) * dpr));
    const h = Math.max(1, Math.round((cssHeight ?? this.canvas.clientHeight ?? 640) * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  /**
   * Compute viewport fit transform mapping skeleton canvas space into display canvas pixels.
   *
   * @param {{ width: number, height: number }} spec
   * @param {number} paddingPx
   */
  _computeViewport(spec, paddingPx = 24) {
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const pad = paddingPx * dpr;
    const availW = Math.max(1, cw - pad * 2);
    const availH = Math.max(1, ch - pad * 2);
    const scale = Math.min(availW / spec.width, availH / spec.height);
    const drawW = spec.width * scale;
    const drawH = spec.height * scale;
    const offsetX = (cw - drawW) * 0.5;
    const offsetY = (ch - drawH) * 0.5;
    return { scale, offsetX, offsetY, drawW, drawH, dpr };
  }

  _drawBackground(mode) {
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;

    if (mode === 'light') {
      ctx.fillStyle = '#dcdcdc';
      ctx.fillRect(0, 0, cw, ch);
      return;
    }
    if (mode === 'checker') {
      if (!this._checkerPattern && typeof document !== 'undefined') {
        const tile = document.createElement('canvas');
        const sz = 24;
        tile.width = sz * 2;
        tile.height = sz * 2;
        const tctx = tile.getContext('2d');
        tctx.fillStyle = '#2a2d34';
        tctx.fillRect(0, 0, sz * 2, sz * 2);
        tctx.fillStyle = '#363a44';
        tctx.fillRect(0, 0, sz, sz);
        tctx.fillRect(sz, sz, sz, sz);
        this._checkerPattern = ctx.createPattern(tile, 'repeat');
      }
      if (this._checkerPattern) {
        ctx.fillStyle = this._checkerPattern;
        ctx.fillRect(0, 0, cw, ch);
      } else {
        ctx.fillStyle = '#2a2d34';
        ctx.fillRect(0, 0, cw, ch);
      }
      return;
    }
    // Default: dark
    ctx.fillStyle = '#141414';
    ctx.fillRect(0, 0, cw, ch);
  }

  /**
   * Render all slot meshes onto the internal WebGL2 canvas, then composite onto the 2D canvas.
   */
  _renderMeshesWebGL2(geometries, spec, viewport) {
    const gl = this._gl;
    const glCanvas = this._glCanvas;
    const targetW = Math.max(1, Math.round(spec.width));
    const targetH = Math.max(1, Math.round(spec.height));

    if (glCanvas.width !== targetW || glCanvas.height !== targetH) {
      glCanvas.width = targetW;
      glCanvas.height = targetH;
    }

    gl.viewport(0, 0, targetW, targetH);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(this._glProgram);
    gl.uniform2f(this._glUniformCanvasSize, spec.width, spec.height);
    gl.uniform1i(this._glUniformTex, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindVertexArray(this._glVao);

    for (let i = 0; i < geometries.length; i++) {
      const geom = geometries[i];
      const resolved = this._resolveImage(geom);
      if (!resolved) continue;
      const tex = this._getOrCreateGlTexture(resolved.key, resolved.img);
      if (!tex) continue;

      const n = geom.vertexCount;
      const interleaved = new Float32Array(n * 4);
      for (let k = 0; k < n; k++) {
        interleaved[k * 4] = geom.vertices[2 * k];
        interleaved[k * 4 + 1] = geom.vertices[2 * k + 1];
        interleaved[k * 4 + 2] = geom.uvs[2 * k];
        interleaved[k * 4 + 3] = geom.uvs[2 * k + 1];
      }

      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.bindBuffer(gl.ARRAY_BUFFER, this._glVbo);
      gl.bufferData(gl.ARRAY_BUFFER, interleaved, gl.STREAM_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this._glIbo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, geom.triangles, gl.STREAM_DRAW);
      gl.drawElements(gl.TRIANGLES, geom.triangles.length, gl.UNSIGNED_INT, 0);
    }

    gl.bindVertexArray(null);
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
    this.ctx.drawImage(
      glCanvas,
      viewport.offsetX,
      viewport.offsetY,
      viewport.drawW,
      viewport.drawH
    );
  }

  /**
   * Canvas 2D fallback: affine triangle texture mapping with subtle centroid expansion to avoid seams.
   */
  _renderMeshesCanvas2D(geometries, viewport) {
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(viewport.offsetX, viewport.offsetY);
    ctx.scale(viewport.scale, viewport.scale);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    const bleed = 0.35;

    for (let g = 0; g < geometries.length; g++) {
      const geom = geometries[g];
      const resolved = this._resolveImage(geom);
      if (!resolved) continue;
      const img = resolved.img;
      const iw = img.naturalWidth || img.width;
      const ih = img.naturalHeight || img.height;
      if (!iw || !ih) continue;

      const verts = geom.vertices;
      const uvs = geom.uvs;
      const tris = geom.triangles;

      for (let t = 0; t < tris.length; t += 3) {
        const i0 = tris[t];
        const i1 = tris[t + 1];
        const i2 = tris[t + 2];

        const x0 = verts[2 * i0];
        const y0 = verts[2 * i0 + 1];
        const x1 = verts[2 * i1];
        const y1 = verts[2 * i1 + 1];
        const x2 = verts[2 * i2];
        const y2 = verts[2 * i2 + 1];

        const u0 = uvs[2 * i0] * iw;
        const v0 = uvs[2 * i0 + 1] * ih;
        const u1 = uvs[2 * i1] * iw;
        const v1 = uvs[2 * i1 + 1] * ih;
        const u2 = uvs[2 * i2] * iw;
        const v2 = uvs[2 * i2 + 1] * ih;

        const det = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0);
        if (Math.abs(det) < 1e-8) continue;

        const m11 = ((x1 - x0) * (v2 - v0) - (x2 - x0) * (v1 - v0)) / det;
        const m12 = ((y1 - y0) * (v2 - v0) - (y2 - y0) * (v1 - v0)) / det;
        const m21 = ((u1 - u0) * (x2 - x0) - (u2 - u0) * (x1 - x0)) / det;
        const m22 = ((u1 - u0) * (y2 - y0) - (u2 - u0) * (y1 - y0)) / det;
        const dx = x0 - m11 * u0 - m21 * v0;
        const dy = y0 - m12 * u0 - m22 * v0;

        // Expand triangle slightly outward from centroid to prevent anti-aliasing seam gaps
        const cx = (x0 + x1 + x2) / 3;
        const cy = (y0 + y1 + y2) / 3;
        const expand = (px, py) => {
          const vx = px - cx;
          const vy = py - cy;
          const len = Math.hypot(vx, vy);
          if (len < 1e-6) return [px, py];
          return [px + (vx / len) * bleed, py + (vy / len) * bleed];
        };
        const [ex0, ey0] = expand(x0, y0);
        const [ex1, ey1] = expand(x1, y1);
        const [ex2, ey2] = expand(x2, y2);

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(ex0, ey0);
        ctx.lineTo(ex1, ey1);
        ctx.lineTo(ex2, ey2);
        ctx.closePath();
        ctx.clip();
        ctx.transform(m11, m12, m21, m22, dx, dy);
        ctx.drawImage(img, 0, 0);
        ctx.restore();
      }
    }

    ctx.restore();
  }

  _drawPartsOverlay(geometries, viewport) {
    const ctx = this.ctx;
    const { scale, offsetX, offsetY, dpr } = viewport;

    ctx.save();
    for (let i = 0; i < geometries.length; i++) {
      const geom = geometries[i];
      const hull = computeConvexHull(geom.vertices);
      if (hull.length < 3) continue;

      ctx.beginPath();
      for (let k = 0; k < hull.length; k++) {
        const px = offsetX + hull[k][0] * scale;
        const py = offsetY + hull[k][1] * scale;
        if (k === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fillStyle = slotColor(i, 0.16);
      ctx.fill();
      ctx.strokeStyle = slotColor(i, 0.82);
      ctx.lineWidth = 1.25 * dpr;
      ctx.stroke();

      // Label at hull centroid
      let cx = 0;
      let cy = 0;
      for (let k = 0; k < hull.length; k++) {
        cx += hull[k][0];
        cy += hull[k][1];
      }
      cx = offsetX + (cx / hull.length) * scale;
      cy = offsetY + (cy / hull.length) * scale;

      const label = geom.slotName;
      ctx.font = `600 ${Math.round(10 * dpr)}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
      const tw = ctx.measureText(label).width;
      const padX = 4 * dpr;
      const bh = 14 * dpr;
      ctx.fillStyle = 'rgba(15, 17, 22, 0.78)';
      ctx.fillRect(cx - tw * 0.5 - padX, cy - bh * 0.5, tw + padX * 2, bh);
      ctx.fillStyle = slotColor(i, 0.98);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, cx, cy);
    }
    ctx.restore();
  }

  _drawBonesOverlay(skeleton, spec, viewport) {
    const ctx = this.ctx;
    const { scale, offsetX, offsetY, dpr } = viewport;
    const segments = skeleton.getBoneWorldSegments(spec);
    const ikTargets = skeleton.getIkTargets(spec);

    const toScreen = (x, y) => [offsetX + x * scale, offsetY + y * scale];

    ctx.save();

    // 1. Parent -> child hierarchy links (subtle dashed)
    ctx.setLineDash([3 * dpr, 3 * dpr]);
    ctx.lineWidth = 1 * dpr;
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)';
    for (const seg of segments) {
      if (seg.parentX === null || seg.isIkTarget) continue;
      const [px, py] = toScreen(seg.parentX, seg.parentY);
      const [bx, by] = toScreen(seg.x1, seg.y1);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // 2. Bone tapered bodies (for bones with length > 0)
    for (const seg of segments) {
      if (seg.isIkTarget) continue;
      const [x1, y1] = toScreen(seg.x1, seg.y1);
      const [x2, y2] = toScreen(seg.x2, seg.y2);
      const dx = x2 - x1;
      const dy = y2 - y1;
      const len = Math.hypot(dx, dy);

      if (len > 2 * dpr) {
        const ux = dx / len;
        const uy = dy / len;
        const nx = -uy;
        const ny = ux;
        const w = Math.min(6 * dpr, Math.max(2.5 * dpr, len * 0.12));
        const midX = x1 + dx * 0.2;
        const midY = y1 + dy * 0.2;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(midX + nx * w, midY + ny * w);
        ctx.lineTo(x2, y2);
        ctx.lineTo(midX - nx * w, midY - ny * w);
        ctx.closePath();

        ctx.fillStyle = seg.isIkBone
          ? 'rgba(251, 191, 36, 0.28)'
          : 'rgba(56, 189, 248, 0.24)';
        ctx.fill();
        ctx.strokeStyle = seg.isIkBone
          ? 'rgba(251, 191, 36, 0.95)'
          : 'rgba(56, 189, 248, 0.92)';
        ctx.lineWidth = 1.4 * dpr;
        ctx.stroke();
      }
    }

    // 3. Joint circles
    for (const seg of segments) {
      if (seg.isIkTarget) continue;
      const [x1, y1] = toScreen(seg.x1, seg.y1);
      const r = (seg.parent === null ? 4.5 : 3.2) * dpr;
      ctx.beginPath();
      ctx.arc(x1, y1, r, 0, Math.PI * 2);
      ctx.fillStyle = seg.isIkBone ? '#fbbf24' : '#38bdf8';
      ctx.fill();
      ctx.strokeStyle = '#0b0e14';
      ctx.lineWidth = 1.2 * dpr;
      ctx.stroke();
    }

    // 4. IK target pins (highlighted diamond + crosshair + label)
    for (const ik of ikTargets) {
      const [tx, ty] = toScreen(ik.x, ik.y);
      const r = 7 * dpr;

      ctx.beginPath();
      ctx.moveTo(tx, ty - r);
      ctx.lineTo(tx + r, ty);
      ctx.lineTo(tx, ty + r);
      ctx.lineTo(tx - r, ty);
      ctx.closePath();
      ctx.fillStyle = 'rgba(244, 63, 94, 0.35)';
      ctx.fill();
      ctx.strokeStyle = '#fb7185';
      ctx.lineWidth = 1.8 * dpr;
      ctx.stroke();

      // Crosshair
      const cr = 10 * dpr;
      ctx.beginPath();
      ctx.moveTo(tx - cr, ty);
      ctx.lineTo(tx + cr, ty);
      ctx.moveTo(tx, ty - cr);
      ctx.lineTo(tx, ty + cr);
      ctx.strokeStyle = 'rgba(251, 113, 133, 0.85)';
      ctx.lineWidth = 1.1 * dpr;
      ctx.stroke();

      // Label
      const label = `IK:${ik.name}`;
      ctx.font = `600 ${Math.round(10 * dpr)}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
      const tw = ctx.measureText(label).width;
      const lx = tx + 10 * dpr;
      const ly = ty - 12 * dpr;
      ctx.fillStyle = 'rgba(15, 17, 22, 0.85)';
      ctx.fillRect(lx - 3 * dpr, ly - 7 * dpr, tw + 6 * dpr, 14 * dpr);
      ctx.fillStyle = '#fda4af';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, lx, ly);
    }

    ctx.restore();
  }

  _drawCompareBadge(viewport) {
    const ctx = this.ctx;
    const dpr = viewport.dpr;
    const text = 'SETUP / FRAME 0';
    ctx.save();
    ctx.font = `700 ${Math.round(11 * dpr)}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
    const tw = ctx.measureText(text).width;
    const x = 14 * dpr;
    const y = 14 * dpr;
    const padX = 8 * dpr;
    const h = 22 * dpr;
    ctx.fillStyle = 'rgba(245, 158, 11, 0.92)';
    ctx.fillRect(x, y, tw + padX * 2, h);
    ctx.fillStyle = '#111318';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x + padX, y + h * 0.5);
    ctx.restore();
  }

  /**
   * Render the skeleton's current evaluated pose onto the canvas.
   *
   * @param {import('./spine42-runtime.js').Spine42Skeleton} skeleton
   * @param {{
   *   background?: 'dark' | 'light' | 'checker',
   *   compareFrame0?: boolean,
   *   showParts?: boolean,
   *   showBones?: boolean,
   *   padding?: number
   * }} [options]
   */
  render(skeleton, options = {}) {
    const background = options.background ?? 'dark';
    const compareFrame0 = !!options.compareFrame0;
    const showParts = !!options.showParts;
    const showBones = !!options.showBones;
    const padding = options.padding ?? 24;

    this._drawBackground(background);
    if (!skeleton) return;

    const spec = skeleton.getCanvasSpec();
    const viewport = this._computeViewport(spec, padding);

    const geometries = [];
    for (let i = 0; i < skeleton.slots.length; i++) {
      const geom = skeleton.computeWorldGeometry(skeleton.slots[i], spec);
      if (geom) geometries.push(geom);
    }

    if (this._gl && this._glCanvas) {
      this._renderMeshesWebGL2(geometries, spec, viewport);
    } else {
      this._renderMeshesCanvas2D(geometries, viewport);
    }

    if (showParts) {
      this._drawPartsOverlay(geometries, viewport);
    }
    if (showBones) {
      this._drawBonesOverlay(skeleton, spec, viewport);
    }
    if (compareFrame0) {
      this._drawCompareBadge(viewport);
    }
  }

  dispose() {
    this._clearGlTextures();
    if (this._gl) {
      if (this._glVbo) this._gl.deleteBuffer(this._glVbo);
      if (this._glIbo) this._gl.deleteBuffer(this._glIbo);
      if (this._glVao) this._gl.deleteVertexArray(this._glVao);
      if (this._glProgram) this._gl.deleteProgram(this._glProgram);
    }
    this._gl = null;
    this._glCanvas = null;
    this.images.clear();
  }
}
