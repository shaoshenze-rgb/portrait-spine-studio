/** Batch review, hard-lock approval, fix-note rejection, and 4K unified export workbench. */

export function createBatchWorkbench({ request, onSelectJob, onClearJob }) {
  const $ = (id) => document.getElementById(id);
  const shell = $('batchWorkbench');
  const mainSlot = $('batchMainSlot');
  const canvasWorkbenchEl = $('canvasWorkbench');
  const originalParent = canvasWorkbenchEl ? canvasWorkbenchEl.parentNode : null;
  const originalNext = canvasWorkbenchEl ? canvasWorkbenchEl.nextSibling : null;

  let batch = null;
  let activeItemId = null;
  let filterKey = 'all';
  let exportPollTimer = null;
  let busyAction = false;

  const esc = (s) =>
    String(s ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;');

  function stopPoll() {
    if (exportPollTimer) {
      window.clearInterval(exportPollTimer);
      exportPollTimer = null;
    }
  }

  function statusLabel(item) {
    if (item.status === 'exported') return '已导出';
    if (item.locked || item.status === 'approved_locked') return '已锁定 🔒';
    if (item.status === 'needs_fix') return '待修改';
    if (item.status === 'failed') return '失败';
    if (item.status === 'preview_ready') return '待过目';
    if (item.status === 'prepared' || item.status === 'matted') return '待分层';
    return '待处理';
  }

  function matchesFilter(item) {
    if (filterKey === 'all') return true;
    if (filterKey === 'unreviewed') return !item.locked && item.status !== 'needs_fix' && item.status !== 'failed';
    if (filterKey === 'locked') return !!item.locked;
    if (filterKey === 'needs_fix') return item.status === 'needs_fix';
    if (filterKey === 'issues') return item.traffic_light === 'red' || item.traffic_light === 'yellow';
    return true;
  }

  function currentItem() {
    if (!batch || !activeItemId) return null;
    return (batch.items || []).find((it) => it.item_id === activeItemId) || null;
  }

  function readExportForm() {
    const sizePreset = $('bwSizePreset').value;
    const formats = Array.from(document.querySelectorAll('input[name="bwFormat"]:checked')).map((el) => el.value);
    const payload = {
      size_preset: sizePreset,
      align: $('bwAlign').value,
      upscale_policy: $('bwUpscalePolicy').value,
      fps: Number($('bwFps').value),
      formats,
      output_dir: $('bwOutputDir').value.trim() || null,
    };
    if (sizePreset === 'custom') {
      payload.width = Number($('bwCustomW').value) || 2160;
      payload.height = Number($('bwCustomH').value) || 3840;
    }
    return payload;
  }

  function hydrateExportForm() {
    if (!batch) return;
    const prof = batch.export_profile || {};
    const catalog = batch.export_catalog || {};
    const presets = catalog.size_presets || {};
    const presetSel = $('bwSizePreset');
    if (!presetSel.options.length && Object.keys(presets).length) {
      for (const [key, info] of Object.entries(presets)) {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = info.label || key;
        presetSel.appendChild(opt);
      }
    }
    presetSel.value = prof.size_preset || '4k_portrait';
    $('bwCustomSizeWrap').hidden = presetSel.value !== 'custom';
    $('bwCustomW').value = prof.width || 2160;
    $('bwCustomH').value = prof.height || 3840;
    $('bwAlign').value = prof.align || 'bottom_center';
    $('bwUpscalePolicy').value = prof.upscale_policy || 'allow_upscale';
    $('bwFps').value = String(prof.fps || 30);
    $('bwOutputDir').value = prof.output_dir || '';

    const selected = new Set(prof.formats || ['transparent_web']);
    for (const cb of document.querySelectorAll('input[name="bwFormat"]')) {
      cb.checked = selected.has(cb.value);
    }
  }

  function renderExportStatus() {
    if (!batch) return;
    const c = batch.counts || {};
    const lockedCount = c.locked || 0;
    const btn = $('bwRunExport');
    const prog = batch.export_progress;
    const last = batch.last_batch_export;
    const statusEl = $('bwExportStatus');

    if (prog && prog.running) {
      btn.disabled = true;
      btn.textContent = `批量导出中 (${prog.completed}/${prog.total})…`;
      statusEl.innerHTML = `正在按统一规格导出：<strong>${esc(prog.current_character || '')}</strong>（${prog.completed}/${prog.total}）`;
      return;
    }
    btn.disabled = busyAction || lockedCount === 0;
    btn.textContent = `一键批量导出已锁定角色 (${lockedCount} 张)`;
    if (last) {
      const errNote = last.error_count ? `，失败 ${last.error_count} 张` : '';
      statusEl.innerHTML = `最近导出：<strong>${last.exported_count} 张成功</strong>${errNote} · 输出目录：<code>${esc(last.output_dir || '')}</code>`;
    } else {
      statusEl.textContent = lockedCount
        ? `已锁定 ${lockedCount} 张，确认规格后点击右侧按钮即可统一批量导出。`
        : '请先逐张过目并点击「确认锁定」，锁定完成后即可一键统一导出。';
    }
  }

  function renderHeader() {
    if (!batch) return;
    const c = batch.counts || {};
    $('bwBatchTitle').textContent = batch.name || batch.batch_id;
    $('bwBatchIdBadge').textContent = batch.batch_id;
    $('bwSourceDir').textContent = batch.source_dir || '';
    $('bwStats').innerHTML = [
      `<span class="bw-stat-pill">共 ${c.total || 0} 张</span>`,
      `<span class="bw-stat-pill is-locked">已锁定 ${c.locked || 0}</span>`,
      `<span class="bw-stat-pill is-fix">待修改 ${c.needs_fix || 0}</span>`,
      c.exported ? `<span class="bw-stat-pill">已导出 ${c.exported}</span>` : '',
      c.failed ? `<span class="bw-stat-pill is-err">失败 ${c.failed}</span>` : '',
    ].join('');

    const ovLink = $('bwOverviewLink');
    if (batch.matte_overview_url) {
      ovLink.href = batch.matte_overview_url;
      ovLink.hidden = false;
    } else {
      ovLink.hidden = true;
    }
    renderExportStatus();
  }

  function renderSidebar() {
    if (!batch) return;
    const list = $('bwItemList');
    list.innerHTML = '';
    const items = (batch.items || []).filter(matchesFilter);
    if (!items.length) {
      list.innerHTML = '<div style="padding:16px;color:#8c919e;text-align:center;">当前筛选下暂无角色</div>';
      return;
    }
    for (const item of items) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'bw-item' + (item.item_id === activeItemId ? ' active' : '');
      const tagClass = item.status === 'exported'
        ? 'exported'
        : item.locked
        ? 'locked'
        : item.status === 'needs_fix'
        ? 'fix'
        : '';
      const sizeStr = Array.isArray(item.source_size) ? `${item.source_size[0]}×${item.source_size[1]}` : '';
      btn.innerHTML = `
        <img class="bw-thumb" src="${esc(item.thumb_url || '')}" alt="" loading="lazy" />
        <div class="bw-item-meta">
          <div class="bw-item-name">
            <span class="bw-dot ${esc(item.traffic_light || 'pending')}"></span>
            <span>${esc(item.display_name || item.character)}</span>
          </div>
          <div class="bw-item-sub">
            <span class="bw-tag ${tagClass}">${esc(statusLabel(item))}</span>
            ${sizeStr ? `<span>${esc(sizeStr)}</span>` : ''}
          </div>
        </div>`;
      btn.onclick = () => selectItem(item.item_id);
      list.appendChild(btn);
    }
  }

  function renderReviewBar() {
    const bar = $('bwReviewBar');
    const item = currentItem();
    if (!item) {
      bar.hidden = true;
      return;
    }
    bar.hidden = false;
    bar.className = 'bw-review-bar' + (item.locked ? ' is-locked' : item.status === 'needs_fix' ? ' is-fix' : '');

    const sizeTxt = Array.isArray(item.source_size) ? ` · 输入 ${item.source_size[0]}×${item.source_size[1]}` : '';
    $('bwActiveTitle').textContent = `${item.display_name || item.character}${sizeTxt}`;
    $('bwActiveSummary').textContent = item.locked
      ? '已确认锁定 🔒：受服务端写保护，重跑文件夹或集中返修打回项绝不会改动此角色。'
      : item.status === 'needs_fix'
      ? `已标记待修改：${item.fix_note || ''}（全部过目完后回对话说「修这一批里打回的」即可集中重修）`
      : item.traffic_summary || '请在下方观看完整循环与微调，满意后点「确认锁定」。';

    $('bwBtnLock').hidden = !!item.locked;
    $('bwBtnLock').disabled = busyAction || !item.job_id || item.traffic_light === 'pending';
    $('bwBtnUnlock').hidden = !item.locked;
    $('bwBtnUnlock').disabled = busyAction;
    $('bwFixNoteInput').hidden = !!item.locked;
    $('bwFixNoteInput').value = item.fix_note || '';
    $('bwBtnReject').hidden = !!item.locked;
    $('bwBtnReject').disabled = busyAction;
  }

  async function selectItem(itemId) {
    activeItemId = itemId;
    renderSidebar();
    renderReviewBar();
    const item = currentItem();
    if (!item) return;

    const url = new URL(window.location.href);
    url.searchParams.set('batch', batch.batch_id);
    if (item.job_id) url.searchParams.set('job', item.job_id);
    else url.searchParams.delete('job');
    window.history.replaceState({}, '', url.toString());

    const emptyEl = $('bwEmptyStage');
    if (!item.job_id) {
      emptyEl.hidden = false;
      emptyEl.textContent = item.error
        ? `该角色处理失败：${item.error}`
        : '该角色尚未完成语义分层预览，请稍候或从对话继续推进本批次。';
      onClearJob();
      return;
    }
    try {
      const job = await request(`/api/jobs/${encodeURIComponent(item.job_id)}`);
      if (job.canvas_mode && job.canvas_preview_html) {
        emptyEl.hidden = true;
        await onSelectJob(job);
      } else {
        emptyEl.hidden = false;
        emptyEl.textContent = `当前任务状态：${job.status}（尚未生成实时分层预览，等待分层完成后自动可用）`;
        onClearJob();
      }
    } catch (err) {
      emptyEl.hidden = false;
      emptyEl.textContent = `加载角色预览失败：${err.message}`;
      onClearJob();
    }
  }

  function jumpNextUnreviewed() {
    if (!batch || !batch.items?.length) return;
    const items = batch.items;
    const curIdx = items.findIndex((it) => it.item_id === activeItemId);
    for (let step = 1; step <= items.length; step++) {
      const cand = items[(curIdx + step) % items.length];
      if (!cand.locked && cand.status !== 'needs_fix' && cand.job_id) {
        selectItem(cand.item_id);
        return;
      }
    }
  }

  async function refreshBatch(preserveSelection = true) {
    if (!batch) return;
    const prevId = activeItemId;
    batch = await request(`/api/batches/${encodeURIComponent(batch.batch_id)}`);
    renderHeader();
    hydrateExportForm();
    renderSidebar();
    if (preserveSelection && prevId && (batch.items || []).some((it) => it.item_id === prevId)) {
      activeItemId = prevId;
      renderReviewBar();
    }
    if (batch.export_progress && batch.export_progress.running && !exportPollTimer) {
      exportPollTimer = window.setInterval(async () => {
        try {
          await refreshBatch(true);
          if (!batch.export_progress || !batch.export_progress.running) {
            stopPoll();
          }
        } catch (_e) {
          stopPoll();
        }
      }, 1200);
    }
  }

  async function openBatch(batchId, initialJobId = null) {
    stopPoll();
    batch = await request(`/api/batches/${encodeURIComponent(batchId)}`);
    document.body.classList.add('batch-mode');
    shell.hidden = false;
    if (canvasWorkbenchEl && canvasWorkbenchEl.parentNode !== mainSlot) {
      mainSlot.appendChild(canvasWorkbenchEl);
    }
    renderHeader();
    hydrateExportForm();
    renderSidebar();

    const items = batch.items || [];
    let chosen = null;
    if (initialJobId) {
      chosen = items.find((it) => it.job_id === initialJobId);
    }
    if (!chosen) {
      chosen = items.find((it) => !it.locked && it.job_id) || items[0] || null;
    }
    if (chosen) {
      await selectItem(chosen.item_id);
    } else {
      renderReviewBar();
    }
    if (batch.export_progress && batch.export_progress.running) {
      await refreshBatch(true);
    }
  }

  function closeBatch() {
    stopPoll();
    batch = null;
    activeItemId = null;
    document.body.classList.remove('batch-mode');
    shell.hidden = true;
    if (canvasWorkbenchEl && originalParent && canvasWorkbenchEl.parentNode !== originalParent) {
      originalParent.insertBefore(canvasWorkbenchEl, originalNext);
    }
  }

  // Bind UI events once
  for (const btn of document.querySelectorAll('.bw-filter-btn')) {
    btn.onclick = () => {
      filterKey = btn.dataset.filter || 'all';
      for (const b of document.querySelectorAll('.bw-filter-btn')) {
        b.classList.toggle('active', b === btn);
      }
      renderSidebar();
    };
  }

  $('bwToggleExportPanel').onclick = () => {
    const panel = $('bwExportPanel');
    panel.hidden = !panel.hidden;
  };

  $('bwSizePreset').onchange = () => {
    $('bwCustomSizeWrap').hidden = $('bwSizePreset').value !== 'custom';
  };

  $('bwSaveProfile').onclick = async () => {
    if (!batch) return;
    try {
      busyAction = true;
      renderExportStatus();
      batch = await request(`/api/batches/${encodeURIComponent(batch.batch_id)}/profile`, {
        method: 'POST',
        body: JSON.stringify(readExportForm()),
      });
      renderHeader();
      hydrateExportForm();
      $('bwExportStatus').textContent = '统一导出规格已保存（后续导出自动沿用）。';
    } catch (err) {
      $('bwExportStatus').textContent = `保存规格失败：${err.message}`;
    } finally {
      busyAction = false;
      renderExportStatus();
    }
  };

  $('bwRunExport').onclick = async () => {
    if (!batch) return;
    try {
      busyAction = true;
      renderExportStatus();
      const prof = readExportForm();
      await request(`/api/batches/${encodeURIComponent(batch.batch_id)}/profile`, {
        method: 'POST',
        body: JSON.stringify(prof),
      });
      await request(`/api/batches/${encodeURIComponent(batch.batch_id)}/export`, {
        method: 'POST',
        body: JSON.stringify({ async: true, export_profile: prof, output_dir: prof.output_dir }),
      });
      await refreshBatch(true);
    } catch (err) {
      $('bwExportStatus').textContent = `启动批量导出失败：${err.message}`;
    } finally {
      busyAction = false;
      renderExportStatus();
    }
  };

  $('bwBtnLock').onclick = async () => {
    const item = currentItem();
    if (!batch || !item) return;
    try {
      busyAction = true;
      renderReviewBar();
      batch = await request(
        `/api/batches/${encodeURIComponent(batch.batch_id)}/items/${encodeURIComponent(item.item_id)}/lock`,
        { method: 'POST', body: JSON.stringify({ reviewer: '批量验收使用者' }) }
      );
      renderHeader();
      renderSidebar();
      renderReviewBar();
      if (item.job_id) {
        const job = await request(`/api/jobs/${encodeURIComponent(item.job_id)}`);
        await onSelectJob(job);
      }
    } catch (err) {
      $('bwActiveSummary').textContent = `锁定失败：${err.message}`;
    } finally {
      busyAction = false;
      renderReviewBar();
      renderExportStatus();
    }
  };

  $('bwBtnUnlock').onclick = async () => {
    const item = currentItem();
    if (!batch || !item) return;
    try {
      busyAction = true;
      renderReviewBar();
      batch = await request(
        `/api/batches/${encodeURIComponent(batch.batch_id)}/items/${encodeURIComponent(item.item_id)}/unlock`,
        { method: 'POST', body: '{}' }
      );
      renderHeader();
      renderSidebar();
      renderReviewBar();
      if (item.job_id) {
        const job = await request(`/api/jobs/${encodeURIComponent(item.job_id)}`);
        await onSelectJob(job);
      }
    } catch (err) {
      $('bwActiveSummary').textContent = `解锁失败：${err.message}`;
    } finally {
      busyAction = false;
      renderReviewBar();
      renderExportStatus();
    }
  };

  $('bwBtnReject').onclick = async () => {
    const item = currentItem();
    if (!batch || !item) return;
    const note = $('bwFixNoteInput').value.trim();
    if (!note) {
      $('bwActiveSummary').textContent = '请先输入一句修改说明（例如：右脚底有灰边 / 后衣摆露底），再点打回待修。';
      $('bwFixNoteInput').focus();
      return;
    }
    try {
      busyAction = true;
      renderReviewBar();
      batch = await request(
        `/api/batches/${encodeURIComponent(batch.batch_id)}/items/${encodeURIComponent(item.item_id)}/reject`,
        { method: 'POST', body: JSON.stringify({ fix_note: note }) }
      );
      renderHeader();
      renderSidebar();
      renderReviewBar();
    } catch (err) {
      $('bwActiveSummary').textContent = `打回失败：${err.message}`;
    } finally {
      busyAction = false;
      renderReviewBar();
      renderExportStatus();
    }
  };

  $('bwBtnNext').onclick = jumpNextUnreviewed;
  $('bwBtnRefresh').onclick = () => refreshBatch(true);

  return {
    openBatch,
    closeBatch,
    isActive: () => !!batch,
    refreshBatch,
  };
}
