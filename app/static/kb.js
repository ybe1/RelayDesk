(() => {
  const $ = (id) => document.getElementById(id);
  const state = { jobs: new Map(), logCache: new Map(), poller: null };
  const jobTargets = { 'kb-build': 'build-job', 'kb-preview': 'build-job', 'seed-conv': 'mine-job', 'kb-mine': 'mine-job', 'kb-vectorize': 'vector-job', 'kb-reset': 'vector-job' };
  const jobLabels = { 'kb-preview':'材料预览', 'kb-build':'建库导入', 'kb-mine':'对话挖知识', 'kb-vectorize':'补向量', 'seed-conv':'演示会话', 'kb-reset':'重置知识库' };

  function text(value) { return value === null || value === undefined ? '读数失败' : String(value); }
  function clear(node) { node.replaceChildren(); return node; }
  function element(name, className, content) { const el = document.createElement(name); if (className) el.className = className; if (content !== undefined) el.textContent = content; return el; }
  function notice(message, kind = 'error') { const node = $('notice'); node.textContent = message; node.className = 'notice show ' + kind; window.clearTimeout(notice.timer); notice.timer = window.setTimeout(() => node.className = 'notice', 8000); }
  function errorText(payload, fallback) { if (!payload) return fallback; const detail = payload.detail ?? payload.message ?? payload; if (typeof detail === 'string') return detail; if (detail && typeof detail === 'object') return [detail.message, `已提交 ${detail.inserted ?? 0} 条，跳过 ${detail.skipped ?? 0} 条，待补向量 ${detail.pending ?? 0} 条`].filter(Boolean).join('；'); return fallback; }
  async function request(path, options = {}) {
    let response;
    try { response = await fetch(path, options); } catch { throw new Error('无法连接服务，请确认应用正在运行。'); }
    let data = null; try { data = await response.json(); } catch { /* non-JSON errors still receive a visible message */ }
    if (!response.ok) { const error = new Error(errorText(data, `请求失败（${response.status}）`)); error.payload = data?.detail ?? data; throw error; }
    return data;
  }
  function setBusy(button, busy) { button.disabled = busy; button.dataset.original = button.dataset.original || button.textContent; button.textContent = busy ? '处理中…' : button.dataset.original; }
  function rememberJob(job) { state.logCache.set(job.name, job.log_tail || state.logCache.get(job.name) || ''); state.jobs.set(job.name, job); }
  function jobChip(job) { const stateName = job.state || 'stopped'; const tone = stateName === 'succeeded' ? 'ok' : stateName === 'failed' ? 'bad' : stateName === 'running' ? 'warn' : ''; return element('span', `status-chip ${tone}`, stateName); }
  function renderJob(job) {
    const target = $(jobTargets[job.name]); if (!target) return; clear(target);
    const box = element('div', 'job'); const top = element('div', 'job-top'); top.append(element('span', 'job-title', jobLabels[job.name] || job.name), jobChip(job)); box.append(top);
    const actions = element('div', 'job-actions');
    if (job.state === 'running') { const stop = element('button', 'danger', '停止作业'); stop.addEventListener('click', () => stopJob(job.name, stop)); actions.append(stop); }
    box.append(actions);
    const log = element('pre', 'log', state.logCache.get(job.name) || '尚无日志。'); box.append(log); target.append(box);
  }
  function renderJobs() { state.jobs.forEach(renderJob); }
  function hasRunningJobs() { return [...state.jobs.values()].some(job => job.state === 'running'); }
  function syncJobControls() {
    const busy = hasRunningJobs();
    document.querySelectorAll('[data-start-job], #reset-job').forEach(button => {
      button.disabled = busy;
      button.title = busy ? '等待正在运行的维护作业结束后再操作' : '';
    });
  }
  function schedulePolling() { window.clearTimeout(state.poller); if (hasRunningJobs()) state.poller = window.setTimeout(refreshJobs, 1500); }
  async function refreshJobs() {
    try { (await request('/api/jobs')).forEach(rememberJob); renderJobs(); syncJobControls(); if (!hasRunningJobs()) await refreshOverview(); } catch (error) { notice(error.message); }
    schedulePolling();
  }
  async function startJob(name, button, extra = {}) {
    setBusy(button, true);
    try { rememberJob(await request(`/api/jobs/${encodeURIComponent(name)}`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(extra) })); renderJobs(); syncJobControls(); schedulePolling(); notice(`${jobLabels[name] || name} 已启动。`, 'success'); }
    catch (error) { notice(error.message); }
    finally { setBusy(button, false); syncJobControls(); }
  }
  async function stopJob(name, button) { setBusy(button, true); try { rememberJob(await request(`/api/jobs/${encodeURIComponent(name)}/stop`, {method:'POST'})); renderJobs(); syncJobControls(); schedulePolling(); } catch (error) { notice(error.message); } finally { setBusy(button, false); syncJobControls(); } }
  function metric(id, value) { $(id).textContent = text(value); }
  function status(value) { return value === true ? '计数一致' : value === false ? '计数不一致' : '读数失败'; }
  async function refreshOverview() {
    try { const data = await request('/api/kb/overview'); metric('m-total', data.total); metric('m-pending', data.pending); metric('m-done', data.done); metric('m-key', data.key_clause_count); metric('m-milvus', data.milvus_count); metric('m-consistent', status(data.consistent)); $('m-deps').textContent = `MySQL：${data.dependencies?.mysql === 'available' ? '可用' : '读数失败'} · Milvus：${data.dependencies?.milvus === 'available' ? '可用' : '读数失败'}`; const counts = data.staging || {}; $('staging-counts').replaceChildren(...['extracted','kept','discarded'].map(key => element('span','tag', `${key}：${text(counts[key])}`))); }
    catch (error) { ['m-total','m-pending','m-done','m-key','m-milvus','m-consistent'].forEach(id => metric(id, null)); $('m-deps').textContent = '依赖读数失败'; notice(error.message); }
  }
  function feature(value, label) { return element('span', 'tag', `${label}：${value ? '已触发' : '未触发'}`); }
  function renderPreview(data) { const features = data.features || {}; const heading = $('preview-features'); heading.replaceChildren(feature(features.heading_split, '标题切分'), feature(features.overlap, '重叠'), feature(features.table_header_copied, '表格表头复制')); const out = clear($('manual-preview-result')); const chunks = data.chunks || []; if (!chunks.length) { out.append(element('p','empty','没有可展示的知识块。')); return; } chunks.forEach((chunk, index) => { const card = element('article','preview-item'); card.append(element('h3','', `块 ${index + 1} · ${chunk.category || '未分类'}`)); card.append(element('div','muted', `路径：${chunk.section_path || '—'} · ${chunk.characters} 字`)); const tags = element('div','feature-list'); if (chunk.is_key_clause) tags.append(element('span','tag','关键条款')); if (chunk.is_table) tags.append(element('span','tag','表格块')); if (chunk.duplicate) tags.append(element('span','tag','库中可能重复')); card.append(tags); card.append(element('p','', `问法：${chunk.questions}\n\n${chunk.answer}`)); out.append(card); }); }
  function manualPayload() { return { markdown: $('manual-markdown').value, content_type: $('manual-type').value, source_name: $('manual-name').value }; }
  async function previewManual(button) { setBusy(button, true); try { renderPreview(await request('/api/kb/preview', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(manualPayload()) })); } catch (error) { notice(error.message); } finally { setBusy(button, false); } }
  async function ingestManual(button) { setBusy(button, true); try { const result = await request('/api/kb/ingest', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({...manualPayload(), vectorize:$('manual-vectorize').checked}) }); notice(`录入完成：新增 ${result.inserted} 条，跳过 ${result.skipped} 条，已向量化 ${result.indexed} 条，待补 ${result.pending} 条。`, 'success'); await refreshAll(); } catch (error) { const partial = error.payload; if (partial && typeof partial === 'object' && partial.pending !== undefined) notice(`原文已入库，向量待补。新增 ${partial.inserted ?? 0} 条，跳过 ${partial.skipped ?? 0} 条，待补 ${partial.pending} 条。`, 'error'); else notice(error.message); await refreshAll(); } finally { setBusy(button, false); } }
  async function loadDocuments() { try { const docs = await request('/api/kb/documents'); const target = clear($('documents')); docs.forEach(doc => { const row = element('div','job'); const top = element('div','job-top'); top.append(element('span','job-title',doc.name), element('span','tag',doc.content_type)); row.append(top); row.append(element('p','muted', `${doc.chunks} 块 · ${doc.features?.table ? '含表格' : '无表格'} · ${doc.features?.key_clause ? '含关键条款' : '无关键条款'}`)); const preview = element('button','secondary','预览此材料'); preview.addEventListener('click', async () => { setBusy(preview,true); try { renderPreview(await request('/api/kb/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({document:doc.name})})); document.getElementById('manual-preview-result').scrollIntoView({behavior:'smooth',block:'start'}); } catch(error) { notice(error.message); } finally { setBusy(preview,false); } }); const actions = element('div','job-actions'); actions.append(preview); row.append(actions); target.append(row); }); } catch (error) { $('documents').replaceChildren(element('p','empty',error.message)); } }
  async function loadStaging() { try { const rows = await request('/api/kb/staging?limit=20'); const body = clear($('staging-body')); if (!rows.length) { body.append(row(['暂无暂存记录。'], 3)); return; } rows.forEach(item => body.append(row([`${item.id}\n${item.source_ref || '—'}`, `${item.question}\n${item.answer}`, item.status]))); } catch (error) { $('staging-body').replaceChildren(row([error.message],3)); } }
  async function loadRecent() { try { const rows = await request('/api/kb/recent?limit=20'); const body = clear($('recent-body')); if (!rows.length) { body.append(row(['暂无知识记录。'],4)); return; } rows.forEach(item => body.append(row([item.id, `${item.category}\n${item.questions}`, item.section_path || '—', `${item.vectorize_status}${item.is_key_clause ? ' · 关键条款' : ''}`]))); } catch (error) { $('recent-body').replaceChildren(row([error.message],4)); } }
  function row(values, span) { const tr = document.createElement('tr'); if (span) { const td = element('td','empty',values[0]); td.colSpan=span; tr.append(td); } else values.forEach(value => tr.append(element('td','',value))); return tr; }
  async function search(button) { setBusy(button,true); try { const data = await request('/api/kb/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:$('search-query').value,top_k:Number($('search-top-k').value)})}); const target=clear($('search-results')); if (!(data.hits || []).length) target.append(element('p','empty','没有达到当前阈值的结果。')); else data.hits.forEach(hit => { const item=element('article','hit'); item.append(element('strong','',`${hit.question} · ${(Number(hit.score) || 0).toFixed(3)}`)); item.append(element('p','muted',`${hit.category}\n${hit.answer}`)); target.append(item); }); } catch(error) { $('search-results').replaceChildren(element('p','empty',error.message)); } finally { setBusy(button,false); } }
  async function reset(button) { if (!window.confirm('将只清除第三章知识表和 knowledge 向量集合。确认继续吗？')) return; if (!window.confirm('请再次确认：此操作会移除 Ch03 知识库内容。')) return; await startJob('kb-reset', button, {confirmation:'RESET_CH03'}); }
  async function refreshAll() { await Promise.all([refreshOverview(),loadDocuments(),loadStaging(),loadRecent(),refreshJobs()]); }
  function setup() { if (window.mountAdminNav) window.mountAdminNav('/kb'); $('refresh-all').addEventListener('click', refreshAll); $('manual-preview').addEventListener('click', event => previewManual(event.currentTarget)); $('manual-ingest').addEventListener('click', event => ingestManual(event.currentTarget)); $('load-staging').addEventListener('click', loadStaging); $('search-btn').addEventListener('click', event => search(event.currentTarget)); $('reset-job').addEventListener('click', event => reset(event.currentTarget)); document.querySelectorAll('[data-start-job]').forEach(button => button.addEventListener('click', () => startJob(button.dataset.startJob,button))); document.querySelectorAll('.preset-query').forEach(button => button.addEventListener('click', () => { $('search-query').value=button.dataset.query; search($('search-btn')); })); refreshAll(); }
  document.addEventListener('DOMContentLoaded', setup);
})();
