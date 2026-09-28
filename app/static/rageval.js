(() => {
  const $ = id => document.getElementById(id);
  const make = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  };
  const ratio = value => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
  const count = value => typeof value === 'number' && Number.isFinite(value) ? String(value) : '—';
  const strategies = ['vector', 'bm25', 'hybrid', 'hybrid_rerank'];
  const names = { vector: 'Vector', bm25: 'BM25', hybrid: 'Hybrid', hybrid_rerank: 'Hybrid + Rerank' };
  const bucketNames = { A_policy: 'A 政策', B_model: 'B 型号', C_colloquial: 'C 口语', D_out_of_scope: 'D 越界', E_multi: 'E 多要点' };
  let currentReport = null;
  let ledgerPage = 1;
  let ledgerTotal = 0;
  let jobTimer = null;

  function notice(message, error = false) {
    const node = $('eval-notice');
    node.textContent = message;
    node.className = message ? `notice show ${error ? 'error' : 'success'}` : 'notice';
  }
  async function jsonRequest(url, options) {
    const response = await fetch(url, options);
    let data;
    try { data = await response.json(); } catch { data = null; }
    if (!response.ok) throw Object.assign(new Error(data?.detail || `请求失败（${response.status}）`), { status: response.status });
    return data;
  }
  function row(cells) {
    const tr = document.createElement('tr');
    for (const cell of cells) tr.append(make('td', cell));
    return tr;
  }
  function empty(target, message, columns) {
    const tr = document.createElement('tr');
    const td = make('td', message, 'empty');
    td.colSpan = columns;
    tr.append(td);
    target.replaceChildren(tr);
  }
  function bar(label, value, secondary = false) {
    const row = make('div', undefined, 'eval-bar-row');
    const title = make('span', label);
    const track = make('div', undefined, 'eval-bar-track');
    const fill = make('div', undefined, `eval-bar-fill${secondary ? ' secondary' : ''}`);
    fill.style.width = typeof value === 'number' && Number.isFinite(value) ? `${Math.max(0, Math.min(100, value * 100))}%` : '0%';
    track.append(fill);
    row.append(title, track, make('span', ratio(value), `score${secondary ? ' secondary' : ''}`));
    return row;
  }
  function renderBuckets() {
    const target = $('bucket-chart');
    target.replaceChildren();
    const selected = $('bucket-strategy').value;
    const buckets = currentReport?.retrieval?.[selected]?.buckets;
    if (!buckets || !Object.keys(buckets).length) { target.append(make('p', '暂无分桶数据。', 'empty')); return; }
    for (const [key, values] of Object.entries(buckets)) {
      target.append(bar(bucketNames[key] || key, values?.excluded ? null : values?.mrr));
    }
  }
  function renderReport(data) {
    const report = data?.report && typeof data.report === 'object' ? data.report : null;
    currentReport = report;
    const meta = report?.meta || {};
    const generation = report?.generation || {};
    const faith = generation.faithfulness || {};
    const statusText = { complete: '报告已完成', partial: '部分报告', missing: '报告尚不存在', corrupt: '报告无法读取' }[data?.report_status] || '报告状态未知';
    const chips = [statusText];
    if (meta.case_count !== undefined) chips.push(`题目 ${count(meta.case_count)}`);
    if (meta.retrieval_status) chips.push(`检索 ${meta.retrieval_status}`);
    if (meta.updated_at) chips.push(`更新 ${meta.updated_at}`);
    if (meta.dataset_sha256) chips.push(`数据集 ${String(meta.dataset_sha256).slice(0, 12)}…`);
    $('report-meta').replaceChildren(...chips.map((value, index) => make('span', value, `status-chip${index === 0 && data.report_status === 'complete' ? ' ok' : ''}`)));
    $('report-message').textContent = report ?
      `Top-K ${count(meta.top_k)} · 裁判模型 ${meta.judge_model || '未提供'}。指标只显示报告中已保存的值；跳过的调用不进入均值分母。` :
      '暂无可用指标。编造台账和作业控制仍可使用。';
    const best = data?.best_mrr;
    const strategy = best?.strategy;
    $('kpi-mrr').textContent = ratio(best?.mrr);
    $('kpi-mrr-hint').textContent = strategy ? names[strategy] || strategy : '暂无有效策略';
    $('kpi-recall').textContent = ratio(report?.retrieval?.[strategy]?.overall?.recall_at_5);
    $('kpi-evidence').textContent = ratio(report?.evidence_coverage?.[strategy]?.overall?.coverage);
    $('kpi-faith').textContent = ratio(faith.faithfulness_rate);
    $('kpi-faith-hint').textContent = `已裁判 ${count(faith.completed)} / ${count(faith.case_count)}`;
    const available = strategies.filter(key => report?.retrieval?.[key] || report?.generation?.strategies?.[key]);
    for (const key of Object.keys(report?.retrieval || {})) if (!available.includes(key)) available.push(key);
    const chart = $('strategy-chart');
    chart.replaceChildren();
    if (!available.length) chart.append(make('p', '暂无策略数据。', 'empty'));
    else for (const key of available) chart.append(bar(names[key] || key, report?.retrieval?.[key]?.overall?.mrr, key === strategy));
    const chooser = $('bucket-strategy');
    const previous = chooser.value;
    chooser.replaceChildren(...available.map(key => {
      const option = make('option', names[key] || key);
      option.value = key;
      return option;
    }));
    if (available.includes(previous)) chooser.value = previous;
    else if (available.includes(strategy)) chooser.value = strategy;
    renderBuckets();
    const retrievalBody = $('retrieval-body');
    if (!available.length) empty(retrievalBody, '暂无检索汇总。', 5);
    else retrievalBody.replaceChildren(...available.map(key => {
      const overall = report?.retrieval?.[key]?.overall || {};
      return row([names[key] || key, `${count(overall.completed)} / ${count(overall.case_count)}`, ratio(overall.recall_at_5), ratio(overall.mrr), ratio(report?.evidence_coverage?.[key]?.overall?.coverage)]);
    }));
    $('generation-summary').textContent = `生成状态：${generation.status || '未开始'} · 裁判 ${count(faith.completed)} / ${count(faith.case_count)} · 跳过 ${count(faith.skipped)}。`;
    const generationBody = $('generation-body');
    if (!available.length) empty(generationBody, '暂无生成汇总。', 5);
    else generationBody.replaceChildren(...available.map(key => {
      const item = generation.strategies?.[key] || {};
      const overall = item.overall || {};
      return row([names[key] || key, `${count(overall.completed)} / ${count(overall.case_count)}`, count(overall.skipped), ratio(overall.answer_point_coverage), ratio(item.buckets?.D_out_of_scope?.refusal_rate)]);
    }));
  }
  async function loadReport() {
    $('eval-refresh').disabled = true;
    try {
      renderReport(await jsonRequest('/api/rag-eval/overview'));
      notice('');
    } catch (error) {
      renderReport({ report_status: 'missing', report: null });
      notice(`报告读取失败：${error.message}`, true);
    } finally { $('eval-refresh').disabled = false; }
  }

  function ledgerRow(item) {
    const tr = document.createElement('tr');
    const identity = make('td', `${item.eval_id || item.id}\n${bucketNames[item.bucket] || item.bucket || '—'} · ${names[item.strategy] || item.strategy || '—'}`);
    const answer = make('td', `${item.query || '—'}\n\n回答：${item.answer || '—'}`);
    const reason = make('td', item.reason || '—');
    if (Array.isArray(item.citations) && item.citations.length) {
      const details = document.createElement('details');
      details.append(make('summary', `查看 ${item.citations.length} 条证据`));
      const evidence = make('pre', JSON.stringify(item.citations, null, 2), 'log');
      evidence.style.maxHeight = '220px';
      details.append(evidence);
      reason.append(details);
    }
    const status = make('td', `${item.status || '—'}${item.recurrence ? ' · 复发' : ''}\n累计 ${count(item.seen_count)} 次\n${item.resolution || ''}`);
    const action = make('td', undefined, 'case-actions');
    const select = document.createElement('select');
    select.setAttribute('aria-label', `${item.eval_id || item.id} 处置状态`);
    for (const value of ['未解决', '已解决', '无需解决']) {
      const option = make('option', value); option.value = value; select.append(option);
    }
    select.value = item.status;
    const resolution = document.createElement('input');
    resolution.type = 'text'; resolution.maxLength = 300;
    resolution.placeholder = '处置说明（解决时必填）';
    resolution.setAttribute('aria-label', `${item.eval_id || item.id} 处置说明`);
    resolution.value = item.resolution || '';
    const save = make('button', '保存处置');
    save.type = 'button';
    save.addEventListener('click', async () => {
      const text = resolution.value.trim();
      if (select.value !== '未解决' && !text) { $('faith-notice').textContent = '标记已解决或无需解决时，请填写处置说明。'; resolution.focus(); return; }
      save.disabled = true;
      $('faith-notice').textContent = '正在保存…';
      try {
        await jsonRequest(`/api/rag-eval/faith-cases/${encodeURIComponent(item.id)}/status`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: select.value, resolution: select.value === '未解决' ? null : text, expected_seen_count: item.seen_count })
        });
        $('faith-notice').textContent = '处置已保存。';
        await loadLedger(true);
      } catch (error) {
        $('faith-notice').textContent = error.status === 409 ? '个案已被更新，请刷新台账后重新处置。' : `保存失败：${error.message}`;
        save.disabled = false;
      }
    });
    action.append(select, resolution, save);
    tr.append(identity, answer, reason, status, action);
    return tr;
  }
  async function loadLedger(preserveNotice = false) {
    if (!preserveNotice) $('faith-notice').textContent = '正在读取台账…';
    const query = new URLSearchParams({ page: String(ledgerPage), size: '20' });
    if ($('faith-status').value) query.set('status', $('faith-status').value);
    try {
      const data = await jsonRequest(`/api/rag-eval/faith-cases?${query}`);
      ledgerTotal = Number(data.total) || 0;
      const items = Array.isArray(data.items) ? data.items : [];
      if (items.length) $('faith-body').replaceChildren(...items.map(ledgerRow));
      else empty($('faith-body'), '当前筛选条件下没有个案。', 5);
      $('faith-page').textContent = `第 ${ledgerPage} 页 · 共 ${ledgerTotal} 条`;
      $('faith-prev').disabled = ledgerPage <= 1;
      $('faith-next').disabled = ledgerPage * 20 >= ledgerTotal;
      if (!preserveNotice) $('faith-notice').textContent = '';
    } catch (error) {
      empty($('faith-body'), '台账暂不可用。', 5);
      $('faith-notice').textContent = `台账读取失败：${error.message}`;
      $('faith-prev').disabled = true;
      $('faith-next').disabled = true;
    }
  }
  function renderJob(job) {
    const state = job?.state || '未知';
    $('job-state').textContent = `eval-rag · ${state}${job?.exit_code == null ? '' : ` · 退出码 ${job.exit_code}`}`;
    $('job-log').textContent = job?.log_tail || '尚无日志。';
    $('eval-start').disabled = state === 'running';
    $('eval-stop').disabled = state !== 'running';
    if (jobTimer) clearTimeout(jobTimer);
    if (state === 'running') jobTimer = setTimeout(loadJob, 4000);
  }
  async function loadJob() {
    try { renderJob(await jsonRequest('/api/jobs/eval-rag?limit=200')); }
    catch (error) { $('job-state').textContent = `作业状态读取失败：${error.message}`; }
  }
  async function actJob(action) {
    if (action === 'start' && !window.confirm('启动完整 RAG 评估？该作业可能调用已配置的模型服务并更新评估报告。')) return;
    const button = action === 'start' ? $('eval-start') : $('eval-stop');
    button.disabled = true;
    try {
      const url = action === 'start' ? '/api/jobs/eval-rag' : '/api/jobs/eval-rag/stop';
      const options = { method: 'POST' };
      if (action === 'start') {
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify({ confirmation: 'RUN_EVAL_RAG' });
      }
      renderJob(await jsonRequest(url, options));
      notice(action === 'start' ? '评估任务已启动。' : '已请求停止评估。');
    } catch (error) { notice(`作业操作失败：${error.message}`, true); await loadJob(); }
  }
  document.addEventListener('DOMContentLoaded', () => {
    window.mountAdminNav?.('/rag-eval');
    $('eval-refresh').addEventListener('click', loadReport);
    $('bucket-strategy').addEventListener('change', renderBuckets);
    $('faith-status').addEventListener('change', () => { ledgerPage = 1; loadLedger(); });
    $('faith-refresh').addEventListener('click', () => loadLedger());
    $('faith-prev').addEventListener('click', () => { if (ledgerPage > 1) { ledgerPage--; loadLedger(); } });
    $('faith-next').addEventListener('click', () => { if (ledgerPage * 20 < ledgerTotal) { ledgerPage++; loadLedger(); } });
    $('eval-start').addEventListener('click', () => actJob('start'));
    $('eval-stop').addEventListener('click', () => actJob('stop'));
    $('job-refresh').addEventListener('click', loadJob);
    loadReport(); loadLedger(); loadJob();
  });
})();
