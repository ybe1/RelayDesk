(() => {
  const $ = id => document.getElementById(id);
  const value = item => item === null || item === undefined ? '读数失败' : String(item);
  function alert(message) { const node = $('admin-notice'); node.textContent = message; node.className = 'notice show error'; }
  function job(item) { const box = document.createElement('div'); box.className = 'job'; const top = document.createElement('div'); top.className = 'job-top'; const title = document.createElement('span'); title.className = 'job-title'; title.textContent = item.name; const state = document.createElement('span'); state.className = `status-chip ${item.state === 'succeeded' ? 'ok' : item.state === 'failed' ? 'bad' : item.state === 'running' ? 'warn' : ''}`; state.textContent = item.state; top.append(title, state); const log = document.createElement('pre'); log.className = 'log'; log.textContent = item.log_tail || '尚无日志。'; box.append(top, log); return box; }
  async function refresh() {
    try {
      const response = await fetch('/api/admin/overview'); const data = await response.json(); if (!response.ok) throw new Error('读取后台概览失败'); const rag = data.rag_eval; const best = rag?.best_mrr; $('a-rag-eval').textContent = rag ? `报告：${rag.report_status} · 最佳 MRR：${best ? `${best.strategy} ${best.mrr}` : '暂无'}` : '评估报告暂不可用'; const kb = data.knowledge; if (!kb) throw new Error('知识库概览暂不可用');
      $('a-total').textContent = value(kb.total); $('a-pending').textContent = value(kb.pending); $('a-done').textContent = value(kb.done); $('a-milvus').textContent = value(kb.milvus_count); $('a-consistent').textContent = kb.consistent === true ? '计数一致' : kb.consistent === false ? '计数不一致' : '读数失败'; $('a-deps').textContent = `MySQL ${kb.dependencies?.mysql === 'available' ? '可用' : '读数失败'} · Milvus ${kb.dependencies?.milvus === 'available' ? '可用' : '读数失败'}`;
      const config = kb.config || {}; $('a-top-k').textContent = value(config.rag_top_k); $('a-min-score').textContent = value(config.rag_min_score); $('a-endpoint').textContent = [config.embed_model, config.milvus_endpoint].filter(Boolean).join(' · ') || '读数失败';
      const target = $('admin-jobs'); target.replaceChildren(); if (!data.jobs?.length) { target.textContent = '作业状态暂不可用。'; target.className = 'empty'; } else data.jobs.forEach(item => target.append(job(item)));
    } catch (error) { ['a-total','a-pending','a-done','a-milvus','a-consistent','a-deps','a-top-k','a-min-score','a-endpoint'].forEach(id => $(id).textContent = '读数失败'); alert(error.message || '读取后台概览失败'); }
  }
  document.addEventListener('DOMContentLoaded', () => { if (window.mountAdminNav) window.mountAdminNav('/admin'); $('admin-refresh').addEventListener('click', refresh); refresh(); });
})();
