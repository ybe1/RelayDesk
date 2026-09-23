(() => {
  function mountAdminNav(active) {
    const target = document.getElementById('app-nav');
    if (!target) return;
    const links = [['/', '客服聊天'], ['/kb', '知识库'], ['/admin', '后台管理']];
    target.replaceChildren(...links.map(([href, label]) => {
      const link = document.createElement('a');
      link.href = href;
      link.textContent = label;
      if (href === active) link.setAttribute('aria-current', 'page');
      return link;
    }));
  }
  window.mountAdminNav = mountAdminNav;
})();
