/**
 * ArxivMiner - Modern Frontend Application
 * ==========================================
 */

class ArxivMinerApp {
    constructor() {
        this.apiBase = '/api';
        this.state = {
            page: 1,
            pageSize: 20,
            q: '',
            tag: '',
            category: '',
            sort: 'published',
            hideHidden: true,
            papers: [],
            total: 0,
            totalPages: 0,
            currentPaper: null,
            tags: [],
            isLoading: false
        };
        
        this.debounceTimer = null;
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadTags();
        this.loadPapers();
        this.setupRouting();
        this.loadPreferences();
    }
    
    // Event Bindings
    bindEvents() {
        // Search with debounce
        const searchInput = document.getElementById('search-input');
        searchInput?.addEventListener('input', (e) => {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = setTimeout(() => {
                this.state.q = e.target.value;
                this.state.page = 1;
                this.loadPapers();
            }, 300);
        });
        
        searchInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                clearTimeout(this.debounceTimer);
                this.state.q = e.target.value;
                this.state.page = 1;
                this.loadPapers();
            }
        });
        
        document.getElementById('search-btn')?.addEventListener('click', () => {
            this.state.q = searchInput?.value || '';
            this.state.page = 1;
            this.loadPapers();
        });
        
        // Filters
        document.getElementById('category-filter')?.addEventListener('change', (e) => {
            this.state.category = e.target.value;
            this.state.page = 1;
            this.loadPapers();
            this.savePreferences();
        });
        
        document.getElementById('sort-filter')?.addEventListener('change', (e) => {
            this.state.sort = e.target.value;
            this.state.page = 1;
            this.loadPapers();
            this.savePreferences();
        });
        
        document.getElementById('hide-hidden')?.addEventListener('change', (e) => {
            this.state.hideHidden = e.target.checked;
            this.state.page = 1;
            this.loadPapers();
            this.savePreferences();
        });
        
        // Refresh
        document.getElementById('refresh-btn')?.addEventListener('click', () => {
            this.loadPapers();
        });
        
        // Pagination
        document.getElementById('prev-page')?.addEventListener('click', () => {
            if (this.state.page > 1) {
                this.state.page--;
                this.loadPapers();
                this.scrollToTop();
            }
        });
        
        document.getElementById('next-page')?.addEventListener('click', () => {
            if (this.state.page < this.state.totalPages) {
                this.state.page++;
                this.loadPapers();
                this.scrollToTop();
            }
        });
        
        // Back button
        document.getElementById('back-btn')?.addEventListener('click', () => {
            this.showPage('list');
            window.location.hash = '/';
        });
        
        // Admin buttons
        document.getElementById('fetch-btn')?.addEventListener('click', () => {
            this.adminAction('/api/admin/fetch', '抓取完成', '正在抓取论文...');
        });
        
        document.getElementById('retranslate-btn')?.addEventListener('click', () => {
            this.adminAction('/api/admin/retranslate?status=failed', '翻译重试完成', '正在重试翻译...');
        });
        
        document.getElementById('retag-btn')?.addEventListener('click', () => {
            this.adminAction('/api/admin/retag?status=failed', '标记重试完成', '正在重试标记...');
        });
        
        document.getElementById('process-pending-btn')?.addEventListener('click', () => {
            this.adminAction('/api/admin/process-pending', '处理完成', '正在处理待处理项...');
        });
        
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = e.target.dataset.page;
                this.showPage(page === 'list' ? 'list' : page);
                this.updateNav(page);
            });
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (window.location.hash.startsWith('#/paper/')) {
                    this.showPage('list');
                    window.location.hash = '/';
                }
            }
            if (e.key === 'ArrowLeft' && this.state.page > 1 && window.location.hash === '#/') {
                this.state.page--;
                this.loadPapers();
            }
            if (e.key === 'ArrowRight' && this.state.page < this.state.totalPages && window.location.hash === '#/') {
                this.state.page++;
                this.loadPapers();
            }
        });
    }
    
    // Routing
    setupRouting() {
        window.addEventListener('hashchange', () => this.handleRoute());
        this.handleRoute();
    }
    
    handleRoute() {
        const hash = window.location.hash || '#/';
        
        if (hash.startsWith('#/paper/')) {
            const arxivId = hash.split('/')[2];
            this.showPaperDetail(arxivId);
        } else if (hash === '#/admin') {
            this.showPage('admin');
            this.updateNav('admin');
            this.loadAdminStats();
        } else {
            this.showPage('list');
            this.updateNav('list');
        }
    }
    
    updateNav(page) {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
            if (link.dataset.page === page) {
                link.classList.add('active');
            }
        });
    }
    
    showPage(pageName) {
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
        });
        document.getElementById(`page-${pageName}`)?.classList.add('active');
        
        if (pageName === 'list') {
            document.querySelector('.filter-bar')?.classList.remove('hidden');
        } else {
            document.querySelector('.filter-bar')?.classList.add('hidden');
        }
    }
    
    // API Calls
    async loadTags() {
        try {
            const res = await fetch(`${this.apiBase}/tags?limit=50`);
            const data = await res.json();
            this.state.tags = data;
            this.renderTagCloud();
        } catch (err) {
            console.error('Failed to load tags:', err);
        }
    }
    
    renderTagCloud() {
        const container = document.getElementById('tag-cloud');
        if (!container) return;
        
        container.innerHTML = '';
        
        this.state.tags.forEach(tag => {
            const el = document.createElement('span');
            el.className = 'tag' + (this.state.tag === tag.name ? ' active' : '');
            el.innerHTML = `${this.escapeHtml(tag.name)} <small>${tag.count}</small>`;
            el.addEventListener('click', () => {
                this.state.tag = this.state.tag === tag.name ? '' : tag.name;
                this.state.page = 1;
                this.renderTagCloud();
                this.loadPapers();
            });
            container.appendChild(el);
        });
    }
    
    async loadPapers() {
        if (this.state.isLoading) return;
        this.state.isLoading = true;
        
        const container = document.getElementById('papers-list');
        if (container) {
            container.innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>加载中...</span>
                </div>
            `;
        }
        
        try {
            const params = new URLSearchParams({
                page: this.state.page,
                page_size: this.state.pageSize,
                sort: this.state.sort,
                hide_hidden: this.state.hideHidden
            });
            
            if (this.state.q) params.append('q', this.state.q);
            if (this.state.category) params.append('category', this.state.category);
            if (this.state.tag) params.append('tag', this.state.tag);
            
            const res = await fetch(`${this.apiBase}/papers?${params}`);
            const data = await res.json();
            
            this.state.papers = data.items;
            this.state.total = data.total;
            this.state.totalPages = data.total_pages;
            
            this.renderPapers();
            this.updatePagination();
            
        } catch (err) {
            console.error('Failed to load papers:', err);
            if (container) {
                container.innerHTML = `
                    <div class="empty">
                        <div class="empty-icon">⚠️</div>
                        <p>加载失败，请稍后重试</p>
                    </div>
                `;
            }
        } finally {
            this.state.isLoading = false;
        }
    }
    
    renderPapers() {
        const container = document.getElementById('papers-list');
        if (!container) return;
        
        container.innerHTML = '';
        
        const countEl = document.getElementById('papers-count');
        if (countEl) {
            countEl.textContent = `共 ${this.state.total} 篇论文`;
        }
        
        if (this.state.papers.length === 0) {
            container.innerHTML = `
                <div class="empty">
                    <div class="empty-icon">📭</div>
                    <p>暂无论文</p>
                </div>
            `;
            return;
        }
        
        this.state.papers.forEach((paper, index) => {
            const card = this.createPaperCard(paper);
            card.style.animationDelay = `${index * 0.05}s`;
            container.appendChild(card);
        });
    }
    
    createPaperCard(paper) {
        const div = document.createElement('div');
        div.className = 'paper-card' + (paper.user_state?.is_hidden ? ' hidden' : '');
        div.dataset.arxivId = paper.arxiv_id;
        
        const userState = paper.user_state || { is_read: false, is_favorite: false, is_hidden: false };
        
        div.innerHTML = `
            <div class="paper-card-header">
                <div class="paper-title">${this.escapeHtml(paper.title)}</div>
                <span class="paper-category">${this.escapeHtml(paper.primary_category)}</span>
            </div>
            <div class="paper-meta">
                <span>📅 ${this.formatDate(paper.published_at)}</span>
                <span>✍️ ${(paper.authors || []).slice(0, 3).join(', ')}${paper.authors?.length > 3 ? ' 等' : ''}</span>
            </div>
            <div class="paper-abstract">${this.escapeHtml(paper.abstract_en)}</div>
            ${paper.abstract_zh ? `<div class="paper-abstract zh">${this.escapeHtml(paper.abstract_zh)}</div>` : ''}
            <div class="paper-tags">
                ${(paper.tags || []).slice(0, 6).map(tag => `<span class="tag">${this.escapeHtml(tag)}</span>`).join('')}
            </div>
            <div class="paper-actions">
                <button class="action-btn read-btn ${userState.is_read ? 'active' : ''}" data-arxiv="${paper.arxiv_id}" title="标记已读">
                    ${userState.is_read ? '✓ 已读' : '○ 未读'}
                </button>
                <button class="action-btn fav-btn ${userState.is_favorite ? 'active' : ''}" data-arxiv="${paper.arxiv_id}" title="收藏">
                    ${userState.is_favorite ? '★ 已收藏' : '☆ 收藏'}
                </button>
                <button class="action-btn hide-btn ${userState.is_hidden ? 'active' : ''}" data-arxiv="${paper.arxiv_id}" title="隐藏">
                    ${userState.is_hidden ? '👁 已隐藏' : '👁 隐藏'}
                </button>
                <button class="action-btn" onclick="app.showPaperDetail('${paper.arxiv_id}')">查看详情 →</button>
            </div>
        `;
        
        // Bind events
        div.querySelector('.read-btn')?.addEventListener('click', (e) => {
            this.toggleRead(paper.arxiv_id, e.target);
        });
        
        div.querySelector('.fav-btn')?.addEventListener('click', (e) => {
            this.toggleFavorite(paper.arxiv_id, e.target);
        });
        
        div.querySelector('.hide-btn')?.addEventListener('click', (e) => {
            this.toggleHide(paper.arxiv_id, e.target);
        });
        
        div.querySelector('.paper-title')?.addEventListener('click', () => {
            this.showPaperDetail(paper.arxiv_id);
        });
        
        return div;
    }
    
    updatePagination() {
        const prevBtn = document.getElementById('prev-page');
        const nextBtn = document.getElementById('next-page');
        const pageInfo = document.getElementById('page-info');
        
        if (prevBtn) prevBtn.disabled = this.state.page <= 1;
        if (nextBtn) nextBtn.disabled = this.state.page >= this.state.totalPages;
        if (pageInfo) pageInfo.textContent = `${this.state.page} / ${this.state.totalPages || 1}`;
    }
    
    // Paper Detail
    async showPaperDetail(arxivId) {
        const container = document.getElementById('paper-detail');
        if (container) {
            container.innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>加载中...</span>
                </div>
            `;
        }
        
        this.showPage('detail');
        window.location.hash = `/paper/${arxivId}`;
        
        try {
            const res = await fetch(`${this.apiBase}/papers/${arxivId}`);
            const paper = await res.json();
            
            this.state.currentPaper = paper;
            this.renderPaperDetail(paper);
            
        } catch (err) {
            console.error('Failed to load paper:', err);
            if (container) {
                container.innerHTML = `
                    <div class="empty">
                        <div class="empty-icon">⚠️</div>
                        <p>加载失败</p>
                    </div>
                `;
            }
        }
    }
    
    renderPaperDetail(paper) {
        const container = document.getElementById('paper-detail');
        if (!container) return;
        
        const userState = paper.user_state || { is_read: false, is_favorite: false, is_hidden: false };
        
        container.innerHTML = `
            <button class="back-btn" id="back-btn">← 返回列表</button>
            
            <div class="detail-header">
                <h2 class="detail-title">${this.escapeHtml(paper.title)}</h2>
                <div class="detail-meta">
                    <span class="detail-meta-item">📁 ${this.escapeHtml(paper.primary_category)}</span>
                    <span class="detail-meta-item">📅 发布: ${this.formatDate(paper.published_at)}</span>
                    <span class="detail-meta-item">🔄 更新: ${this.formatDate(paper.updated_at)}</span>
                    <span class="detail-meta-item">✍️ ${(paper.authors || []).join(', ')}</span>
                </div>
                <div class="detail-links">
                    <a href="${paper.abs_url}" target="_blank" class="detail-link">
                        📄 arXiv 原文
                    </a>
                    ${paper.pdf_url ? `
                        <a href="${paper.pdf_url}" target="_blank" class="detail-link secondary">
                            📥 PDF 下载
                        </a>
                    ` : ''}
                </div>
            </div>
            
            <div class="abstract-section">
                <h3>英文摘要</h3>
                <div class="abstract-content">${this.escapeHtml(paper.abstract_en)}</div>
            </div>
            
            <div class="abstract-section">
                <h3>中文摘要</h3>
                ${paper.abstract_zh 
                    ? `<div class="abstract-content zh">${this.escapeHtml(paper.abstract_zh)}</div>`
                    : `<div class="abstract-content pending">⏳ 翻译中或翻译失败，请稍后刷新</div>`
                }
            </div>
            
            <div class="detail-tags">
                ${(paper.tags || []).map(tag => `<span class="tag">${this.escapeHtml(tag)}</span>`).join('')}
            </div>
            
            <div class="detail-actions">
                <button class="action-btn read-btn ${userState.is_read ? 'active' : ''}" data-arxiv="${paper.arxiv_id}">
                    ${userState.is_read ? '✓ 已读' : '○ 标记已读'}
                </button>
                <button class="action-btn fav-btn ${userState.is_favorite ? 'active' : ''}" data-arxiv="${paper.arxiv_id}">
                    ${userState.is_favorite ? '★ 已收藏' : '☆ 收藏'}
                </button>
                <button class="action-btn hide-btn ${userState.is_hidden ? 'active' : ''}" data-arxiv="${paper.arxiv_id}">
                    ${userState.is_hidden ? '👁 已隐藏' : '👁 隐藏'}
                </button>
            </div>
        `;
        
        // Rebind back button
        container.querySelector('.back-btn')?.addEventListener('click', () => {
            this.showPage('list');
            window.location.hash = '/';
        });
        
        container.querySelector('.read-btn')?.addEventListener('click', (e) => {
            this.toggleRead(paper.arxiv_id, e.target);
        });
        
        container.querySelector('.fav-btn')?.addEventListener('click', (e) => {
            this.toggleFavorite(paper.arxiv_id, e.target);
        });
        
        container.querySelector('.hide-btn')?.addEventListener('click', (e) => {
            this.toggleHide(paper.arxiv_id, e.target);
        });
    }
    
    // User Actions
    async toggleRead(arxivId, btn) {
        try {
            const isActive = btn.classList.contains('active');
            const res = await fetch(`${this.apiBase}/papers/${arxivId}/state`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_read: !isActive })
            });
            
            if (res.ok) {
                btn.classList.toggle('active');
                btn.innerHTML = btn.classList.contains('active') ? '✓ 已读' : '○ 标记已读';
            }
        } catch (err) {
            console.error('Failed to update state:', err);
            this.showToast('操作失败', 'error');
        }
    }
    
    async toggleFavorite(arxivId, btn) {
        try {
            const isActive = btn.classList.contains('active');
            const res = await fetch(`${this.apiBase}/papers/${arxivId}/state`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_favorite: !isActive })
            });
            
            if (res.ok) {
                btn.classList.toggle('active');
                btn.innerHTML = btn.classList.contains('active') ? '★ 已收藏' : '☆ 收藏';
            }
        } catch (err) {
            console.error('Failed to update state:', err);
            this.showToast('操作失败', 'error');
        }
    }
    
    async toggleHide(arxivId, btn) {
        try {
            const isActive = btn.classList.contains('active');
            const res = await fetch(`${this.apiBase}/papers/${arxivId}/state`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_hidden: !isActive })
            });
            
            if (res.ok) {
                btn.classList.toggle('active');
                btn.innerHTML = btn.classList.contains('active') ? '👁 已隐藏' : '👁 隐藏';
                
                // Hide the card if we're on the list page
                const card = document.querySelector(`.paper-card[data-arxiv-id="${arxivId}"]`);
                if (card) {
                    card.style.opacity = '0';
                    setTimeout(() => card.remove(), 300);
                }
            }
        } catch (err) {
            console.error('Failed to update state:', err);
            this.showToast('操作失败', 'error');
        }
    }
    
    // Admin Functions
    async loadAdminStats() {
        try {
            const res = await fetch('/api/admin/stats', {
                headers: { 'X-Admin-Token': 'admin_secret_token' }
            });
            const stats = await res.json();
            this.renderAdminStats(stats);
        } catch (err) {
            console.error('Failed to load admin stats:', err);
        }
    }
    
    renderAdminStats(stats) {
        const container = document.getElementById('admin-stats');
        if (!container) return;
        
        container.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${stats.total_papers || 0}</div>
                <div class="stat-label">总论文数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.papers_today || 0}</div>
                <div class="stat-label">今日新增</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.translations?.success || 0}</div>
                <div class="stat-label">已翻译</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.tags?.success || 0}</div>
                <div class="stat-label">已标记</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.translations?.pending || 0}</div>
                <div class="stat-label">翻译中</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.tags?.pending || 0}</div>
                <div class="stat-label">标记中</div>
            </div>
        `;
    }
    
    async adminAction(url, successMsg, loadingMsg = '处理中...') {
        const logContainer = document.getElementById('log-content');
        if (logContainer) {
            logContainer.textContent = `${new Date().toLocaleString()}\n${loadingMsg}\n\n` + (logContainer.textContent || '');
        }
        
        this.showToast(loadingMsg, 'info');
        
        try {
            const res = await fetch(url, {
                headers: { 'X-Admin-Token': 'admin_secret_token' }
            });
            const data = await res.json();
            
            const log = `${new Date().toLocaleString()}\n✅ ${successMsg}\n${JSON.stringify(data, null, 2)}\n\n`;
            if (logContainer) {
                logContainer.textContent = log + logContainer.textContent;
            }
            
            this.showToast(successMsg, 'success');
            
            // Refresh data
            this.loadAdminStats();
            this.loadTags();
            
            if (this.state.currentPaper) {
                this.showPaperDetail(this.state.currentPaper.arxiv_id);
            }
            
        } catch (err) {
            console.error('Admin action failed:', err);
            const errorMsg = `❌ 错误: ${err.message}`;
            if (logContainer) {
                logContainer.textContent = `${new Date().toLocaleString()}\n${errorMsg}\n\n` + logContainer.textContent;
            }
            this.showToast('操作失败', 'error');
        }
    }
    
    // Preferences (localStorage)
    loadPreferences() {
        try {
            const prefs = JSON.parse(localStorage.getItem('arxivminer_prefs') || '{}');
            if (prefs.category) {
                this.state.category = prefs.category;
                const select = document.getElementById('category-filter');
                if (select) select.value = prefs.category;
            }
            if (prefs.sort) {
                this.state.sort = prefs.sort;
                const select = document.getElementById('sort-filter');
                if (select) select.value = prefs.sort;
            }
            if (prefs.hideHidden !== undefined) {
                this.state.hideHidden = prefs.hideHidden;
                const checkbox = document.getElementById('hide-hidden');
                if (checkbox) checkbox.checked = prefs.hideHidden;
            }
        } catch (e) {
            console.warn('Failed to load preferences:', e);
        }
    }
    
    savePreferences() {
        try {
            localStorage.setItem('arxivminer_prefs', JSON.stringify({
                category: this.state.category,
                sort: this.state.sort,
                hideHidden: this.state.hideHidden
            }));
        } catch (e) {
            console.warn('Failed to save preferences:', e);
        }
    }
    
    // Utilities
    scrollToTop() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    
    showToast(message, type = 'info') {
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
    
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    formatDate(dateStr) {
        if (!dateStr) return '';
        const date = new Date(dateStr);
        return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
    }
}

// Initialize app
const app = new ArxivMinerApp();
