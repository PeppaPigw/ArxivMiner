/**
 * ArxivMiner Frontend Application
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
            tags: []
        };
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadTags();
        this.loadPapers();
        this.setupRouting();
    }
    
    bindEvents() {
        // Search
        document.getElementById('search-btn').addEventListener('click', () => {
            this.state.q = document.getElementById('search-input').value;
            this.state.page = 1;
            this.loadPapers();
        });
        
        document.getElementById('search-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.state.q = e.target.value;
                this.state.page = 1;
                this.loadPapers();
            }
        });
        
        // Filters
        document.getElementById('category-filter').addEventListener('change', (e) => {
            this.state.category = e.target.value;
            this.state.page = 1;
            this.loadPapers();
        });
        
        document.getElementById('sort-filter').addEventListener('change', (e) => {
            this.state.sort = e.target.value;
            this.state.page = 1;
            this.loadPapers();
        });
        
        document.getElementById('hide-hidden').addEventListener('change', (e) => {
            this.state.hideHidden = e.target.checked;
            this.state.page = 1;
            this.loadPapers();
        });
        
        // Refresh
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.loadPapers();
        });
        
        // Pagination
        document.getElementById('prev-page').addEventListener('click', () => {
            if (this.state.page > 1) {
                this.state.page--;
                this.loadPapers();
            }
        });
        
        document.getElementById('next-page').addEventListener('click', () => {
            if (this.state.page < this.state.totalPages) {
                this.state.page++;
                this.loadPapers();
            }
        });
        
        // Back button
        document.getElementById('back-btn').addEventListener('click', () => {
            this.showPage('list');
        });
        
        // Admin buttons
        document.getElementById('fetch-btn').addEventListener('click', () => {
            this.adminAction('/api/admin/fetch', '抓取完成');
        });
        
        document.getElementById('retranslate-btn').addEventListener('click', () => {
            this.adminAction('/api/admin/retranslate?status=failed', '翻译重试完成');
        });
        
        document.getElementById('retag-btn').addEventListener('click', () => {
            this.adminAction('/api/admin/retag?status=failed', '标记重试完成');
        });
        
        document.getElementById('process-pending-btn').addEventListener('click', () => {
            this.adminAction('/api/admin/process-pending', '处理完成');
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
    }
    
    setupRouting() {
        // Hash-based routing
        window.addEventListener('hashchange', () => {
            this.handleRoute();
        });
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
        document.getElementById(`page-${pageName}`).classList.add('active');
    }
    
    async loadTags() {
        try {
            const res = await fetch(`${this.apiBase}/tags?limit=30`);
            const data = await res.json();
            this.state.tags = data;
            this.renderTagCloud();
        } catch (err) {
            console.error('Failed to load tags:', err);
        }
    }
    
    renderTagCloud() {
        const container = document.getElementById('tag-cloud');
        container.innerHTML = '';
        
        this.state.tags.forEach(tag => {
            const el = document.createElement('span');
            el.className = 'tag' + (this.state.tag === tag.name ? ' active' : '');
            el.textContent = `${tag.name} (${tag.count})`;
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
        const container = document.getElementById('papers-list');
        container.innerHTML = '<div class="loading">加载中...</div>';
        
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
            container.innerHTML = '<div class="empty">加载失败，请稍后重试</div>';
        }
    }
    
    renderPapers() {
        const container = document.getElementById('papers-list');
        container.innerHTML = '';
        
        document.getElementById('papers-count').textContent = 
            `共 ${this.state.total} 篇论文`;
        
        if (this.state.papers.length === 0) {
            container.innerHTML = '<div class="empty">暂无论文</div>';
            return;
        }
        
        this.state.papers.forEach(paper => {
            const card = this.createPaperCard(paper);
            container.appendChild(card);
        });
    }
    
    createPaperCard(paper) {
        const div = document.createElement('div');
        div.className = 'paper-card' + (paper.user_state?.is_hidden ? ' hidden' : '');
        div.dataset.arxivId = paper.arxiv_id;
        
        const userState = paper.user_state || { is_read: false, is_favorite: false, is_hidden: false };
        
        div.innerHTML = `
            <div class="paper-title">${this.escapeHtml(paper.title)}</div>
            <div class="paper-meta">
                <span class="category">${paper.primary_category}</span>
                <span>${this.formatDate(paper.published_at)}</span>
                <span>${paper.authors.slice(0, 3).join(', ')}${paper.authors.length > 3 ? ' et al.' : ''}</span>
            </div>
            <div class="paper-abstract en">${this.escapeHtml(paper.abstract_en)}</div>
            ${paper.abstract_zh ? `<div class="paper-abstract zh">${this.escapeHtml(paper.abstract_zh)}</div>` : ''}
            <div class="paper-tags">
                ${(paper.tags || []).map(tag => `<span class="tag">${tag}</span>`).join('')}
            </div>
            <div class="paper-actions">
                <button class="action-btn read-btn ${userState.is_read ? 'active' : ''}" data-arxiv="${paper.arxiv_id}">
                    ${userState.is_read ? '已读' : '标记已读'}
                </button>
                <button class="action-btn fav-btn ${userState.is_favorite ? 'active' : ''}" data-arxiv="${paper.arxiv_id}">
                    ${userState.is_favorite ? '已收藏' : '收藏'}
                </button>
                <button class="action-btn hide-btn ${userState.is_hidden ? 'active' : ''}" data-arxiv="${paper.arxiv_id}">
                    ${userState.is_hidden ? '已隐藏' : '隐藏'}
                </button>
                <button class="action-btn" onclick="app.showPaperDetail('${paper.arxiv_id}')">查看详情</button>
            </div>
        `;
        
        // Bind action events
        div.querySelector('.read-btn').addEventListener('click', (e) => {
            this.toggleRead(paper.arxiv_id, e.target);
        });
        
        div.querySelector('.fav-btn').addEventListener('click', (e) => {
            this.toggleFavorite(paper.arxiv_id, e.target);
        });
        
        div.querySelector('.hide-btn').addEventListener('click', (e) => {
            this.toggleHide(paper.arxiv_id, e.target);
        });
        
        div.querySelector('.paper-title').addEventListener('click', () => {
            this.showPaperDetail(paper.arxiv_id);
        });
        
        return div;
    }
    
    updatePagination() {
        const prevBtn = document.getElementById('prev-page');
        const nextBtn = document.getElementById('next-page');
        const pageInfo = document.getElementById('page-info');
        
        prevBtn.disabled = this.state.page <= 1;
        nextBtn.disabled = this.state.page >= this.state.totalPages;
        pageInfo.textContent = `${this.state.page} / ${this.state.totalPages}`;
    }
    
    async showPaperDetail(arxivId) {
        const container = document.getElementById('paper-detail');
        container.innerHTML = '<div class="loading">加载中...</div>';
        
        this.showPage('detail');
        window.location.hash = `/paper/${arxivId}`;
        
        try {
            const res = await fetch(`${this.apiBase}/papers/${arxivId}`);
            const paper = await res.json();
            
            this.state.currentPaper = paper;
            this.renderPaperDetail(paper);
            
        } catch (err) {
            console.error('Failed to load paper:', err);
            container.innerHTML = '<div class="empty">加载失败</div>';
        }
    }
    
    renderPaperDetail(paper) {
        const container = document.getElementById('paper-detail');
        const userState = paper.user_state || { is_read: false, is_favorite: false, is_hidden: false };
        
        container.innerHTML = `
            <div class="detail-header">
                <h2 class="detail-title">${this.escapeHtml(paper.title)}</h2>
                <div class="detail-meta">
                    <span class="category">${paper.primary_category}</span>
                    <span>发布: ${this.formatDate(paper.published_at)}</span>
                    <span>更新: ${this.formatDate(paper.updated_at)}</span>
                    <span>作者: ${paper.authors.join(', ')}</span>
                </div>
                <div class="detail-links">
                    <a href="${paper.abs_url}" target="_blank" class="detail-link">arXiv 原文</a>
                    ${paper.pdf_url ? `<a href="${paper.pdf_url}" target="_blank" class="detail-link">PDF 下载</a>` : ''}
                </div>
            </div>
            
            <div class="abstract-section">
                <h3>英文摘要</h3>
                <p class="en">${this.escapeHtml(paper.abstract_en)}</p>
            </div>
            
            <div class="abstract-section">
                <h3>中文摘要</h3>
                ${paper.abstract_zh 
                    ? `<p class="zh">${this.escapeHtml(paper.abstract_zh)}</p>`
                    : `<p class="pending">翻译中或翻译失败，请稍后刷新</p>`
                }
            </div>
            
            <div class="detail-tags">
                ${(paper.tags || []).map(tag => `<span class="tag">${tag}</span>`).join('')}
            </div>
            
            <div class="detail-actions">
                <button class="action-btn read-btn ${userState.is_read ? 'active' : ''}" data-arxiv="${paper.arxiv_id}">
                    ${userState.is_read ? '已读' : '标记已读'}
                </button>
                <button class="action-btn fav-btn ${userState.is_favorite ? 'active' : ''}" data-arxiv="${paper.arxiv_id}">
                    ${userState.is_favorite ? '已收藏' : '收藏'}
                </button>
                <button class="action-btn hide-btn ${userState.is_hidden ? 'active' : ''}" data-arxiv="${paper.arxiv_id}">
                    ${userState.is_hidden ? '已隐藏' : '隐藏'}
                </button>
            </div>
        `;
        
        container.querySelector('.read-btn').addEventListener('click', (e) => {
            this.toggleRead(paper.arxiv_id, e.target);
        });
        
        container.querySelector('.fav-btn').addEventListener('click', (e) => {
            this.toggleFavorite(paper.arxiv_id, e.target);
        });
        
        container.querySelector('.hide-btn').addEventListener('click', (e) => {
            this.toggleHide(paper.arxiv_id, e.target);
        });
    }
    
    async toggleRead(arxivId, btn) {
        try {
            const res = await fetch(`${this.apiBase}/papers/${arxivId}/state`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_read: !btn.classList.contains('active') })
            });
            
            if (res.ok) {
                btn.classList.toggle('active');
                btn.textContent = btn.classList.contains('active') ? '已读' : '标记已读';
            }
        } catch (err) {
            console.error('Failed to update state:', err);
        }
    }
    
    async toggleFavorite(arxivId, btn) {
        try {
            const res = await fetch(`${this.apiBase}/papers/${arxivId}/state`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_favorite: !btn.classList.contains('active') })
            });
            
            if (res.ok) {
                btn.classList.toggle('active');
                btn.textContent = btn.classList.contains('active') ? '已收藏' : '收藏';
            }
        } catch (err) {
            console.error('Failed to update state:', err);
        }
    }
    
    async toggleHide(arxivId, btn) {
        try {
            const res = await fetch(`${this.apiBase}/papers/${arxivId}/state`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_hidden: !btn.classList.contains('active') })
            });
            
            if (res.ok) {
                btn.classList.toggle('active');
                btn.textContent = btn.classList.contains('active') ? '已隐藏' : '隐藏';
            }
        } catch (err) {
            console.error('Failed to update state:', err);
        }
    }
    
    // Admin
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
        container.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${stats.total_papers}</div>
                <div class="stat-label">总论文数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.papers_today}</div>
                <div class="stat-label">今日新增</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.translations.success}</div>
                <div class="stat-label">已翻译</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.tags.success}</div>
                <div class="stat-label">已标记</div>
            </div>
        `;
    }
    
    async adminAction(url, successMsg) {
        const logContainer = document.getElementById('log-content');
        logContainer.textContent = `执行中...\n`;
        
        try {
            const res = await fetch(url, {
                headers: { 'X-Admin-Token': 'admin_secret_token' }
            });
            const data = await res.json();
            
            const log = `${new Date().toLocaleString()}\n${JSON.stringify(data, null, 2)}\n\n`;
            logContainer.textContent = log + logContainer.textContent;
            
            this.loadAdminStats();
            this.loadTags();
            
            if (this.state.currentPaper) {
                this.showPaperDetail(this.state.currentPaper.arxiv_id);
            }
            
        } catch (err) {
            logContainer.textContent = `错误: ${err.message}\n\n` + logContainer.textContent;
        }
    }
    
    // Utilities
    escapeHtml(text) {
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
