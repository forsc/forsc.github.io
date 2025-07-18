// Blog Reader for Local Markdown Files
class BlogReader {
    constructor() {
        this.posts = [];
        this.currentPost = null;
        this.blogList = [
            'rag-applications-fastapi.md',
            'recommendation-systems-scale.md',
            'distributed-training-ray.md'
            // Add more blog files here
        ];
        this.init();
    }

    async init() {
        this.loadAllPosts();
        this.updateBlogCards();
        this.setupEventListeners();
    }

    loadAllPosts() {
        // Use embedded blog data to avoid CORS issues on GitHub Pages
        if (window.blogData && window.blogData.length > 0) {
            this.posts = window.blogData.map(post => ({
                ...post,
                htmlContent: this.markdownToHtml(post.content)
            }));
            console.log(`Loaded ${this.posts.length} blog posts from embedded data`);
        } else {
            console.warn('Blog data not found. Make sure blog-data.js is loaded.');
            // Fallback: try to load from markdown files (will fail on GitHub Pages due to CORS)
            this.loadMarkdownPosts();
        }
        
        // Sort posts by date (newest first)
        this.posts.sort((a, b) => new Date(b.date) - new Date(a.date));
    }

    async loadMarkdownPosts() {
        // Fallback method for local development
        for (const filename of this.blogList) {
            try {
                const response = await fetch(`blog/${filename}`);
                if (response.ok) {
                    const content = await response.text();
                    const post = this.parseMarkdown(content, filename);
                    this.posts.push(post);
                }
            } catch (error) {
                console.warn(`Could not load blog post: ${filename}`, error);
            }
        }
    }

    parseMarkdown(content, filename) {
        // Extract frontmatter
        const frontmatterRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;
        const match = content.match(frontmatterRegex);
        
        if (!match) {
            throw new Error('Invalid markdown format');
        }

        const frontmatter = this.parseFrontmatter(match[1]);
        const markdownContent = match[2];
        
        return {
            ...frontmatter,
            content: markdownContent,
            filename: filename,
            id: filename.replace('.md', ''),
            htmlContent: this.markdownToHtml(markdownContent)
        };
    }

    parseFrontmatter(frontmatter) {
        const lines = frontmatter.trim().split('\n');
        const result = {};
        
        lines.forEach(line => {
            const [key, ...valueParts] = line.split(':');
            if (key && valueParts.length > 0) {
                let value = valueParts.join(':').trim();
                
                // Remove quotes
                if (value.startsWith('"') && value.endsWith('"')) {
                    value = value.slice(1, -1);
                }
                
                // Parse arrays (tags)
                if (value.startsWith('[') && value.endsWith(']')) {
                    value = value.slice(1, -1).split(',').map(item => 
                        item.trim().replace(/"/g, '')
                    );
                }
                
                result[key.trim()] = value;
            }
        });
        
        return result;
    }

    markdownToHtml(markdown) {
        // Simple markdown to HTML converter
        let html = markdown;
        
        // Headers
        html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
        html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
        
        // Code blocks
        html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
        
        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Bold and italic
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        
        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
        
        // Lists
        html = html.replace(/^\* (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        
        // Paragraphs
        html = html.replace(/\n\n/g, '</p><p>');
        html = '<p>' + html + '</p>';
        
        // Clean up empty paragraphs
        html = html.replace(/<p><\/p>/g, '');
        html = html.replace(/<p>(<h[1-6]>)/g, '$1');
        html = html.replace(/(<\/h[1-6]>)<\/p>/g, '$1');
        html = html.replace(/<p>(<ul>)/g, '$1');
        html = html.replace(/(<\/ul>)<\/p>/g, '$1');
        html = html.replace(/<p>(<pre>)/g, '$1');
        html = html.replace(/(<\/pre>)<\/p>/g, '$1');
        
        return html;
    }

    updateBlogCards() {
        const blogGrid = document.querySelector('.blog-grid');
        if (!blogGrid) return;

        // If we have successfully loaded markdown posts, remove fallback cards
        if (this.posts.length > 0) {
            const fallbackCards = blogGrid.querySelectorAll('.fallback-card');
            fallbackCards.forEach(card => card.remove());
        }

        // Add dynamic cards for loaded posts
        this.posts.forEach((post, index) => {
            const cardElement = this.createBlogCard(post, index);
            blogGrid.appendChild(cardElement);
        });

        // If no posts loaded, show a message
        if (this.posts.length === 0) {
            console.log('No markdown blog posts loaded. Showing fallback cards.');
        }
    }

    createBlogCard(post, index) {
        const card = document.createElement('article');
        card.className = 'blog-card';
        card.innerHTML = `
            <div class="blog-header">
                <div class="blog-category">${post.category}</div>
                <div class="blog-date">${this.formatDate(post.date)}</div>
            </div>
            <h3 class="blog-title">${post.title}</h3>
            <p class="blog-excerpt">${post.excerpt}</p>
            <div class="blog-tags">
                ${post.tags.map(tag => `<span class="blog-tag">${tag}</span>`).join('')}
            </div>
            <a href="#" class="blog-read-more" data-post-id="${post.id}">
                Read More <i class="fas fa-arrow-right"></i>
            </a>
        `;

        // Add hover effects
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-10px)';
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
        });

        return card;
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    }

    setupEventListeners() {
        // Handle "Read More" clicks
        document.addEventListener('click', (e) => {
            if (e.target.closest('.blog-read-more')) {
                e.preventDefault();
                const postId = e.target.closest('.blog-read-more').dataset.postId;
                if (postId) {
                    this.openBlogPost(postId);
                }
            }
        });

        // Handle modal close
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('blog-modal') || 
                e.target.classList.contains('blog-modal-close')) {
                this.closeBlogPost();
            }
        });

        // Handle escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeBlogPost();
            }
        });
    }

    openBlogPost(postId) {
        const post = this.posts.find(p => p.id === postId);
        if (!post) return;

        this.currentPost = post;
        this.createBlogModal(post);
    }

    createBlogModal(post) {
        // Remove existing modal
        const existingModal = document.querySelector('.blog-modal');
        if (existingModal) {
            existingModal.remove();
        }

        const modal = document.createElement('div');
        modal.className = 'blog-modal';
        modal.innerHTML = `
            <div class="blog-modal-content">
                <div class="blog-modal-header">
                    <button class="blog-modal-close">&times;</button>
                    <div class="blog-modal-meta">
                        <span class="blog-modal-category">${post.category}</span>
                        <span class="blog-modal-date">${this.formatDate(post.date)}</span>
                    </div>
                    <h1 class="blog-modal-title">${post.title}</h1>
                    <div class="blog-modal-tags">
                        ${post.tags.map(tag => `<span class="blog-tag">${tag}</span>`).join('')}
                    </div>
                </div>
                <div class="blog-modal-body">
                    ${post.htmlContent}
                </div>
                <div class="blog-modal-footer">
                    <p>Published on ${this.formatDate(post.date)}</p>
                    <div class="blog-modal-social">
                        <a href="https://linkedin.com/in/rahulbhow" target="_blank" class="social-link">
                            <i class="fab fa-linkedin"></i> Share on LinkedIn
                        </a>
                        <a href="mailto:?subject=${encodeURIComponent(post.title)}&body=${encodeURIComponent('Check out this blog post: ' + post.title)}" class="social-link">
                            <i class="fas fa-envelope"></i> Share via Email
                        </a>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        document.body.style.overflow = 'hidden';

        // Add syntax highlighting if Prism is available
        if (window.Prism) {
            Prism.highlightAllUnder(modal);
        }

        // Animate modal in
        setTimeout(() => {
            modal.style.opacity = '1';
            modal.querySelector('.blog-modal-content').style.transform = 'translateY(0)';
        }, 10);
    }

    closeBlogPost() {
        const modal = document.querySelector('.blog-modal');
        if (modal) {
            modal.style.opacity = '0';
            modal.querySelector('.blog-modal-content').style.transform = 'translateY(-50px)';
            
            setTimeout(() => {
                modal.remove();
                document.body.style.overflow = 'auto';
            }, 300);
        }
        this.currentPost = null;
    }

    // Search functionality
    searchPosts(query) {
        const filteredPosts = this.posts.filter(post => {
            const searchText = `${post.title} ${post.excerpt} ${post.tags.join(' ')} ${post.category}`.toLowerCase();
            return searchText.includes(query.toLowerCase());
        });
        return filteredPosts;
    }

    // Filter by category
    filterByCategory(category) {
        return this.posts.filter(post => post.category === category);
    }

    // Get all categories
    getAllCategories() {
        const categories = [...new Set(this.posts.map(post => post.category))];
        return categories.sort();
    }

    // Get all tags
    getAllTags() {
        const tags = [...new Set(this.posts.flatMap(post => post.tags))];
        return tags.sort();
    }
}

// Initialize blog reader when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    if (document.querySelector('.blog-section') || document.querySelector('#blog')) {
        window.blogReader = new BlogReader();
    }
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BlogReader;
} 