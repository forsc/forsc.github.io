# Rahul Bhowmick - Portfolio Website

A modern, responsive portfolio website showcasing machine learning engineering expertise and professional experience.

## 🌟 Features

- **Modern Design**: Clean, professional layout with gradient accents and smooth animations
- **Responsive**: Optimized for all device sizes (desktop, tablet, mobile)
- **Interactive Navigation**: Smooth scrolling with active section highlighting
- **Timeline Experience**: Beautiful timeline layout for professional experience
- **Blog Section**: Professional blog showcase with categorized posts and social integration
- **Skill Showcase**: Organized skill categories with hover effects
- **Contact Integration**: Direct links to email, LinkedIn, and GitHub
- **Fast Loading**: Optimized CSS and JavaScript for performance
- **GitHub Pages Ready**: Configured for easy deployment

## 🚀 Quick Setup

### Option 1: Deploy to GitHub Pages (Recommended)

1. **Fork or Download** this repository
2. **Create a new repository** on GitHub (name it `your-username.github.io` for personal site)
3. **Upload files** to your repository:
   - `index.html`
   - `styles.css`
   - `script.js`
   - `README.md`

4. **Enable GitHub Pages**:
   - Go to repository Settings
   - Scroll to "Pages" section
   - Select "Deploy from a branch"
   - Choose "main" branch and "/ (root)" folder
   - Click Save

5. **Access your site**: `https://your-username.github.io`

### Option 2: Local Development

1. **Clone/Download** the repository
2. **Open** `index.html` in a web browser
3. **Use Live Server** (VS Code extension) for better development experience

## 📁 File Structure

```
portfolio/
├── index.html          # Main HTML file
├── styles.css          # All CSS styling
├── script.js           # JavaScript functionality
├── blog-reader.js      # Blog system for markdown files
├── blog-data.js        # Blog posts data (GitHub Pages compatible)
├── convert-blog.py     # Convert markdown to JavaScript (optional)
├── README.md           # This file
└── blog/               # Blog posts directory (for reference)
    ├── index.md        # Blog management guide
    ├── rag-applications-fastapi.md
    ├── recommendation-systems-scale.md
    └── distributed-training-ray.md
```

## 🎨 Customization

### Personal Information

Edit `index.html` to update:
- Name and title in `<title>` and hero section
- Contact information (email, phone, location)
- Social media links (LinkedIn, GitHub)
- Professional experience details
- Education information
- Skills and technologies
- Blog posts (title, excerpt, category, tags, dates)
- Medium/LinkedIn links for blog CTA

### Styling

Modify `styles.css`:
- Color scheme: Update CSS custom properties in `:root`
- Fonts: Change the Google Fonts import and font-family
- Layout: Adjust spacing, sizes, and grid layouts
- Animations: Customize transition timings and effects

### Color Scheme

Current colors (easily customizable in CSS):
```css
--primary-color: #667eea;
--secondary-color: #764ba2;
--accent-color: #f093fb;
```

### Blog Customization

To customize the blog section:

1. **Add/Edit Blog Posts**: Create new markdown files in the `/blog/` directory (see structure below)
2. **Update Blog List**: Add new markdown filenames to the `blogList` array in `blog-reader.js`
3. **Update Categories**: Each post has a colored category badge defined in the frontmatter
4. **Change Social Links**: Update the Medium/LinkedIn links in the blog CTA section
5. **Modify Colors**: Blog categories use different gradient colors defined in `script.js`

**Markdown Blog Post Structure**:
```markdown
---
title: "Your Blog Post Title"
date: "2024-12-15"
category: "Machine Learning"
tags: ["Python", "AI", "Tutorial"]
excerpt: "A brief description of your blog post that appears on the blog cards."
---

# Your Blog Post Title

## Introduction

Your blog content here in markdown format...

### Code Examples
```python
def hello_world():
    print("Hello, World!")
```

## Conclusion

More content...
```

**Adding New Blog Posts**:
1. Create a new `.md` file in the `/blog/` directory
2. Add the filename to `blogList` array in `blog-reader.js`:
```javascript
this.blogList = [
    'rag-applications-fastapi.md',
    'recommendation-systems-scale.md',
    'distributed-training-ray.md',
    'your-new-post.md'  // Add your new post here
];
```

## 🛠️ Technologies Used

- **HTML5**: Semantic markup and structure
- **CSS3**: Modern styling with Grid, Flexbox, and animations
- **JavaScript**: Interactive functionality, animations, and blog system
- **Markdown**: Blog posts written in markdown with frontmatter
- **Font Awesome**: Icons for social links and contact
- **Google Fonts**: Poppins font family

## 📱 Responsive Design

- **Desktop**: Full-width layout with sidebar navigation
- **Tablet**: Adjusted grid layouts and spacing
- **Mobile**: Stacked layout with hamburger menu

## ⚡ Performance Features

- Optimized CSS with custom properties
- Efficient JavaScript with event delegation
- Lazy loading animations with Intersection Observer
- Minimal external dependencies
- Compressed and optimized code

## 🌐 Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## 📞 Contact Information

Update the contact section in `index.html` with your details:

- **Email**: rahul.bnghs@gmail.com
- **Phone**: +91-9836534899 / +91-7980498265
- **Location**: Bengaluru, Karnataka, India
- **LinkedIn**: [linkedin.com/in/rahulbhow](https://linkedin.com/in/rahulbhow)
- **GitHub**: [github.com/forsc](https://github.com/forsc)

## 📝 Content Sections

1. **Hero**: Introduction and call-to-action
2. **About**: Professional summary and highlights
3. **Experience**: Timeline of work history
4. **Skills**: Technical skills organized by category
5. **Blog**: Latest blog posts with categories and tags
6. **Contact**: Contact information and social links

## 🚀 Deployment Options

### GitHub Pages (Free)
- Perfect for personal portfolios
- Custom domain support available
- Automatic SSL certificate

### Netlify (Free tier available)
- Drag and drop deployment
- Form handling capabilities
- Advanced build options

### Vercel (Free tier available)
- Instant deployment
- Global CDN
- Analytics included

## 📊 Analytics (Optional)

To add Google Analytics:

1. Get your tracking ID from Google Analytics
2. Add this code before `</head>` in `index.html`:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

## 🔧 Troubleshooting

### Common Issues:

1. **Site not loading**: Check if files are in the root directory
2. **Styling not applied**: Verify CSS file path in HTML
3. **JavaScript not working**: Check browser console for errors
4. **Mobile layout issues**: Test responsiveness with browser dev tools

### Blog System Issues:

1. **Blog posts not loading on GitHub Pages**: 
   - **Problem**: CORS restrictions prevent fetching `.md` files
   - **Solution**: Use `blog-data.js` (already included) instead of markdown files
   - **Status**: ✅ Fixed - blog posts are embedded as JavaScript data

2. **Adding new blog posts**:
   - **Option 1**: Edit `blog-data.js` directly (quick)
   - **Option 2**: Use `convert-blog.py` to convert markdown files
   - **Option 3**: Add posts to `window.blogData` array manually

3. **Blog content not displaying**:
   - Check browser console for errors
   - Ensure `blog-data.js` loads before `blog-reader.js`
   - Verify `window.blogData` exists in browser dev tools

### GitHub Pages Specific:

1. **404 Error**: Ensure `index.html` is in the root directory
2. **Changes not updating**: GitHub Pages can take a few minutes to update
3. **Custom domain issues**: Check DNS settings and repository settings
4. **Blog CORS errors**: Use the JavaScript data approach (already implemented)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Feel free to fork this project and adapt it for your own portfolio. If you make improvements that could benefit others, consider submitting a pull request!

## 📧 Support

If you need help customizing this portfolio or have questions about deployment, feel free to reach out through the contact information provided.

---

**Built with ❤️ for the developer community** 