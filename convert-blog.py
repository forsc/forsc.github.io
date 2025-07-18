#!/usr/bin/env python3
"""
Convert markdown blog posts to JavaScript data format for GitHub Pages deployment.
This script reads all .md files from the blog/ directory and creates blog-data.js
"""

import os
import re
import json
from pathlib import Path

def parse_frontmatter(content):
    """Extract YAML frontmatter from markdown content"""
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(frontmatter_pattern, content, re.DOTALL)
    
    if not match:
        raise ValueError("No valid frontmatter found")
    
    frontmatter_content = match.group(1)
    markdown_content = match.group(2)
    
    # Parse frontmatter
    frontmatter = {}
    for line in frontmatter_content.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Remove quotes
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            
            # Parse arrays (tags)
            if value.startswith('[') and value.endswith(']'):
                # Simple array parsing
                value = value[1:-1]
                if value:
                    value = [item.strip().strip('"') for item in value.split(',')]
                else:
                    value = []
            
            frontmatter[key] = value
    
    return frontmatter, markdown_content

def escape_js_string(content):
    """Escape content for JavaScript template literal"""
    # Escape backticks and backslashes for template literals
    content = content.replace('\\', '\\\\')
    content = content.replace('`', '\\`')
    content = content.replace('${', '\\${')
    return content

def convert_markdown_to_js():
    """Convert all markdown files to JavaScript data"""
    blog_dir = Path('blog')
    
    if not blog_dir.exists():
        print("Error: blog/ directory not found")
        return
    
    posts = []
    
    # Process all .md files except index.md
    for md_file in blog_dir.glob('*.md'):
        if md_file.name == 'index.md':
            continue
            
        print(f"Processing {md_file.name}...")
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            frontmatter, markdown_content = parse_frontmatter(content)
            
            # Create post object
            post = {
                'id': md_file.stem,
                'title': frontmatter.get('title', ''),
                'date': frontmatter.get('date', ''),
                'category': frontmatter.get('category', ''),
                'tags': frontmatter.get('tags', []),
                'excerpt': frontmatter.get('excerpt', ''),
                'content': escape_js_string(markdown_content.strip())
            }
            
            posts.append(post)
            
        except Exception as e:
            print(f"Error processing {md_file.name}: {e}")
    
    # Sort posts by date (newest first)
    posts.sort(key=lambda x: x['date'], reverse=True)
    
    # Generate JavaScript file
    js_content = """// Blog data embedded as JavaScript to avoid CORS issues
window.blogData = [
"""
    
    for i, post in enumerate(posts):
        js_content += "    {\n"
        js_content += f'        id: "{post["id"]}",\n'
        js_content += f'        title: "{post["title"]}",\n'
        js_content += f'        date: "{post["date"]}",\n'
        js_content += f'        category: "{post["category"]}",\n'
        js_content += f'        tags: {json.dumps(post["tags"])},\n'
        js_content += f'        excerpt: "{post["excerpt"]}",\n'
        js_content += f'        content: `\n{post["content"]}\n        `\n'
        js_content += "    }"
        
        if i < len(posts) - 1:
            js_content += ","
        js_content += "\n"
    
    js_content += "];"
    
    # Write to file
    with open('blog-data.js', 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print(f"\n✅ Successfully converted {len(posts)} blog posts to blog-data.js")
    print("📝 Posts included:")
    for post in posts:
        print(f"   - {post['title']} ({post['date']})")
    
    print("\n🚀 Your blog is now ready for GitHub Pages!")
    print("   Upload blog-data.js along with your other files.")

if __name__ == "__main__":
    convert_markdown_to_js() 