"""
使用Python库将Markdown转为PDF (无需外部工具)
"""
from pathlib import Path
from datetime import datetime
import re

def md_to_html(md_content):
    """简单的Markdown到HTML转换"""
    html = md_content

    # 标题
    html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)

    # 粗体和斜体
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)

    # 代码块
    html = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'`(.*?)`', r'<code>\1</code>', html)

    # 列表
    html = re.sub(r'^\- (.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'((?:<li>.*?</li>\n?)+)', r'<ul>\1</ul>', html, flags=re.DOTALL)

    # 链接
    html = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', html)

    # 表格 (简化处理)
    html = re.sub(r'\| (.*?) \|', r'<td>\1</td>', html)

    # 换行
    html = html.replace('\n', '<br>\n')

    return html

def create_html_wrapper(title, content, author="论文作者"):
    """创建HTML包装"""
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; }}
        body {{
            font-family: 'SimSun', 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: white;
            padding: 40px;
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 28px;
            margin: 30px 0 20px 0;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 10px;
            color: #0066cc;
        }}
        h2 {{
            font-size: 22px;
            margin: 25px 0 15px 0;
            color: #0066cc;
        }}
        h3 {{
            font-size: 18px;
            margin: 20px 0 10px 0;
            color: #0066cc;
        }}
        p {{ margin: 10px 0; }}
        ul, ol {{ margin: 10px 0 10px 30px; }}
        li {{ margin: 5px 0; }}
        code {{
            background: #f5f5f5;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 10px 0;
            border-left: 4px solid #0066cc;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background: #f0f0f0;
            font-weight: bold;
        }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #0066cc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}
        strong {{ font-weight: bold; color: #0066cc; }}
        em {{ font-style: italic; }}
        a {{ color: #0066cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>生成时间: {datetime.now().strftime('%Y年%m月%d日')}</p>
    </div>

    <div class="content">
        {content}
    </div>

    <div class="footer">
        <p>© 2026 多模态神经算子混合PINN研究项目</p>
        <p>用于学术研究和学术交流</p>
    </div>
</body>
</html>
"""
    return html

def convert_md_to_html_file(md_file, html_file):
    """转换单个MD文件到HTML"""
    md_path = Path(md_file)
    html_path = Path(html_file)

    if not md_path.exists():
        print(f"❌ 文件不存在: {md_file}")
        return False

    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # 提取标题
        title_match = re.search(r'^# (.*?)$', md_content, re.MULTILINE)
        title = title_match.group(1) if title_match else md_file.replace('.md', '')

        # 转换
        html_content = md_to_html(md_content)
        html = create_html_wrapper(title, html_content)

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"✓ {md_file} → {html_file}")
        return True
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def main():
    """转换所有文档到HTML (可用浏览器另存为PDF)"""
    base_dir = Path('.')

    documents = [
        ('PAPER_READY_DATA.md', '01_论文核心数据与对比.html'),
        ('PAPER_VENUE_ANALYSIS.md', '02_期刊投稿分析_CS方向.html'),
        ('CS_VENUE_STRATEGY.md', '03_CS投稿策略_JMLR优先.html'),
        ('WORKSHOP_PAPER_TEMPLATE.md', '04_NeurIPS_Workshop论文模板.html'),
        ('CODE_NAVIGATION.md', '05_多模态神经算子代码导航.html'),
        ('EXPERIMENTS_CHECKPOINTS.md', '06_完整实验Checkpoints汇总.html'),
        ('ABLATION_ANALYSIS_THEORY.md', '07_消融实验理论分析.html'),
        ('FINAL_PROJECT_SUMMARY.md', '08_项目最终总结_发表级.html'),
    ]

    print("="*60)
    print("转换Markdown到HTML (用浏览器另存为PDF)")
    print("="*60 + "\n")

    success = 0
    for md_file, html_file in documents:
        if convert_md_to_html_file(base_dir / md_file, base_dir / html_file):
            success += 1

    print(f"\n{'='*60}")
    print(f"✓ 成功转换 {success} 个文件")
    print(f"{'='*60}")
    print("\n📝 使用方法:")
    print("  1. 用Chrome/Firefox打开HTML文件")
    print("  2. Ctrl+P 或 Cmd+P 打开打印对话")
    print("  3. 选择 '另存为PDF'")
    print("  4. 设置: 背景图形=开, 边距=默认")
    print(f"\n📁 HTML文件位置: {base_dir.resolve()}")

if __name__ == '__main__':
    main()
