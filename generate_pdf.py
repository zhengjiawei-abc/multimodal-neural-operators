"""
使用reportlab自动生成PDF (无需浏览器)
"""
import sys
from pathlib import Path

def try_install_reportlab():
    """尝试安装reportlab"""
    try:
        import reportlab
        return True
    except ImportError:
        try:
            import subprocess
            print("📦 正在安装reportlab...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'reportlab', '-q'])
            print("✓ reportlab安装成功")
            return True
        except:
            return False

def html_to_pdf_with_reportlab():
    """用reportlab生成PDF"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
        from reportlab.lib import colors
        from datetime import datetime
        import re

        documents = [
            ('PAPER_READY_DATA.md', '01_论文核心数据与对比.pdf'),
            ('CS_VENUE_STRATEGY.md', '03_CS投稿策略_JMLR优先.pdf'),
            ('WORKSHOP_PAPER_TEMPLATE.md', '04_NeurIPS_Workshop论文模板.pdf'),
            ('FINAL_PROJECT_SUMMARY.md', '08_项目最终总结_发表级.pdf'),
        ]

        base_dir = Path('.')
        success = 0

        print("="*60)
        print("使用reportlab生成PDF")
        print("="*60 + "\n")

        for md_file, pdf_file in documents:
            md_path = base_dir / md_file
            pdf_path = base_dir / pdf_file

            if not md_path.exists():
                print(f"❌ 文件不存在: {md_file}")
                continue

            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 提取标题
                title_match = re.search(r'^# (.*?)$', content, re.MULTILINE)
                title = title_match.group(1) if title_match else md_file

                # 创建PDF
                doc = SimpleDocTemplate(
                    str(pdf_path),
                    pagesize=A4,
                    rightMargin=72, leftMargin=72,
                    topMargin=72, bottomMargin=18
                )

                styles = getSampleStyleSheet()
                story = []

                # 标题
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor=colors.HexColor('#0066cc'),
                    spaceAfter=30,
                    alignment=1,  # 居中
                )
                story.append(Paragraph(title, title_style))
                story.append(Spacer(1, 0.3*inch))

                # 内容行
                lines = content.split('\n')
                for line in lines:
                    if line.startswith('# '):
                        story.append(Paragraph(line[2:], styles['Heading1']))
                        story.append(Spacer(1, 0.2*inch))
                    elif line.startswith('## '):
                        story.append(Paragraph(line[3:], styles['Heading2']))
                        story.append(Spacer(1, 0.15*inch))
                    elif line.startswith('### '):
                        story.append(Paragraph(line[4:], styles['Heading3']))
                        story.append(Spacer(1, 0.1*inch))
                    elif line.strip():
                        story.append(Paragraph(line, styles['Normal']))
                        story.append(Spacer(1, 0.1*inch))
                    else:
                        story.append(Spacer(1, 0.05*inch))

                # 页脚
                story.append(Spacer(1, 0.5*inch))
                footer_text = f"生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M')} | 多模态神经算子混合PINN研究"
                story.append(Paragraph(footer_text, ParagraphStyle(
                    'Footer',
                    parent=styles['Normal'],
                    fontSize=9,
                    textColor=colors.grey,
                    alignment=1
                )))

                # 生成PDF
                doc.build(story)

                file_size = pdf_path.stat().st_size / 1024
                print(f"✓ {md_file} → {pdf_file} ({file_size:.0f}KB)")
                success += 1

            except Exception as e:
                print(f"❌ {md_file}: {str(e)[:50]}")

        print(f"\n{'='*60}")
        print(f"✓ 生成 {success} 个PDF文件")
        print(f"{'='*60}")

        return success > 0

    except ImportError as e:
        print(f"❌ 缺少库: {e}")
        return False

if __name__ == '__main__':
    if try_install_reportlab():
        html_to_pdf_with_reportlab()
    else:
        print("\n⚠️ 无法生成PDF")
        print("备选方案:")
        print("  1. 用浏览器打开HTML文件")
        print("  2. Ctrl+P → 另存为PDF")
        print("  3. 设置: 背景图形=开")
