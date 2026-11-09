"""
将Markdown文档转换为PDF
需要: pandoc + wkhtmltopdf 或 markdown库
"""
import subprocess
from pathlib import Path
import sys

# 要转换的文档
DOCUMENTS = {
    'PAPER_READY_DATA.md': '01_论文核心数据与对比.pdf',
    'PAPER_VENUE_ANALYSIS.md': '02_期刊投稿分析(CS方向).pdf',
    'CS_VENUE_STRATEGY.md': '03_CS投稿策略(JMLR优先).pdf',
    'WORKSHOP_PAPER_TEMPLATE.md': '04_NeurIPS_Workshop论文模板.pdf',
    'CODE_NAVIGATION.md': '05_多模态神经算子代码导航.pdf',
    'EXPERIMENTS_CHECKPOINTS.md': '06_完整实验Checkpoints汇总.pdf',
    'ABLATION_ANALYSIS_THEORY.md': '07_消融实验理论分析.pdf',
    'FINAL_PROJECT_SUMMARY.md': '08_项目最终总结(发表级).pdf',
}

def convert_md_to_pdf(md_file, pdf_file, base_dir='.'):
    """使用pandoc将MD转为PDF"""
    md_path = Path(base_dir) / md_file
    pdf_path = Path(base_dir) / pdf_file

    if not md_path.exists():
        print(f"❌ 文件不存在: {md_path}")
        return False

    try:
        # 方法1: 用pandoc (最好)
        cmd = [
            'pandoc',
            str(md_path),
            '-o', str(pdf_path),
            '--pdf-engine=xelatex',  # 支持中文
            '-V', 'mainfont=SimSun',  # 中文字体
            '-V', 'geometry:margin=1in',
            '--table-of-contents',
            '--number-sections',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            size_mb = pdf_path.stat().st_size / (1024*1024)
            print(f"✓ {md_file} → {pdf_file} ({size_mb:.1f}MB)")
            return True
        else:
            print(f"⚠️ pandoc错误: {result.stderr[:100]}")

            # 方法2: 降级到基础HTML→PDF
            try:
                cmd = [
                    'pandoc',
                    str(md_path),
                    '-o', str(pdf_path),
                    '--pdf-engine=wkhtmltopdf',
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✓ {md_file} → {pdf_file} (wkhtmltopdf)")
                    return True
            except Exception as e:
                print(f"❌ wkhtmltopdf也失败: {e}")
                return False

    except FileNotFoundError:
        print("❌ 错误: 未安装pandoc")
        print("   Windows上用: choco install pandoc")
        print("   或访问: https://pandoc.org/installing.html")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False

def convert_all(base_dir):
    """转换所有文档"""
    print("="*60)
    print("开始转换Markdown文档到PDF")
    print("="*60 + "\n")

    success = 0
    failed = 0

    for md_file, pdf_file in DOCUMENTS.items():
        if convert_md_to_pdf(md_file, pdf_file, base_dir):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"转换完成: ✓{success} ✗{failed}")
    print(f"{'='*60}")

    if success > 0:
        print(f"\n📁 PDF文件位置: {Path(base_dir).resolve()}")
        print("\n文件列表:")
        for md_file, pdf_file in DOCUMENTS.items():
            pdf_path = Path(base_dir) / pdf_file
            if pdf_path.exists():
                size = pdf_path.stat().st_size / 1024
                print(f"  ✓ {pdf_file} ({size:.0f}KB)")

    return success > 0

if __name__ == '__main__':
    base_dir = '.'
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]

    success = convert_all(base_dir)
    sys.exit(0 if success else 1)
