#!/usr/bin/env python3
"""
summarize.py - 用 Claude AI 总结知识库内容。

Usage:
    python summarize.py                         # 总结所有内容
    python summarize.py --query "工作笔记"       # 总结相关主题
    python summarize.py --query "会议" --top 20
    python summarize.py --output summary.txt    # 保存到文件
"""
import argparse
import sys
from pathlib import Path

import anthropic
from rich.console import Console
from rich.panel import Panel

from knowledge_base import KnowledgeBase
from config import ANTHROPIC_API_KEY, SUMMARIZE_MODEL

console = Console()

MAX_CONTEXT_CHARS = 100_000   # ~25K tokens，在 haiku 上下文窗口范围内
MAX_DOCS_FOR_SUMMARY = 50     # 单次总结最多文档数


def build_context(docs: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """将文档列表拼接为上下文字符串，超出限制时截断。"""
    parts = []
    total = 0
    for doc in docs:
        entry = f"=== {doc['file_name']} ===\n{doc['ocr_text']}\n\n"
        if total + len(entry) > max_chars:
            parts.append(
                f"[...还有 {len(docs) - len(parts)} 份文档因超出长度限制未显示...]"
            )
            break
        parts.append(entry)
        total += len(entry)
    return "".join(parts)


def summarize_with_claude(context: str, focus: str | None) -> str:
    """调用 Claude API 生成总结。"""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    focus_instruction = (
        f"\n请重点关注与【{focus}】相关的信息。"
        if focus
        else ""
    )

    system_prompt = (
        "你是一个专业的知识整理助手，擅长从照片OCR提取的文字中提炼关键信息。"
        "内容可能包含中文、英文或混合文字。请用中文进行总结。"
        "总结要简洁有重点，突出重要信息。"
    )

    user_prompt = f"""以下是从照片中通过OCR提取的文字内容。
{focus_instruction}

请根据以下内容提供一份结构化总结，包括：
1. 主要主题和内容分类
2. 关键信息、日期、人名、数字等重要细节
3. 值得关注的要点或待办事项（如有）

--- 开始 OCR 文字 ---
{context}
--- 结束 OCR 文字 ---

请提供清晰的结构化总结："""

    response = client.messages.create(
        model=SUMMARIZE_MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def main():
    parser = argparse.ArgumentParser(description="用 AI 总结照片知识库内容")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="只总结与该主题相关的内容（使用语义搜索筛选）",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=MAX_DOCS_FOR_SUMMARY,
        help="最多总结多少份文档",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="将总结保存到指定文件",
    )
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        console.print("[red]错误：未设置 ANTHROPIC_API_KEY，请配置 .env 文件。[/red]")
        sys.exit(1)

    kb = KnowledgeBase()

    try:
        if args.query:
            console.print(f"[cyan]正在搜索与以下主题相关的内容:[/cyan] [bold]{args.query}[/bold]")
            results = kb.search(args.query, top_k=args.top)
            if not results:
                console.print("[yellow]未找到相关文档。[/yellow]")
                return
            docs = [{"file_name": r["file_name"], "ocr_text": r["text"]} for r in results]
        else:
            docs = kb.get_all_texts()
            docs = docs[: args.top]

        if not docs:
            console.print("[yellow]知识库中暂无内容，请先运行 batch_process.py。[/yellow]")
            return

        console.print(f"[cyan]正在总结[/cyan] [bold]{len(docs)}[/bold] [cyan]份文档...[/cyan]")
        context = build_context(docs)
        summary = summarize_with_claude(context, focus=args.query)

        title = f'总结：{args.query}' if args.query else "知识库总结"
        console.print(Panel(summary, title=title, border_style="green"))

        if args.output:
            args.output.write_text(summary, encoding="utf-8")
            console.print(f"[green]总结已保存到 {args.output}[/green]")

    finally:
        kb.close()


if __name__ == "__main__":
    main()
