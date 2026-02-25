#!/usr/bin/env python3
"""
query_kb.py - 搜索和浏览知识库。

Usage:
    python query_kb.py search "会议记录"
    python query_kb.py search "meeting notes" --top 10
    python query_kb.py keyword "发票"
    python query_kb.py show IMG_1234.jpg
    python query_kb.py stats
    python query_kb.py list --has-text
"""
import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from knowledge_base import KnowledgeBase
from config import TOP_K_RESULTS

console = Console()


def cmd_search(kb: KnowledgeBase, query: str, top_k: int):
    """语义搜索（向量相似度）。"""
    if kb.collection.count() == 0:
        console.print("[yellow]知识库为空，请先运行 batch_process.py。[/yellow]")
        return

    console.print(f"\n[cyan]语义搜索:[/cyan] [bold]{query}[/bold]\n")
    results = kb.search(query, top_k=top_k)

    if not results:
        console.print("[yellow]未找到相关结果。[/yellow]")
        return

    for i, hit in enumerate(results, 1):
        sim = hit["similarity"]
        if sim > 0.7:
            color = "green"
        elif sim > 0.4:
            color = "yellow"
        else:
            color = "red"

        header = (
            f"[bold]#{i}[/bold]  "
            f"[{color}]相似度: {sim:.2%}[/{color}]  "
            f"[dim]{hit['file_name']}[/dim]"
        )
        preview = hit["text"][:400] + ("..." if len(hit["text"]) > 400 else "")
        console.print(Panel(preview, title=header, border_style="dim"))


def cmd_keyword(kb: KnowledgeBase, query: str, top_k: int):
    """精确关键词搜索（适合查找特定数字、名称）。"""
    console.print(f"\n[cyan]关键词搜索:[/cyan] [bold]{query}[/bold]\n")
    results = kb.keyword_search(query, top_k=top_k)

    if not results:
        console.print("[yellow]未找到包含该关键词的照片。[/yellow]")
        return

    table = Table(title=f'包含 "{query}" 的照片', box=box.ROUNDED, show_lines=True)
    table.add_column("文件名", style="cyan", max_width=30)
    table.add_column("文字预览", max_width=70)

    for hit in results:
        # Highlight the query term in preview
        text = hit["text"]
        idx = text.find(query)
        if idx >= 0:
            start = max(0, idx - 60)
            end = min(len(text), idx + len(query) + 60)
            preview = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
        else:
            preview = text[:120]
        table.add_row(hit["file_name"], preview)

    console.print(table)


def cmd_show(kb: KnowledgeBase, name: str):
    """显示指定照片的完整提取文字。"""
    row = kb.get_photo_by_name(name)
    if not row:
        console.print(f"[red]数据库中未找到: {name}[/red]")
        return

    console.print(Panel(
        row["ocr_text"] or "[dim]该照片无文字内容[/dim]",
        title=f"[bold]{row['file_name']}[/bold]  [dim]{row['processed_at']}[/dim]",
        border_style="cyan",
    ))


def cmd_stats(kb: KnowledgeBase):
    """显示知识库统计信息。"""
    stats = kb.get_stats()
    table = Table(title="知识库统计", box=box.ROUNDED)
    table.add_column("指标", style="cyan")
    table.add_column("数值", style="green", justify="right")
    table.add_row("已处理照片总数", str(stats["total"] or 0))
    table.add_row("含文字的照片", str(stats["with_text"] or 0))
    table.add_row("无文字的照片", str((stats["total"] or 0) - (stats["with_text"] or 0)))
    table.add_row("处理错误", str(stats["errors"] or 0))
    table.add_row("已建立向量索引", str(stats["indexed"] or 0))
    table.add_row("累计消耗 Tokens", f"{stats['total_tokens'] or 0:,}")
    console.print(table)


def cmd_list(kb: KnowledgeBase, has_text: bool, limit: int):
    """列出已处理的照片。"""
    where = "WHERE has_text = 1" if has_text else ""
    cursor = kb.sqlite_conn.execute(
        f"SELECT file_name, has_text, tokens_used, processed_at "
        f"FROM photos {where} ORDER BY processed_at DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()

    if not rows:
        console.print("[yellow]暂无记录。[/yellow]")
        return

    table = Table(
        title="照片列表" + ("（仅含文字）" if has_text else ""),
        box=box.ROUNDED,
    )
    table.add_column("文件名")
    table.add_column("含文字", justify="center")
    table.add_column("Tokens", justify="right")
    table.add_column("处理时间")

    for row in rows:
        table.add_row(
            row["file_name"],
            "[green]是[/green]" if row["has_text"] else "[dim]否[/dim]",
            str(row["tokens_used"]),
            (row["processed_at"] or "")[:19],
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="搜索个人照片知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python query_kb.py search "会议记录"
  python query_kb.py search "项目计划" --top 10
  python query_kb.py keyword "2024年"
  python query_kb.py show IMG_1234.jpg
  python query_kb.py stats
  python query_kb.py list --has-text
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # search (semantic)
    p_search = subparsers.add_parser("search", help="语义向量搜索")
    p_search.add_argument("query", help="搜索关键词（支持中英文）")
    p_search.add_argument("--top", type=int, default=TOP_K_RESULTS, help="返回结果数量")

    # keyword (exact)
    p_kw = subparsers.add_parser("keyword", help="精确关键词搜索")
    p_kw.add_argument("query", help="要查找的关键词")
    p_kw.add_argument("--top", type=int, default=20, help="返回结果数量")

    # show
    p_show = subparsers.add_parser("show", help="显示指定照片的完整文字")
    p_show.add_argument("file", help="文件名或完整路径")

    # stats
    subparsers.add_parser("stats", help="显示知识库统计信息")

    # list
    p_list = subparsers.add_parser("list", help="列出已处理的照片")
    p_list.add_argument("--has-text", action="store_true", help="只显示含文字的照片")
    p_list.add_argument("--limit", type=int, default=50, help="最多显示条数")

    args = parser.parse_args()
    kb = KnowledgeBase()

    try:
        if args.command == "search":
            cmd_search(kb, args.query, args.top)
        elif args.command == "keyword":
            cmd_keyword(kb, args.query, args.top)
        elif args.command == "show":
            cmd_show(kb, args.file)
        elif args.command == "stats":
            cmd_stats(kb)
        elif args.command == "list":
            cmd_list(kb, args.has_text, args.limit)
    finally:
        kb.close()


if __name__ == "__main__":
    main()
