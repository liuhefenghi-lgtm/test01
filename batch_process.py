#!/usr/bin/env python3
"""
batch_process.py - Process a directory of photos and build the knowledge base.

Usage:
    python batch_process.py /path/to/photos
    python batch_process.py /path/to/photos --resume
    python batch_process.py /path/to/photos --dry-run
    python batch_process.py /path/to/photos --limit 10
"""
import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from knowledge_base import KnowledgeBase
from ocr_processor import OCRProcessor
from utils import collect_images

console = Console()


def print_summary(kb: KnowledgeBase, elapsed: float):
    stats = kb.get_stats()
    table = Table(title="Processing Complete", show_header=True)
    table.add_column("指标", style="cyan")
    table.add_column("数值", style="green", justify="right")
    table.add_row("已处理照片数", str(stats["total"] or 0))
    table.add_row("含文字的照片", str(stats["with_text"] or 0))
    table.add_row("无文字的照片", str((stats["total"] or 0) - (stats["with_text"] or 0)))
    table.add_row("处理错误", str(stats["errors"] or 0))
    table.add_row("已建立向量索引", str(stats["indexed"] or 0))
    table.add_row("累计消耗 Tokens", f"{stats['total_tokens'] or 0:,}")
    cost = (stats["total_tokens"] or 0) / 1_000_000 * 0.80
    table.add_row("预计 API 费用", f"~${cost:.3f} USD")
    table.add_row("耗时", f"{elapsed:.1f}s")
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="批量处理照片，提取文字，建立知识库"
    )
    parser.add_argument("photo_dir", type=Path, help="照片所在目录")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过已处理的文件（根据数据库状态判断）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计文件数量，不实际处理",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只处理前 N 张（用于测试）",
    )
    args = parser.parse_args()

    if not args.photo_dir.exists():
        console.print(f"[red]目录不存在: {args.photo_dir}[/red]")
        sys.exit(1)

    console.print(f"[cyan]扫描目录: {args.photo_dir}[/cyan]")
    images = collect_images(args.photo_dir)

    if not images:
        console.print("[yellow]未找到支持的图片文件。[/yellow]")
        return

    if args.limit > 0:
        images = images[: args.limit]

    console.print(f"找到 [bold]{len(images)}[/bold] 张图片")

    if args.dry_run:
        console.print("[yellow]演习模式，未处理任何文件。[/yellow]")
        return

    kb = KnowledgeBase()
    ocr = OCRProcessor()
    start_time = time.monotonic()

    # Filter already-processed files when resuming
    if args.resume:
        original_count = len(images)
        images = [img for img in images if not kb.is_already_processed(img)]
        skipped = original_count - len(images)
        if skipped:
            console.print(
                f"[yellow]续传模式：跳过已处理 {skipped} 张，剩余 {len(images)} 张[/yellow]"
            )

    if not images:
        console.print("[bold green]所有图片均已处理完毕！[/bold green]")
        print_summary(kb, time.monotonic() - start_time)
        kb.close()
        return

    console.print(f"开始处理 [bold]{len(images)}[/bold] 张图片...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("处理照片中...", total=len(images))

        for image_path in images:
            progress.update(
                task,
                description=f"[cyan]{image_path.name[:40]}[/cyan]",
            )

            # Run OCR
            result = ocr.extract_text(image_path)

            # Save to SQLite
            row_id = kb.save_ocr_result(
                file_path=image_path,
                ocr_text=result["text"],
                tokens_used=result["tokens_used"],
                error=result["error"],
            )

            # Index in ChromaDB if text was extracted
            if result["text"] and not result["error"]:
                kb.index_in_chroma(row_id, image_path, result["text"])

            progress.advance(task)

    elapsed = time.monotonic() - start_time
    print_summary(kb, elapsed)
    kb.close()


if __name__ == "__main__":
    main()
