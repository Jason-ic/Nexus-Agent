"""Nexus CLI 入口"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from nexus.config import load_settings
from nexus.constraints.engine import ConstraintEngine
from nexus.context.dynamic import DynamicContext
from nexus.context.manager import ContextManager
from nexus.context.static import StaticContext
from nexus.core.pipeline import Pipeline
from nexus.core.types import Message, NexusRequest
from nexus.feedback.collector import FeedbackCollector
from nexus.feedback.store import FeedbackStore
from nexus.memory.search import MemorySearch
from nexus.memory.session import SessionManager
from nexus.memory.store import MemoryStore
from nexus.providers.anthropic import AnthropicProvider
from nexus.providers.ollama import OllamaProvider
from nexus.providers.openai import OpenAIProvider
from nexus.providers.registry import ProviderRegistry

cli_app = typer.Typer(help="Nexus - 让任何 AI 模型越用越好")
train_app = typer.Typer(help="训练相关命令")
console = Console()


def _setup():
    project_dir = Path.cwd()
    settings = load_settings(project_dir)
    nexus_dir = project_dir / settings.nexus_dir

    # Provider 注册
    reg = ProviderRegistry()
    for name, cfg in settings.providers.items():
        if name == "anthropic":
            reg.register(AnthropicProvider(api_key=cfg.api_key))
        else:
            provider = OpenAIProvider(api_key=cfg.api_key, base_url=cfg.base_url)
            provider.name = name
            reg.register(provider)
    reg.register(OllamaProvider(base_url=settings.ollama_base_url))

    # 上下文 + 记忆
    static = StaticContext(nexus_dir)
    dynamic = DynamicContext()
    memory_store = MemoryStore(nexus_dir / "memory")
    memory_search = MemorySearch(memory_store)
    session_mgr = SessionManager(memory_store)
    ctx_manager = ContextManager(
        static=static, dynamic=dynamic, memory_search=memory_search,
    )

    # 约束 + 反馈
    constraint_engine = ConstraintEngine()
    feedback_store = FeedbackStore(nexus_dir / "feedback.db")
    feedback_collector = FeedbackCollector(feedback_store)

    pipeline = Pipeline(
        settings=settings,
        registry=reg,
        context_manager=ctx_manager,
        session_manager=session_mgr,
        constraint_engine=constraint_engine,
        feedback_collector=feedback_collector,
    )
    return pipeline, settings, feedback_collector, feedback_store


@cli_app.command()
def chat(
    message: str = typer.Argument(None, help="发送的消息（留空进入交互模式）"),
    model: str = typer.Option(None, "--model", "-m", help="模型，如 openai/gpt-4o"),
):
    """与 AI 模型对话"""
    pipeline, settings, fb_collector, _ = _setup()
    model = model or settings.default_model

    if message:
        response = asyncio.run(pipeline.run(NexusRequest(
            messages=[Message(role="user", content=message)],
            model=model,
        )))
        console.print(Markdown(response.content))
        return

    # 交互模式
    console.print(f"[bold]Nexus[/bold] · model: [cyan]{model}[/cyan]")
    console.print("[dim]/exit 退出 · /good 👍 · /bad 👎 · /why 文字反馈[/dim]\n")
    history: list[Message] = []

    while True:
        try:
            user_input = console.input("[bold green]> [/bold green]")
        except (EOFError, KeyboardInterrupt):
            break

        stripped = user_input.strip()
        if stripped in ("/exit", "/quit"):
            break
        if stripped == "/good":
            if fb_collector.rate("positive"):
                console.print("[green]👍 已记录正面反馈[/green]")
            continue
        if stripped == "/bad":
            if fb_collector.rate("negative"):
                console.print("[red]👎 已记录负面反馈[/red]")
            continue
        if stripped.startswith("/why "):
            feedback_text = stripped[5:]
            if fb_collector.rate("negative", feedback_text):
                console.print(f"[yellow]📝 已记录反馈: {feedback_text}[/yellow]")
            continue
        if not stripped:
            continue

        history.append(Message(role="user", content=user_input))

        try:
            response = asyncio.run(pipeline.run(NexusRequest(
                messages=history, model=model,
            )))
            history.append(Message(role="assistant", content=response.content))
            console.print()
            console.print(Markdown(response.content))
            console.print()
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@cli_app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="监听地址"),
    port: int = typer.Option(8000, help="端口"),
):
    """启动 API 服务"""
    import uvicorn
    console.print(f"[bold]Nexus API[/bold] starting on [cyan]http://{host}:{port}[/cyan]")
    uvicorn.run("nexus.app:app", host=host, port=port, reload=True)


@cli_app.command()
def memory(
    action: str = typer.Argument(help="操作: search / show / save"),
    query: str = typer.Argument(None, help="搜索关键词或要保存的内容"),
):
    """管理记忆"""
    project_dir = Path.cwd()
    settings = load_settings(project_dir)
    nexus_dir = project_dir / settings.nexus_dir
    store = MemoryStore(nexus_dir / "memory")

    if action == "search":
        if not query:
            console.print("[red]请提供搜索关键词[/red]")
            return
        search = MemorySearch(store)
        results = search.search(query)
        if not results:
            console.print("[dim]未找到相关记忆[/dim]")
            return
        for r in results:
            console.print(f"[cyan]{r.source}[/cyan]")
            console.print(r.content[:300])
            console.print()

    elif action == "show":
        content = store.read_long_term()
        if content:
            console.print(Markdown(content))
        else:
            console.print("[dim]长期记忆为空[/dim]")

    elif action == "save":
        if not query:
            console.print("[red]请提供要保存的内容[/red]")
            return
        store.append_long_term(query)
        console.print("[green]已保存到长期记忆[/green]")

    else:
        console.print(f"[red]未知操作: {action}。可用: search / show / save[/red]")


@cli_app.command()
def feedback(
    action: str = typer.Argument(help="操作: stats / export"),
):
    """查看反馈统计"""
    project_dir = Path.cwd()
    settings = load_settings(project_dir)
    nexus_dir = project_dir / settings.nexus_dir
    store = FeedbackStore(nexus_dir / "feedback.db")

    if action == "stats":
        stats = store.get_stats()
        table = Table(title="反馈统计")
        table.add_column("类型", style="bold")
        table.add_column("数量", justify="right")
        table.add_row("总交互", str(stats["total"]))
        table.add_row("👍 正面", f"[green]{stats['positive']}[/green]")
        table.add_row("👎 负面", f"[red]{stats['negative']}[/red]")
        table.add_row("⚪ 未评", str(stats["neutral"]))
        console.print(table)

    elif action == "export":
        samples = store.get_positive_samples()
        if not samples:
            console.print("[dim]暂无正样本数据[/dim]")
            return
        console.print(f"[green]共 {len(samples)} 条正样本可用于训练[/green]")
        for s in samples[:5]:
            console.print(f"[dim]{s['timestamp']}[/dim] [{s['model']}]")
            console.print(f"  User: {s['user_message'][:80]}")
            console.print(f"  Assistant: {s['assistant_message'][:80]}")
            console.print()

    else:
        console.print(f"[red]未知操作: {action}。可用: stats / export[/red]")


@cli_app.command()
def train(
    base: str = typer.Option("Qwen/Qwen2.5-3B-Instruct", "--base", "-b", help="基础模型"),
    data: str = typer.Option(None, "--data", "-d", help="训练数据路径（JSONL）"),
    output: str = typer.Option(None, "--output", "-o", help="输出目录"),
    rank: int = typer.Option(8, "--rank", help="LoRA rank"),
    epochs: int = typer.Option(3, "--epochs", help="训练轮数"),
    auto: bool = typer.Option(False, "--auto", help="自动循环：导出→微调→评测→保留/回滚"),
):
    """LoRA 微调本地模型"""
    from nexus.training.exporter import TrainingExporter
    from nexus.training.lora import LoRAConfig, train_lora

    project_dir = Path.cwd()
    settings = load_settings(project_dir)
    nexus_dir = project_dir / settings.nexus_dir

    # 如果没有指定数据路径，自动从 feedback store 导出
    if data is None:
        data_path = nexus_dir / "training" / "sft_data.jsonl"
        feedback_store = FeedbackStore(nexus_dir / "feedback.db")
        exporter = TrainingExporter(feedback_store)
        count = exporter.export_sft(data_path)
        if count == 0:
            console.print("[red]没有正样本数据可用于训练。请先使用 /good 标记好的回复。[/red]")
            return
        console.print(f"[green]已导出 {count} 条正样本到 {data_path}[/green]")
    else:
        data_path = Path(data)
        if not data_path.exists():
            console.print(f"[red]数据文件不存在: {data_path}[/red]")
            return

    output_dir = output or str(nexus_dir / "training" / "checkpoints")

    config = LoRAConfig(
        base_model=base,
        data_path=str(data_path),
        output_dir=output_dir,
        lora_rank=rank,
        num_epochs=epochs,
    )

    console.print(f"[bold]开始 LoRA 微调[/bold]")
    console.print(f"  基础模型: [cyan]{base}[/cyan]")
    console.print(f"  数据: [cyan]{data_path}[/cyan]")
    console.print(f"  LoRA rank: [cyan]{rank}[/cyan]")
    console.print(f"  轮数: [cyan]{epochs}[/cyan]")
    console.print()

    try:
        final_path = train_lora(config)
        console.print(f"\n[green]训练完成！Adapter 保存在: {final_path}[/green]")
        console.print("[dim]使用 'nexus train-register' 将模型注册到 Ollama[/dim]")
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
    except Exception as e:
        console.print(f"[red]训练失败: {e}[/red]")


@cli_app.command(name="train-export")
def train_export(
    format: str = typer.Option("sft", "--format", "-f", help="导出格式: sft / dpo"),
    output: str = typer.Option(None, "--output", "-o", help="输出路径"),
    limit: int = typer.Option(1000, "--limit", help="最大导出数量"),
):
    """从反馈数据导出训练集"""
    from nexus.training.exporter import TrainingExporter

    project_dir = Path.cwd()
    settings = load_settings(project_dir)
    nexus_dir = project_dir / settings.nexus_dir
    feedback_store = FeedbackStore(nexus_dir / "feedback.db")
    exporter = TrainingExporter(feedback_store)

    if output:
        out_path = Path(output)
    else:
        out_path = nexus_dir / "training" / f"{format}_data.jsonl"

    if format == "sft":
        count = exporter.export_sft(out_path, limit=limit)
    elif format == "dpo":
        count = exporter.export_dpo(out_path, limit=limit)
    else:
        console.print(f"[red]未知格式: {format}。可用: sft / dpo[/red]")
        return

    if count == 0:
        console.print("[yellow]没有可导出的数据[/yellow]")
    else:
        console.print(f"[green]已导出 {count} 条 {format.upper()} 数据到 {out_path}[/green]")


@cli_app.command(name="train-register")
def train_register(
    model_path: str = typer.Argument(help="合并后的模型路径"),
    name: str = typer.Option(None, "--name", "-n", help="Ollama 模型名称"),
    quantize: str = typer.Option("", "--quantize", "-q", help="量化方式: q4_0, q8_0"),
):
    """将微调后的模型注册到 Ollama"""
    from nexus.training.registry import register_to_ollama

    model_name = name or f"nexus-{Path(model_path).parent.name}"

    console.print(f"[bold]注册模型到 Ollama[/bold]: [cyan]{model_name}[/cyan]")
    try:
        registered = register_to_ollama(model_path, model_name, quantize=quantize)
        console.print(f"[green]注册成功！使用: nexus chat --model ollama/{registered}[/green]")
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")


@cli_app.command(name="train-merge")
def train_merge(
    base: str = typer.Argument(help="基础模型路径或名称"),
    adapter: str = typer.Argument(help="LoRA adapter 路径"),
    output: str = typer.Argument(help="合并输出路径"),
):
    """将 LoRA adapter 合并回基础模型"""
    from nexus.training.lora import merge_adapter

    console.print(f"[bold]合并 LoRA adapter[/bold]")
    console.print(f"  基础模型: [cyan]{base}[/cyan]")
    console.print(f"  Adapter: [cyan]{adapter}[/cyan]")
    console.print(f"  输出: [cyan]{output}[/cyan]")

    try:
        out_path = merge_adapter(base, adapter, output)
        console.print(f"\n[green]合并完成！模型保存在: {out_path}[/green]")
        console.print("[dim]使用 'nexus train-register' 将模型注册到 Ollama[/dim]")
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")


def main():
    cli_app()


if __name__ == "__main__":
    main()
