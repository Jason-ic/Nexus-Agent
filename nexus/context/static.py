"""静态上下文 — 从 .nexus/ 目录加载项目规则和文档"""

from __future__ import annotations

from pathlib import Path


class StaticContext:
    def __init__(self, nexus_dir: Path):
        self._nexus_dir = nexus_dir

    def load_rules(self) -> str | None:
        """加载 .nexus/rules.md"""
        rules_file = self._nexus_dir / "rules.md"
        if rules_file.exists():
            return rules_file.read_text(encoding="utf-8").strip()
        return None

    def load_all_docs(self) -> list[tuple[str, str]]:
        """加载 .nexus/docs/ 目录下的所有 markdown 文件"""
        docs_dir = self._nexus_dir / "docs"
        if not docs_dir.exists():
            return []
        results = []
        for f in sorted(docs_dir.glob("*.md")):
            content = f.read_text(encoding="utf-8").strip()
            if content:
                results.append((f.name, content))
        return results

    def assemble(self) -> str:
        """组装所有静态上下文为一个字符串"""
        parts = []

        rules = self.load_rules()
        if rules:
            parts.append(f"# Project Rules\n\n{rules}")

        for name, content in self.load_all_docs():
            parts.append(f"# {name}\n\n{content}")

        return "\n\n---\n\n".join(parts)
