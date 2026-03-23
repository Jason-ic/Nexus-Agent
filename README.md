# Nexus

让任何 AI 模型越用越好的开源 Agent 平台。

短期靠记忆和上下文，长期靠训练优化权重。

## 架构

```
┌─────────────────────────────────────────────────────┐
│  Layer 3: 训练层（让模型大脑变强）                     │
│  经验数据 → LoRA 微调 → 权重更新 → 更强的模型          │
├─────────────────────────────────────────────────────┤
│  Layer 2: 记忆层（让模型记住经验）                     │
│  文件记忆 + 语义检索 + 会话管理 + 自动归档             │
├─────────────────────────────────────────────────────┤
│  Layer 1: Harness 层（让模型当下表现更好）             │
│  多模型接口 + 上下文工程 + 约束验证 + 反馈收集         │
└─────────────────────────────────────────────────────┘
```

## 快速开始

```bash
# 安装
pip install -e .

# 配置
cp .nexus/config.yaml.example .nexus/config.yaml
# 编辑 .nexus/config.yaml 填入你的 API Key

# 对话
nexus chat --model moonshot/kimi-k2.5 "你好"
nexus chat  # 交互模式
```

## 项目结构

```
nexus/
├── cli.py                    # CLI 入口
├── app.py                    # FastAPI API 服务
├── config.py                 # 配置加载
├── core/
│   ├── types.py              # 共享数据模型
│   └── pipeline.py           # 请求管线编排
├── providers/                # 多模型统一接口
│   ├── base.py               # ModelProvider Protocol
│   ├── registry.py           # provider/model 路由
│   ├── openai.py             # OpenAI / 兼容 API
│   ├── anthropic.py          # Claude 系列
│   └── ollama.py             # 本地模型
├── context/                  # 上下文工程
│   ├── manager.py            # token 预算感知组装
│   ├── static.py             # 静态规则 (.nexus/rules.md)
│   └── dynamic.py            # 对话历史滑动窗口
├── memory/                   # 持久记忆
│   ├── store.py              # Markdown 文件存储
│   ├── search.py             # SQLite FTS5 全文检索
│   └── session.py            # 会话管理 + 每日日志
├── constraints/              # 约束系统
│   ├── engine.py             # 约束验证引擎
│   ├── base.py               # Rule Protocol
│   └── rules.py              # 内置规则
├── feedback/                 # 反馈收集
│   ├── collector.py          # 用户评分 (+1/-1/文本)
│   ├── evaluator.py          # 自动评估
│   └── store.py              # SQLite 持久化
└── training/                 # 本地训练
    ├── exporter.py           # 导出 SFT/DPO 数据
    ├── lora.py               # LoRA 微调
    └── registry.py           # 模型注册到 Ollama
```

## 支持的模型

| Provider | 模型示例 | 类型 |
|----------|---------|------|
| OpenAI | `openai/gpt-4o` | API |
| Anthropic | `anthropic/claude-sonnet-4-6` | API |
| Moonshot | `moonshot/kimi-k2.5` | API |
| Ollama | `ollama/qwen3`, `ollama/llama3` | 本地 |

任何 OpenAI 兼容 API 都可通过配置 `base_url` 接入。

## 配置

创建 `.nexus/config.yaml`：

```yaml
default_model: moonshot/kimi-k2.5

providers:
  moonshot:
    api_key: your-api-key
    base_url: https://api.moonshot.cn/v1
  openai:
    api_key: your-openai-key

# ollama_base_url: http://localhost:11434
```

## CLI 命令

### 对话

```bash
nexus chat "你好"                          # 单次对话
nexus chat                                 # 交互模式
nexus chat --model openai/gpt-4o "你好"    # 指定模型
```

交互模式中支持：
- `/good` — 标记上一条回复为正面反馈
- `/bad` — 标记为负面反馈
- `/why <原因>` — 附带文字说明的负面反馈
- `/exit` — 退出

### 记忆管理

```bash
nexus memory show                # 查看长期记忆
nexus memory search "关键词"     # 搜索记忆
nexus memory save "重要信息"     # 手动保存
```

### 反馈统计

```bash
nexus feedback stats             # 查看反馈统计
nexus feedback export            # 查看正样本
```

### 训练（LoRA 微调）

```bash
# 导出训练数据
nexus train-export                         # 导出 SFT 数据
nexus train-export --format dpo            # 导出 DPO 数据

# 微调（自动导出正样本 + LoRA 训练）
nexus train --base Qwen/Qwen2.5-3B-Instruct
nexus train --base Qwen/Qwen2.5-3B-Instruct --rank 16 --epochs 5

# 合并 adapter 到基础模型
nexus train-merge Qwen/Qwen2.5-3B-Instruct .nexus/training/checkpoints/final ./merged

# 注册到 Ollama
nexus train-register ./merged --name nexus-qwen3

# 使用微调后的模型
nexus chat --model ollama/nexus-qwen3
```

### API 服务

```bash
nexus serve                      # 启动 API 服务
nexus serve --port 9000          # 指定端口

# 调用
curl -X POST http://localhost:8000/v1/complete \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}]}'
```

## 上下文工程

在项目目录创建 `.nexus/rules.md`，内容会自动注入每次对话的 system prompt：

```markdown
# 项目规则
- 使用中文回复
- 代码风格遵循 PEP 8
```

## 数据流

```
用户请求 → 配置解析(选择 provider/model)
         → 上下文组装(静态规则 + 记忆检索 + 对话历史)
         → 模型调用
         → 约束检查(失败则带错误信息重试，最多 2 次)
         → 反馈收集(SQLite 持久化)
         → 记忆保存(每日日志 + 长期记忆)
         → 返回响应
```

## 致谢

- [Harness Engineering](https://www.nxcode.io/resources/news/harness-engineering-complete-guide-ai-agent-codex-2026) — 上下文工程理论框架
- [OpenClaw](https://github.com/openclaw/openclaw) — 文件优先记忆机制
- [OpenClaw-RL](https://github.com/OpenClaw-RL/OpenClaw-RL) — RL 训练管线
- [Autoresearch](https://github.com/karpathy/autoresearch) — 棘轮机制

## License

MIT
