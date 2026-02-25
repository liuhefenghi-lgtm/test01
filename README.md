# 照片文字提取知识库

从电脑照片中批量提取文字，建立个人知识库，支持中英文语义搜索和 AI 总结。

## 功能特点

- **批量 OCR**：使用 Claude Vision API 处理照片，对中文识别效果优秀
- **断点续传**：中断后重新运行自动跳过已处理的照片
- **语义搜索**：支持中英文自然语言搜索（"会议记录"、"项目计划"等）
- **精确搜索**：关键词精确匹配（适合查找特定数字、人名）
- **AI 总结**：按主题生成结构化总结报告

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 首次安装较慢（约 2-3GB，包含 PyTorch 和语言模型）。
> 若只需 CPU 版 PyTorch：`pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 Anthropic API Key
```

在 [console.anthropic.com](https://console.anthropic.com/) 获取 API Key。

### 3. 处理照片

```bash
# 先演习一下，确认文件数量
python batch_process.py ~/Pictures --dry-run

# 先处理 5 张测试效果
python batch_process.py ~/Pictures --limit 5

# 处理全部照片（约 15-25 分钟处理 500 张）
python batch_process.py ~/Pictures

# 中断后续传（自动跳过已处理的）
python batch_process.py ~/Pictures --resume
```

### 4. 搜索知识库

```bash
# 语义搜索（自然语言，支持中英文）
python query_kb.py search "会议记录"
python query_kb.py search "项目截止日期" --top 10

# 精确关键词搜索
python query_kb.py keyword "2024年"
python query_kb.py keyword "张三"

# 查看指定照片的完整文字
python query_kb.py show IMG_1234.jpg

# 查看统计信息
python query_kb.py stats

# 列出所有含文字的照片
python query_kb.py list --has-text
```

### 5. AI 总结

```bash
# 总结所有内容
python summarize.py

# 总结特定主题（先语义搜索，再 AI 总结）
python summarize.py --query "工作笔记"
python summarize.py --query "财务记录" --top 30

# 保存总结到文件
python summarize.py --output summary.txt
python summarize.py --query "会议记录" --output meetings_summary.txt
```

## 项目结构

```
.
├── batch_process.py    # 批量处理照片（主入口）
├── query_kb.py         # 搜索知识库
├── summarize.py        # AI 总结
├── ocr_processor.py    # Claude Vision OCR 核心
├── knowledge_base.py   # SQLite + ChromaDB 存储层
├── config.py           # 配置（模型、路径、参数）
├── utils.py            # 图片预处理工具函数
├── requirements.txt    # Python 依赖
├── .env.example        # API Key 配置模板
└── data/               # 自动创建
    ├── photos.db       # SQLite 数据库（原始文字+元数据）
    └── chroma_db/      # ChromaDB 向量索引
```

## 费用估算

使用 `claude-haiku-4-5` 模型（性价比最优）：

| 照片数量 | 预计费用 | 预计时间 |
|----------|----------|----------|
| 100 张   | ~$0.20   | 约 5 分钟 |
| 500 张   | ~$0.80   | 约 20 分钟 |
| 1000 张  | ~$1.50   | 约 40 分钟 |

搜索和总结不消耗 OCR token，总结费用约 $0.01/次。

## 支持的图片格式

JPG、JPEG、PNG、GIF、WebP
