# 银发甄选 (SilverSelect)

专为银发经济设计的智能选品与供应链撮合系统。基于LangGraph构建AI Agent，集成数据分析和供应链管理能力，帮助银发主播进行农特产品科学选品和供应商对接。

<!-- more -->

## 项目特点

- **面向项目的提示词体系**：将智能体的系统提示词重构为"数据参谋 + 供应经理"的双角色定位，明确十项以内的核心指标口径（如场均观看、活跃粉丝、观看/曝光 GPM、转化率、客单价、退货率、地域与品类命中度、7 日内首单），并约定适老化输出格式、标准工作流（取数→建模→建议→撮合→出图）与禁止事项（不做非 SELECT 操作、不输出敏感信息）。该设计让"方法如何落到执行"一目了然，减少沟通成本。
- **安全闸口**：为防止误操作数据库，sql_inter 与 extract_data 增加只读校验，仅允许执行 SELECT；所有取数与抽取均以此为前置条件，确保数据层面的可追溯与安全性。
- **定制化智能体**：基于 [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui) 构建的定制化前端界面
- **适老化呈现规范**：在提示词与工具输出层统一"结论先行、三行短句、最少专业词"的表达方式，配合信号灯语义，降低阅读与决策门槛，确保银发主播与非技术评审均能快速抓住重点。

## 核心功能

### 1. 数据查询与分析
- 支持安全的 SQL SELECT 查询
- 数据提取至 Pandas DataFrame 进行进一步分析
- Python 代码执行环境，支持自定义分析逻辑

### 2. SilverRank 竞争力评分
基于熵权法和 TOPSIS 方法，对商品或供应商进行综合评分：
- 支持正向指标和负向指标的混合评估
- 自动生成权重分配
- 输出排名结果和详细权重信息

**竞争力模型工具（silverrank_score）**：内置"熵权 + TOPSIS"的 SilverRank 评分流程，支持正向与逆向指标的混合建模；输出包含指标权重表、Top-K 排名与各指标的 min/max 口径，便于解释与复核。该工具将多维运营数据压缩为可比较的综合分，服务后续的选品与分层运营决策。

### 3. 智能建议生成
根据数据分析结果，自动生成可执行的业务建议：
- 主推品类建议
- 价格带优化建议
- 上新策略和脚本要点

**建议卡工具（suggest_cards）**：以"三条短句 + 信号灯（🟢/🟡/🔴）"为统一呈现规范，自动生成本周可执行建议（如主推占比、价格带、上新与脚本要点），既适配 PPT 展示，也适配小程序"建议卡"组件。逻辑层预留扩展钩子，可按实际库字段替换命中度与阈值，保证口径可进化。

### 4. 供应商智能撮合
基于白名单机制和多维度评估的供应商匹配：
- 地区和品类筛选
- 价格区间匹配
- 信誉、时效和退货率综合评估

**供应商撮合工具（match_suppliers）**：基于地域、品类、价格带与信誉评分进行多维筛选与排序，支持白名单过滤，并对发货时效、历史退货率给出履约预警；返回 Top-N 可对接清单，可直接用于业务侧的联系和落单，提高"人货场"匹配效率。

### 5. 数据可视化
支持生成各类图表用于数据展示和分析：
- Matplotlib 和 Seaborn 图表生成
- 自动保存至指定目录

**绘图落盘可移植**：绘图工具支持环境变量 IMAGES_DIR，默认写入 ./images，不再依赖硬编码路径，便于前后端解耦与多环境部署；常用于输出"指标总览、前后对比"等可视化证据。

## 技术架构

### 后端
- [LangGraph](https://github.com/langchain-ai/langgraph) 作为核心框架
- DeepSeek 大模型作为推理引擎
- Tavily 搜索工具用于网络搜索
- MySQL 作为数据存储
- Pandas、Matplotlib、Seaborn 用于数据分析和可视化

### 前端
- 基于 [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui) 定制开发的聊天界面
- Next.js 构建的现代化 Web 应用
- Tailwind CSS 样式框架

## 快速开始

### 环境要求
- Python 3.8+
- Node.js 16+
- MySQL 数据库

### 安装步骤

1. 克隆项目仓库
```bash
git clone <repository-url>
cd data_agent
```

2. 安装 Python 依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
创建 `.env` 文件并配置以下参数：
```env
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key
HOST=your_mysql_host
USER=your_mysql_user
MYSQL_PW=your_mysql_password
DB_NAME=your_database_name
PORT=your_mysql_port
```

4. 安装前端依赖
```bash
cd agent-chat-ui
pnpm install
```

### 运行项目

1. 启动 LangGraph 服务
```bash
langgraph up
```

2. 启动前端界面
```bash
cd agent-chat-ui
pnpm dev
```

访问 `http://localhost:3000` 使用银发甄选系统。

## 使用示例

银发甄选支持多种操作指令：

1. 数据查询：
   ```
   请查询最近一周的商品销售数据
   ```

2. 竞争力评分：
   ```
   请计算当前商品的竞争力评分
   ```

3. 生成建议：
   ```
   请为下周提供选品建议
   ```

4. 供应商匹配：
   ```
   请为华南地区的果蔬生鲜类商品匹配供应商
   ```

## 致谢

特别感谢 [Bilibili 视频 BV1Nx38zMEyi](https://www.bilibili.com/video/BV1Nx38zMEyi/?spm_id_from=333.1007.top_right_bar_window_default_collection.content.click&vd_source=94ea7868838952c693d22d68de4e9dad) 提供的思路启发和部分代码参考，我们在其基础上进行了定制化开发和功能增强。

本项目作者：[Jackson ZHOU](https://github.com/Jackksonns)

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件。