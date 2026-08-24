# 中国 A 股多因子中性化投资组合

一个面向 CSI 300 股票池的多因子选股、动态因子加权和风险约束组合优化项目。

项目从传统的“因子暴露打分 + 等权买入”升级为完整的量化研究流程：框架支持在每个月末
构造 Size、Value、Momentum、Quality 和 Low Volatility 五类截面因子，依次完成去极值、
标准化、可用的行业与市值中性化，随后根据历史 ICIR 动态合成信号，最后在单股权重、
风格敞口、组合风险和交易成本约束下生成 long-only 投资组合。程序会识别数据中真正可用
的因子，并自动剔除缺失因子后重新归一化权重。

> 本项目用于量化研究、课程项目和作品集展示，不构成投资建议。`sample` 模式产生的是
> 人工模拟数据，所有模拟业绩只用于验证代码逻辑。

## 0. 当前真实数据版本

当前默认 AKShare 数据已缓存在 `data/akshare/`，运行时不需要联网：

- 数据源：AKShare 前复权（qfq）日线缓存；
- 当前 CSI 300 成分股：300 只，成分股快照日期为 2026-07-29；
- 行情区间：2021-01-04 至 2026-07-29；
- 有效行情记录：393,271 行；
- 市值字段覆盖率：97.24%；
- 当前可用因子：Size、Momentum、Low Volatility；
- Value、Quality：因缺少 point-in-time 基本面数据而自动停用；
- 行业覆盖率：0%，因此当前真实回测没有宣称实现有效的行业中性；
- 基准：当前 300 只成分股的等权收益代理，不是官方沪深 300 指数收益。

真实数据 ICIR 回测使用 2025-01-01 作为样本外起点，结果如下：

| 区间 | 累计收益 | 年化收益 | 年化波动 | 最大回撤 | Sharpe |
|---|---:|---:|---:|---:|---:|
| 全样本 | 68.56% | 12.30% | 12.75% | -12.74% | 0.96 |
| 样本内 | 62.23% | 18.04% | 13.20% | -8.20% | 1.37 |
| 样本外 | 3.91% | 2.45% | 11.75% | -12.74% | 0.21 |

策略全样本表现较好，但样本外收益明显下降，且相对等权基准的 Information Ratio 为
`-0.10`。因此这组结果不能被解读为稳定超额收益证据。当前成分股回溯还存在幸存者偏差，
正式研究报告必须同时披露 `outputs/akshare_icir/data_quality.csv` 中的限制。

## 1. 项目特点

- 五类经典股票因子，而非只使用 SMB/HML 回归暴露；
- MAD 去极值、截面标准化，以及数据允许时的行业/市值中性化；
- 月末生成信号、下一月持有，严格区分信号期和收益实现期；
- 使用 Rank IC、ICIR、t-stat 和分层收益检验因子有效性；
- 固定因子权重和滚动 ICIR 动态权重两种合成方式；
- 正则化 IC 相关矩阵，降低高度相关因子的重复配置；
- 均值—方差组合优化和协方差矩阵收缩；
- 单股、Size 因子敞口和数据允许时的行业主动权重约束；
- 显式计算佣金、滑点、换手率和交易成本；
- 支持样本内/样本外拆分、成本敏感性和参数敏感性分析；
- 自动导出净值、回撤、IC、动态权重、持仓和风险敞口报告；
- 提供确定性模拟数据，断网环境也能完成端到端验证。

## 2. 研究流程

```text
日频行情 + 基本面 + 行业分类
              │
              ▼
        月末截面原始因子
              │
              ▼
   MAD 去极值 → 中性化 → z-score
              │
              ▼
      历史 Rank IC / ICIR 监控
              │
              ▼
       固定权重或动态权重合成
              │
              ▼
 高分候选池 + 行业最低可行股票数
              │
              ▼
     风险、换手和敞口约束优化
              │
              ▼
     下一月收益、成本与绩效归因
```

回测的核心时序如下：

1. 在月末 `t` 使用当时可见的数据计算股票因子；
2. 使用截至 `t` 已经实现的历史 IC 决定因子权重；
3. 在月末 `t` 生成目标组合；
4. 使用 `t+1` 月收益计算组合表现；
5. `t` 月因子的 IC 在 `t+1` 月收益实现后才可用于后续调仓。

因此，当前月信号不会使用当前月之后的收益。

## 3. 因子定义

所有因子经过方向调整，处理后的数值越大代表预期收益越高。

### 3.1 Size

原始信号：

```text
Size_i,t = -log(Price_i,t × OutstandingShares_i,t)
```

负号使小市值股票具有更高的因子值。经济直觉是小市值股票可能包含流动性、经营不确定性
或投资者关注度较低所带来的风险补偿。

Size 因子只对行业虚拟变量做中性化，不再对市值本身回归，否则会机械地消除该因子。

### 3.2 Value

原始信号：

```text
BookToMarket_i,t = NetAssetPerShare_i,t / Price_i,t
Value_i,t = log(BookToMarket_i,t)
```

账面市值比越高，估值通常越低。代码会删除非正的账面市值比，再进行截面处理。

当前 AKShare 缓存没有按公告日对齐的历史每股净资产，因此真实数据运行会停用 Value，
而不是用价格反转等技术指标冒充估值因子。

### 3.3 Momentum

默认使用跳过最近一个月的 12 个月动量：

```text
Momentum_i,t = Price_i,t-1 / Price_i,t-12 - 1
```

跳过最近一个月是为了降低短期反转和微观交易噪声的影响。回看期和跳过期可通过
`momentum_lookback_months`、`momentum_skip_months` 修改。

### 3.4 Quality

原始信号：

```text
Quality_i,t = (ROE_i,t + OperatingCashflowToAssets_i,t) / 2
```

ROE 衡量权益资本的盈利效率，经营现金流/资产用于判断利润是否有现金流支持。如果真实数据
中缺少这两个字段，Quality 会从可用因子集合中停用，而不会因为缺失值删除全部股票或
占用组合权重。

### 3.5 Low Volatility

默认使用最近 60 个交易日收益率的标准差：

```text
Volatility_i,t = Std(DailyReturn_i,t-59:t)
LowVolatility_i,t = -Volatility_i,t
```

负号使波动率更低的股票具有更高的因子值。窗口长度可通过
`volatility_lookback_days` 修改。

## 4. 因子预处理

因子预处理在每个月末、每个因子上独立进行。

### 4.1 MAD 去极值

首先计算截面中位数和绝对中位差：

```text
MAD = median(|x_i - median(x)|)
RobustScale = 1.4826 × MAD
```

默认把原始值限制在：

```text
[median(x) - 3 × RobustScale, median(x) + 3 × RobustScale]
```

相比直接使用均值和标准差，MAD 对财务异常值和极端行情更加稳健。

### 4.2 行业与市值中性化

除 Size 外，其他因子执行以下截面回归：

```text
Factor_i,t = α_t
           + β_t × log(MarketCap_i,t)
           + Σ γ_k,t × IndustryDummy_i,k
           + ε_i,t
```

最终使用残差 `ε_i,t` 作为中性化因子值。这样可以减少组合收益实际来自行业轮动或大小盘
风格漂移的可能性。

如果行业字段全部为 `Unknown`，行业虚拟变量退化为常数项，程序仍能运行，但这种结果不应
被描述为行业中性。当前 AKShare 真实缓存正属于这种情况，相关事实会写入
`data_quality.csv`。

Size 因子执行：

```text
Size_i,t = α_t + Σ γ_k,t × IndustryDummy_i,k + ε_i,t
```

当样本数量不足以支持完整虚拟变量回归时，程序会退化为行业内去均值。

### 4.3 标准化

中性化残差转换为截面 z-score：

```text
Z_i,t = (ε_i,t - mean(ε_t)) / std(ε_t)
```

标准化使不同量纲的因子可以直接合成。

## 5. 因子有效性检验

### 5.1 Rank IC

每个月计算因子值与下一月股票收益之间的 Spearman 秩相关系数：

```text
IC_f,t = SpearmanCorr(FactorExposure_f,t, Return_t+1)
```

项目输出每个因子的：

- Mean IC；
- IC 标准差；
- `ICIR = Mean IC / IC Std`；
- IC 均值的 t-stat；
- IC 为正的月份比例；
- 有效观测月份数；
- 6 个月滚动 IC 曲线。

### 5.2 五分层收益

每个月按单个因子从低到高分为 Q1–Q5，统计下一月各层平均收益。理想情况下，正向因子应
表现出相对单调的分层收益，且 Q5 收益高于 Q1。

### 5.3 因子模拟收益

对每个因子计算高 20% 股票平均收益减去低 20% 股票平均收益，用于观察因子多空收益的
时间序列。该序列用于诊断，不等同于最终 long-only 策略收益。

## 6. 因子合成

项目支持两种方式。

### 6.1 固定权重

默认固定权重定义在 `BacktestConfig.factor_weights`：

| 因子 | 默认权重 |
|---|---:|
| Size | 0.15 |
| Value | 0.25 |
| Momentum | 0.25 |
| Quality | 0.20 |
| Low Volatility | 0.15 |

权重按绝对值之和归一化后使用。

在数据缺失导致部分因子不可计算时，固定权重和 ICIR 权重都只在可用因子之间重新归一化。
当前真实数据运行的有效因子为 Size、Momentum、Low Volatility。

### 6.2 动态 ICIR 权重

ICIR 模式默认读取过去 24 个月、且在调仓时已经实现的 Rank IC：

```text
RawWeight_f,t = Mean(IC_f) / Std(IC_f)
```

为防止短历史权重过度波动：

- 至少需要 12 个月有效历史；
- 单因子原始 ICIR 限制在 `[-3, 3]`；
- 历史不足时回退到固定权重；
- 使用加正则项的 IC 相关矩阵调整重复因子；
- 最终使因子权重绝对值之和等于 1。

动态权重可以为负。如果某因子在历史窗口内持续表现为反向 IC，模型允许对该因子进行反向
配置。

股票综合得分为：

```text
Score_i,t = Σ Weight_f,t × StandardizedExposure_i,f,t
```

## 7. 候选池与组合优化

### 7.1 候选池

程序首先选取综合得分最高的 `top_quantile` 股票。为了保证行业约束可行，还会为每个行业
补充达到行业权重下限所需的最低股票数量。

如果首轮候选池仍无法找到可行解，程序会扩大到整个可用股票池再次优化。两次优化都失败
时才使用确定性的等权回退组合，并把失败原因写入 `optimizer_status.csv`。

### 7.2 风险模型

默认使用过去 24 个月股票月收益估计协方差矩阵并年化。为降低短样本协方差矩阵不稳定问题，
使用简单对角收缩：

```text
Σ_shrunk = 0.60 × Σ_sample + 0.40 × diag(Σ_sample)
```

### 7.3 优化目标

优化器使用 SLSQP 求解下列 long-only 问题：

```text
min_w  -0.01 × Score' w
       + RiskAversion × w' Σ w
       + TurnoverPenalty × CostRate × ||w - w_previous||_1
```

代码用平滑绝对值近似换手项，使目标函数便于数值优化。

默认约束：

```text
Σ w_i = 1
0 ≤ w_i ≤ 10%
|PortfolioIndustryWeight_k - UniverseIndustryWeight_k| ≤ 5%
|PortfolioSizeExposure| ≤ 0.15
```

行业基准是当前可用股票池的等权行业分布，而不是 CSI 300 官方自由流通市值行业权重。
如果拥有正式基准权重，可以进一步把这里替换成真实指数行业权重。

只有行业覆盖率足够时，这项约束才具有经济含义。当前真实缓存行业均为 `Unknown`，因此
实际生效的是单股上限、Size 敞口、风险和换手约束，行业约束不提供额外风险控制。

## 8. 回测与交易成本

默认每月调仓。单边换手率定义为：

```text
Turnover_t = Σ |w_i,t - w_i,t-1|
```

首期组合从现金建立，因此首期换手率通常为 1。默认成本包括：

| 成本 | 默认值 |
|---|---:|
| 佣金 | 3 bps |
| 滑点 | 10 bps |
| 合计 | 13 bps |

净收益计算：

```text
NetReturn_t = GrossReturn_t - Turnover_t × (CommissionRate + SlippageRate)
```

这是简化的线性成本模型，暂未单独建模印花税、涨跌停、停牌、冲击成本和成交量容量。

## 9. 绩效与稳健性分析

报告包含：

- 累计收益和年化收益；
- 年化波动率；
- 最大回撤；
- Sharpe Ratio 和 Sortino Ratio；
- 相对基准的 Information Ratio；
- 月度胜率；
- 对 CSI 300 基准回归得到的年化 Alpha 和 Beta；
- 分年度绩效；
- 自定义样本内/样本外绩效；
- 0、5、13、20、50 bps 成本敏感性；
- 10%、20%、30% 选股比例敏感性；
- 3%、5%、10% 行业偏离约束敏感性。

参数敏感性需要重复运行组合优化，因此默认关闭，通过 `--run-sensitivity` 开启。

## 10. 安装与运行

### 10.1 环境

建议使用 Python 3.10 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

主要依赖：

- pandas、NumPy：数据处理；
- statsmodels：截面中性化和绩效回归；
- SciPy：约束组合优化；
- Matplotlib：图表；
- AKShare、yfinance：可选在线行情源。

### 10.2 离线验证

```powershell
python main.py --use-sample-data --stock-count 60
```

模拟数据使用固定随机种子，适合验证程序是否可以完整运行。

### 10.3 AKShare

```powershell
python main.py `
  --data-source akshare `
  --start-date 20210101 `
  --end-date 20260729 `
  --stock-count 300 `
  --out-of-sample-start 2025-01-01
```

AKShare 路径优先读取 `data/akshare/csi300_prices.csv` 和
`data/akshare/csi300_constituents.csv`，不会重复下载。只有本地缓存不存在时才进入在线下载
回退路径。当前缓存来自 AKShare qfq 日线，包含 300 只当前成分股。

### 10.4 Yahoo Finance

```powershell
python main.py `
  --data-source yfinance `
  --start-date 20180101 `
  --stock-count 50
```

程序会把沪市代码映射为 `.SS`、深市代码映射为 `.SZ`，并依次尝试 CSI 300 ETF 或指数
代码作为基准。

### 10.5 样本内/样本外测试

```powershell
python main.py `
  --use-sample-data `
  --start-date 20180101 `
  --out-of-sample-start 2023-01-01
```

拆分日期只用于报告切分，不会在样本外阶段重新拟合一个固定的样本内模型。动态 ICIR 始终
使用滚动历史数据。

### 10.6 参数敏感性

```powershell
python main.py --use-sample-data --run-sensitivity
```

## 11. 常用命令行参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--start-date` | `20210101` | 回测开始日期，格式 YYYYMMDD |
| `--end-date` | 当前日期 | 回测结束日期 |
| `--stock-count` | `300` | 尝试载入的股票数 |
| `--top-quantile` | `0.20` | 初始高分候选比例 |
| `--scoring-method` | `icir` | `fixed` 或 `icir` |
| `--icir-lookback-months` | `24` | ICIR 滚动窗口 |
| `--icir-min-periods` | `12` | 启用动态权重的最少历史月份 |
| `--max-stock-weight` | `0.10` | 单股权重上限 |
| `--max-industry-deviation` | `0.05` | 行业主动权重上下限 |
| `--max-size-exposure` | `0.15` | Size 因子绝对敞口上限 |
| `--commission-rate` | `0.0003` | 单位换手佣金 |
| `--slippage-rate` | `0.0010` | 单位换手滑点 |
| `--out-of-sample-start` | 空 | 样本外起始日期 |
| `--output-dir` | `outputs` | 报告根目录 |
| `--run-sensitivity` | 关闭 | 是否运行参数敏感性网格 |

完整参数以以下命令为准：

```powershell
python main.py --help
```

## 12. 输入数据

当前真实数据位于：

```text
data/akshare/csi300_prices.csv
data/akshare/csi300_constituents.csv
data/akshare/failed_tickers.csv
```

`csi300_prices.csv` 是长表，包含：

| 字段 | 含义 |
|---|---|
| `date` | 交易日期 |
| `ticker` | 六位股票代码 |
| `open/high/low/close` | AKShare qfq 行情 |
| `volume` | 成交量 |
| `amount` | 成交额 |
| `market_cap` | 缓存市值字段；可用时为价格×股数，否则可能是成交额滚动代理 |

`data/csi300_constituents.csv` 是原项目保留的 50 股静态基本面样例，主要用于旧接口兼容，
不是当前 300 股真实回测的股票池。

如使用自定义静态股票池，最低字段为：

最低字段：

| 字段 | 含义 |
|---|---|
| `code` | 六位股票代码 |
| `name` | 股票名称 |
| `outstanding_share` | 流通股数 |
| `net_asset_per_share` | 每股净资产 |

推荐字段：

| 字段 | 含义 |
|---|---|
| `industry` | 同一层级的行业分类 |
| `roe` | 净资产收益率 |
| `operating_cashflow_to_assets` | 经营现金流/资产 |

数据模块也支持将每股净资产、ROE 和经营现金流/资产传入为带日期索引的 DataFrame。
程序会按日期向前填充后在月末取可见值。生产研究中必须确保索引是数据实际可获得日期或
公告日期，而不是财报期末日期。

## 13. 输出文件

默认写入：

```text
outputs/{data_source}_{scoring_method}/
```

### 图表

| 文件 | 内容 |
|---|---|
| `strategy_performance.png` | 策略/基准净值与策略回撤 |
| `monthly_return_distribution.png` | 月收益分布 |
| `factor_diagnostics.png` | 滚动 IC 与动态因子权重 |
| `portfolio_diagnostics.png` | 行业主动权重、Size 敞口和换手 |

### 因子诊断

| 文件 | 内容 |
|---|---|
| `factor_ic.csv` | 月度因子 Rank IC |
| `factor_ic_summary.csv` | Mean IC、ICIR、t-stat 等汇总 |
| `factor_mimicking_returns.csv` | 因子高减低模拟收益 |
| `factor_quantile_returns.csv` | 因子五分层收益 |
| `factor_weights.csv` | 每次调仓使用的因子权重 |

### 组合与风险

| 文件 | 内容 |
|---|---|
| `portfolio_weights.csv` | 每月完整股票权重 |
| `industry_active_weights.csv` | 相对等权股票池的行业主动权重 |
| `size_exposure.csv` | 组合 Size 因子敞口 |
| `turnover.csv` | 月度单边换手率 |
| `optimizer_status.csv` | 优化成功或回退原因 |
| `monthly_returns.csv` | 毛收益、成本、净收益与基准收益 |
| `data_quality.csv` | 数据来源、覆盖率、有效因子和研究限制 |

### 绩效与稳健性

| 文件 | 内容 |
|---|---|
| `overall_metrics.csv` | 全样本绩效 |
| `yearly_metrics.csv` | 分年度绩效 |
| `sample_metrics.csv` | 样本内/样本外绩效 |
| `cost_sensitivity.csv` | 交易成本敏感性 |
| `parameter_sensitivity.csv` | 可选参数网格结果 |

## 14. 项目结构

```text
factor-model-project/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── csi300_constituents.csv       # 原项目静态样例
│   └── akshare/
│       ├── csi300_prices.csv         # 300 股真实 qfq 日线
│       ├── csi300_constituents.csv   # 当前成分股快照
│       └── failed_tickers.csv
├── outputs/
│   └── akshare_icir/                 # 当前真实数据结果
├── output（原项目）/                  # 原项目历史结果
├── src/
│   └── ff_three_factor/
│       ├── config.py
│       ├── data.py
│       ├── factors.py
│       ├── backtest.py
│       ├── performance.py
│       └── visualization.py
└── tests/
    └── test_pipeline.py
```

`ff_three_factor` 是原项目保留的包名。内部实现已经升级为自适应多因子框架，为避免破坏
原有导入路径，没有强制重命名目录。

## 15. 测试

```powershell
python -m unittest discover -s tests -v
```

当前自动化测试覆盖：

- 行业和市值中性化后的残差性质；
- 因子标准化均值与标准差；
- 权重和为 1；
- 单股权重上限；
- 行业主动权重上下限；
- Size 因子敞口上限。

## 16. 结果解读建议

不要只观察累计收益。建议依次检查：

1. `factor_ic_summary.csv` 中 IC 方向是否符合预期、t-stat 是否稳定；
2. 五分层收益是否近似单调，而非只依赖少数极端月份；
3. 动态因子权重是否频繁翻转；
4. `optimizer_status.csv` 是否出现 fallback；
5. 行业和 Size 敞口是否触及上限；
6. 换手率和成本后收益是否仍有经济意义；
7. 样本外表现是否明显弱于样本内；
8. 不同成本和参数下结论是否保持一致。

高 Sharpe 但 IC 不稳定、换手极高或行业偏离持续触顶的策略，通常缺乏可靠的经济解释。

## 17. 已知限制与后续方向

当前框架已经适合展示研究流程，但真实投资研究仍应补充：

- 历史 CSI 300 成分股，避免使用当前成分股产生幸存者偏差；
- 退市股票和暂停上市股票；
- 按公告日对齐的 point-in-time 财务数据；
- 申万或中信同一层级的历史行业分类；
- 指数真实成分权重，而非等权股票池行业基准；
- ST、停牌、涨跌停和上市时间过滤；
- 分红、送转和复权口径校验；
- 印花税、冲击成本、成交量参与率和容量约束；
- 风险模型中的行业因子、风格因子和特异风险；
- 更严格的 walk-forward 训练、验证和测试窗口；
- 与等权、指数增强、单因子等基准策略的显著性比较。

当前 AKShare 行情是真实 qfq 数据，但股票池是 2026-07-29 的当前成分股静态快照；将其
回溯到 2021 年会产生幸存者偏差。缓存不含历史行业、ROE、现金流和公告日期，因此当前
结果只有 Size、Momentum、Low Volatility 三个有效因子，且不具备有效行业中性。提交
研究报告时必须明确披露这一点，不能把“真实价格”写成“完整 point-in-time 真实回测”。

## 18. 常见问题

### 下载股票数量不足

在线接口可能因网络、限流或字段变化返回空数据。先运行：

```powershell
python main.py --use-sample-data
```

确认本地环境和策略流程正常，再排查数据源。

### 优化器出现 fallback

查看 `optimizer_status.csv`。常见原因是单股上限过低、行业偏离过严、股票数量太少或
Size 敞口限制与候选池冲突。可以增加股票池、放宽约束，或检查缺失行业。

### Quality 权重存在但信号为零

本地基本面缺少 `roe` 和 `operating_cashflow_to_assets` 时，Quality 会从可用因子集合中
停用并重新归一化其余权重。要研究该因子，需要补充按公告日期对齐的字段。

### 中文显示乱码

README 和源码使用 UTF-8。PowerShell 读取文件时可显式指定：

```powershell
Get-Content -Encoding utf8 README.md
```
