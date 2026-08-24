# ML-Factor-Synthesis 项目规格说明书

## 1. 项目定位

项目名称：ML-Factor-Synthesis

项目目标：在沪深300成分股上，使用机器学习方法对多个传统Alpha因子进行非线性合成，构建一个可用于选股的“ML合成因子”，并与传统等权因子、ICIR加权因子进行实盘风格回测与绩效对比。

该项目适合用于量化求职简历，重点展示以下能力：
- 量化因子研究与特征工程
- 时间序列严谨建模与前向验证
- XGBoost回归建模与模型解释
- 多空因子回测与交易成本建模
- Python工程化项目组织与结果产出

---

## 2. 项目背景与研究问题

传统多因子选股策略常见问题包括：
- 线性加权假设过于简单，难以捕捉因子之间的非线性互补关系
- 因子之间存在相关性和噪声，简单组合容易放大过拟合风险
- 传统因子合成难以同时兼顾预测能力与稳健性

本项目拟回答以下研究问题：
1. 是否存在一个基于机器学习的合成因子，能够显著提升未来收益预测能力？
2. 与传统等权/ICIR加权因子相比，ML合成因子是否在多空收益、风险控制、换手率与稳定性方面更具优势？
3. 通过模型解释性分析（如SHAP），能否发现哪些Alpha因子在不同市场阶段更重要？

---

## 3. 数据范围与数据源

### 3.1 目标资产
- 沪深300成分股
- 数据区间：2018-01-01 至 2026-07-31

### 3.2 数据类型
- 日线行情数据：Open / High / Low / Close / Volume
- 股票池基础信息：行业分类、总市值、流通市值
- 可能补充数据：换手率、财务指标（可选，非核心版本可不包含）

### 3.3 数据源策略
优先顺序：
1. akshare
2. tushare
3. yfinance（备用，若前两者不可用时使用，优先用于本地测试或替代数据源）

### 3.4 数据缓存策略
- 本地缓存机制：使用 parquet / csv 存储中间结果
- 若网络不可用，优先读取本地缓存，确保项目可在离线条件下完成演示

---

## 4. 因子设计与特征工程

### 4.1 因子池
从 WorldQuant Alpha101 中选取 15–20 个逻辑差异较大的因子，建议覆盖以下类别：
- 动量类：ROC、MOM、RSI、TSF
- 反转类：反转因子、短期/长期收益差
- 波动率类：Volatility、Beta、ATR
- 成交量类：Volume Ratio、Turnover
- 流动性类：Amihud Illiquidity、Dollar Volume
- 价格结构类：Price Oscillator、High-Low Spread

### 4.2 因子预处理
对每个因子执行以下步骤：
1. 去极值：MAD 法或 3Sigma 法
2. 标准化：Z-score
3. 行业/市值中性化（可选实验）
   - 作为对比实验，分别构建：
     - 原始因子版本
     - 行业中性版本
     - 市值中性版本
     - 行业+市值双中性版本

### 4.3 特征构造
每月末样本构造规则：
- 使用过去 60 个交易日的因子信息作为特征
- 标签为未来 20 个交易日的累计收益率
- 特征矩阵维度：样本数 × 因子数 × 时间窗口

### 4.4 数据样本构造逻辑
- 以月末为切点，构造样本
- 训练样本中不使用未来数据
- 所有特征和标签均基于时间对齐后的前向数据

---

## 5. 标签定义与样本切分

### 5.1 目标变量
- 预测目标：未来 20 个交易日累计收益率
- 公式：
  $$R_{t+20} = \frac{P_{t+20}}{P_t} - 1$$

### 5.2 时间切分
严格按照时间顺序进行切分，避免任何未来函数与随机打乱：
- 训练集：2018–2022
- 验证集：2023
- 测试集：2024–2026

### 5.3 额外增强策略
建议加入以下稳健性方法：
- Walk-forward validation：每年或每两年滚动重训，观察模型稳定性
- Bootstrap：对回测结果进行多次重采样，估计收益与IR的置信区间

---

## 6. 模型设计

### 6.1 模型选择
- 使用 XGBoost 回归模型：XGBRegressor
- 目标函数：reg:squarederror
- 评估指标：RMSE、MAE

### 6.2 训练策略
- 使用验证集进行早停
- Early stopping：patience = 50
- 训练过程中记录验证集 RMSE 并保存最佳模型

### 6.3 超参数搜索
建议采用网格搜索或随机搜索，重点调参：
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- min_child_weight

### 6.4 模型解释
- 输出特征重要性：feature_importances_
- 保存为 CSV
- 可选新增：SHAP summary plot，增强解释性

---

## 7. 合成因子生成与回测

### 7.1 合成因子构造
- 使用训练好的 XGBoost 模型，对测试集每月末样本生成预测值
- 将预测值作为“ML合成因子”

### 7.2 选股逻辑
- 月度调仓
- 做多前 10% 股票
- 做空后 10% 股票
- 多空组合收益定义为：
  $$R_{long-short} = R_{long} - R_{short}$$

### 7.3 对比基准
- 等权因子：所有因子等权平均后做多空
- ICIR加权因子：基于历史IC和IR进行加权组合
- 可选加入：单因子基准，观察合成效应

### 7.4 交易成本与约束
- 手续费：万分之三
- 滑点：千分之一
- 换手率约束：单次调仓不超过 30%
- 若换手率超过阈值，进行裁剪或延迟换仓

---

## 8. 绩效评估指标

### 8.1 关键指标
- 年化收益率
- 夏普比率
- 最大回撤
- 信息比率（IR）
- 多空收益年化
- 换手率
- 胜率 / 平均收益

### 8.2 因子有效性分析
- 月度IC序列
- ICIR
- 因子分层收益曲线
- 相关性分析与稳定性分析

---

## 9. 输出物

### 9.1 文件产出
- factor_importance.csv
- ml_synthesis_performance.png
- ic_analysis.png
- shap_summary.png（可选）
- 回测结果汇总表（CSV / Excel）

### 9.2 终端输出
- 关键绩效指标摘要
- 模型训练日志
- 回测日志
- 结果对比表

---

## 10. 代码组织建议

建议采用模块化结构：

```text
ML-Factor-Synthesis/
├── config.py
├── data/
│   ├── __init__.py
│   ├── downloader.py
│   ├── cache.py
│   └── preprocess.py
├── factors/
│   ├── __init__.py
│   ├── alpha_factors.py
│   ├── neutralize.py
│   └── feature_pipeline.py
├── model/
│   ├── __init__.py
│   ├── train.py
│   └── explain.py
├── backtest/
│   ├── __init__.py
│   ├── portfolio.py
│   ├── metrics.py
│   └── constraints.py
├── notebooks/
├── outputs/
├── main.py
└── README.md
```

---

## 11. 技术栈

- Python 3.9+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- shap（可选）
- joblib / pickle
- pyarrow / parquet（可选）

---

## 12. 版本规划

### Phase 1：基础可运行版本
- 数据下载与缓存
- 15–20个因子计算
- XGBoost训练与预测
- 简化多空回测
- 输出基础图表与指标

### Phase 2：工程化增强
- 模块化代码重构
- 配置文件统一管理
- 日志系统与异常处理
- 可重复运行脚本

### Phase 3：研究增强
- Walk-forward validation
- Bootstrap 置信区间
- 行业/市值中性化对比实验
- SHAP解释分析

---

## 13. 简历亮点表述建议

适合在简历中突出以下关键词：
- Alpha factor synthesis
- Machine learning for factor investing
- XGBoost regression
- Time-series cross-validation
- Walk-forward validation
- Portfolio backtesting with transaction costs
- Feature importance and SHAP interpretation
- Quantitative research pipeline

---

## 14. 交付标准

项目完成后应满足以下要求：
1. 代码可本地运行，且无依赖付费数据源
2. 可输出至少四类结果文件：因子重要性、绩效曲线、IC分析、SHAP图（可选）
3. 回测逻辑包含手续费、滑点、换手率约束
4. 结果可用于简历展示与面试讲解
5. 具备严谨的时间序列切分与避免未来数据泄漏的设计

---

## 15. 建议补充内容

为了让项目更像“研究型量化工程项目”，建议在实现中额外加入以下内容：
- Walk-forward training：每年滚动重训，观察稳定性
- Bootstrap confidence interval：评估收益与IR的波动范围
- 因子分层分析：将样本按预测得分分层，观察收益差异
- 交易频率控制：通过换手率阈值增强可执行性
- 多实验对比：原始因子 vs 中性化因子 vs ML合成因子

这会显著提升项目在简历和面试中的“研究深度”和“工程完整性”。
