# Optimized Real-Data Outputs

此目录是优化版真实行情实验，原始合成数据结果仍保留在 `outputs/`。

- 行情：BaoStock 真实后复权 OHLCV；
- 股票池：AkShare 当前沪深300成分；
- 时间：2018–2022训练、2023验证、2024–2026 walk-forward 测试；
- 限制：非点时成分，存在幸存者偏差；无市值/行业中性化。

主结果：XGBoost年化净收益2.58%、Sharpe 0.28、最大回撤-15.26%，Bootstrap年化区间
[-6.84%, 14.29%]。月度平均 IC 为0.0130，预测/目标月度离散度比为4.68%，因此只能解读为
弱正向证据。

主要验收文件：`run_metadata.json`、`walk_forward_report.csv`、`training_factor_icir.csv`、
`prediction_distribution.csv`、`backtest_summary.csv` 和 `yearly_performance.csv`。
