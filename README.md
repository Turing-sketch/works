# 城市出租车出行数据分析与智能问答系统

本项目使用 NYC TLC 2026 年 1 月黄色出租车真实行程数据，完成数据质量审计、清洗与特征工程、四类分析可视化、区域地图、PyTorch 神经网络与随机森林需求预测，以及本地自然语言问答和 Gradio 可视化界面。

## 环境与安装

- Python 3.9+（建议 3.10–3.12）
- 内存建议 8 GB 以上，首次完整运行约需数分钟

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 数据放置

主数据应位于 `data/yellow_tripdata_2026-01.parquet`。地图文件位于 `data/`，包括 `taxi_zones.shp/.shx/.dbf/.prj`。数据来源：NYC Taxi & Limousine Commission Trip Record Data。

## 运行

```bash
# 数据加载→清洗→分析→训练全流程
python main.py

# 命令行问答
python main.py --mode qa

# Gradio 可视化界面
python main.py --mode web
```

界面启动后打开终端显示的本地地址（通常为 `http://127.0.0.1:7860`）。可问：高峰时段、热门区域 TOP5、周末对比、5 英里车费、18 点需求预测、最拥堵时段。

## 设计亮点

1. 质量报告不仅计算缺失率，还用时间、里程、金额和区域业务规则统计异常。
2. 清洗保留机场长途订单，同时排除倒置时间、零距离、负金额与极端值。
3. 增加速度、每英里车费、周末和高峰等可解释特征。
4. 模型按日期前 80%/后 20% 拆分，避免未来信息泄漏；固定随机种子。
5. 问答接口返回数字结论、解释和相关输出路径，无 API Key 也可运行。
6. 附加区域分级设色地图和 Gradio 图形界面。

## 输出文件

- `data_quality_report.csv`：字段缺失率与异常统计
- `m2_1_demand.png` 至 `m2_4_insight.png`：四项分析
- `m2_bonus_zone_map.png`：地图加分项
- `m3_neural_network_loss.png`：训练曲线
- `m3_model_metrics.csv`：MAE、RMSE 对比

## 可复现性与说明

随机种子固定为 42。模型数据采用“区域—日期—小时”聚合粒度，并选取订单量最高的 80 个区域，以减少极稀疏区域造成的零膨胀。随机森林的类别区域 ID 在此被视为分裂变量；神经网络则对区域进行独热编码并标准化数值特征。

仓库 URL：`【请上传 GitHub/Gitee 后填写】`
