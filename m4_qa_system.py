"""M4 本地自然语言问答与 Gradio 加分界面。"""
import re, pandas as pd
from .config import OUTPUT_DIR

class TaxiQA:
    def __init__(self,df): self.df=df
    def answer(self,q):
        q=q.strip(); nums=[int(x) for x in re.findall(r"\d+",q)]
        if any(k in q for k in ["最高峰","最忙","高峰时段"]):
            g=self.df.groupby("hour").size(); h=int(g.idxmax()); return f"需求最高的是 {h}:00–{h+1}:00，共 {g.max():,} 单。\n相关图表：outputs/m2_1_demand.png"
        if any(k in q for k in ["热门区域","区域排名","最多的区域","TOP"]):
            n=nums[0] if nums else 5; s=self.df.PULocationID.value_counts().head(min(n,20))
            return "热门上车区域：\n"+"\n".join(f"{i}. 区域 {z}：{v:,} 单" for i,(z,v) in enumerate(s.items(),1))+"\n相关文件：outputs/m2_2_zones.png"
        if any(k in q for k in ["周末","工作日"]):
            s=self.df.groupby("is_weekend").size(); return f"工作日 {s.get(0,0):,} 单，周末 {s.get(1,0):,} 单。\n相关图表：outputs/m2_1_demand.png"
        if any(k in q for k in ["车费","费用估算","多少钱"]):
            miles=nums[0] if nums else 5; sub=self.df[(self.df.trip_distance>=miles*.8)&(self.df.trip_distance<=miles*1.2)]
            med=sub.total_amount.median(); return f"参考相近里程历史订单，{miles} 英里总费用中位数约 ${med:.2f}（非实时报价）。\n相关图表：outputs/m2_3_fare.png"
        if any(k in q for k in ["预测","需求量"]):
            h=next((n for n in nums if 0<=n<=23),18); avg=self.df[self.df.hour==h].groupby("date").size().mean()
            return f"基于历史同小时均值，{h}:00 全市需求约 {avg:,.0f} 单；正式模型指标见 outputs/m3_model_metrics.csv。"
        if any(k in q for k in ["速度","拥堵"]):
            g=self.df.groupby("hour").speed_mph.median(); h=int(g.idxmin()); return f"中位速度最低在 {h}:00，约 {g.min():.1f} mph。\n相关图表：outputs/m2_4_insight.png"
        return "暂未识别该问题。可询问：高峰时段、热门区域TOP5、周末对比、5英里车费、18点需求预测、最拥堵时段。"

def launch_gradio(df):
    import gradio as gr
    qa=TaxiQA(df)
    with gr.Blocks(title="城市出租车智能问答") as demo:
        gr.Markdown("# 🚕 城市出租车出行数据智能问答\n基于 2026 年 1 月纽约黄色出租车真实数据")
        inp=gr.Textbox(label="请输入问题",placeholder="例如：热门区域TOP5？18点需求量如何？")
        out=gr.Textbox(label="分析结论",lines=8)
        gr.Examples(["哪个时段最忙？","热门区域TOP5","周末和工作日订单量对比","5英里大约多少钱？","18点需求预测","哪个时段最拥堵？"],inp)
        inp.submit(qa.answer,inp,out); gr.Button("分析").click(qa.answer,inp,out)
    demo.launch()
