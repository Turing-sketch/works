"""M2 四项分析及地图加分项。"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from .config import OUTPUT_DIR, ZONE_SHP

def setup():
    font_path=OUTPUT_DIR.parent/"assets"/"fonts"/"NotoSansSC-Regular.ttf"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family=font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family="DejaVu Sans"
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.family":family,"font.sans-serif":[family,"DejaVu Sans"],
                         "axes.unicode_minus":False,"figure.dpi":130})

def demand_pattern(df):
    g=df.groupby(["is_weekend","hour"]).size().unstack(0,fill_value=0)
    fig,ax=plt.subplots(figsize=(10,5)); ax.plot(g.index,g.get(0),marker="o",label="工作日")
    ax.plot(g.index,g.get(1),marker="o",label="周末")
    ax.set(title="工作日与周末分小时出行需求",xlabel="上车小时",ylabel="订单量"); ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"m2_1_demand.png"); plt.close(fig)

def zone_heat(df):
    top=df.PULocationID.value_counts().head(10)
    hot=df[df.PULocationID.isin(top.index)].groupby(["PULocationID","hour"]).size().unstack(fill_value=0)
    fig,axs=plt.subplots(1,2,figsize=(14,5),gridspec_kw={"width_ratios":[1,1.5]})
    sns.barplot(x=top.values,y=top.index.astype(str),orient="h",ax=axs[0],color="#2E86AB")
    axs[0].set(title="上车量最高的 TOP 10 区域",xlabel="订单量",ylabel="区域ID")
    sns.heatmap(hot,cmap="YlOrRd",ax=axs[1]); axs[1].set(title="热门区域分小时订单热力图",xlabel="小时",ylabel="区域ID")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/"m2_2_zones.png"); plt.close(fig)
    top.rename_axis("PULocationID").rename("trips").to_csv(OUTPUT_DIR/"m2_top_zones.csv",encoding="utf-8-sig")

def fare_factors(df):
    sample=df.sample(min(30000,len(df)),random_state=42)
    hourly=df.groupby("hour").fare_amount.median()
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    axs[0].scatter(sample.trip_distance,sample.fare_amount,s=5,alpha=.12,color="#4C78A8")
    axs[0].set(xlim=(0,40),ylim=(0,150),title="行程距离与基础车费",xlabel="距离（英里）",ylabel="车费（美元）")
    axs[1].bar(hourly.index,hourly.values,color="#F58518"); axs[1].set(title="各小时车费中位数",xlabel="上车小时",ylabel="车费中位数（美元）")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/"m2_3_fare.png"); plt.close(fig)

def insight(df):
    # 自选洞察：速度与小费率的联合变化可同时反映拥堵和服务消费行为。
    x=df.assign(tip_rate=(df.tip_amount/df.total_amount.replace(0,np.nan)*100).clip(0,50))
    g=x.groupby("hour").agg(speed=("speed_mph","median"),tip_rate=("tip_rate","median"))
    fig,ax=plt.subplots(figsize=(10,5)); ax.plot(g.index,g.speed,marker="o",color="#2E86AB",label="中位速度")
    ax2=ax.twinx(); ax2.plot(g.index,g.tip_rate,marker="s",color="#E45756",label="中位小费率")
    ax.set(title="全天交通效率与消费行为：速度—小费率",xlabel="上车小时",ylabel="速度（mph）")
    ax2.set_ylabel("小费率（%）"); ax.legend(loc="upper left"); ax2.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"m2_4_insight.png"); plt.close(fig)

def zone_map(df):
    """用轻量 pyshp 绘制分级设色图，避免强制依赖 GeoPandas。"""
    import shapefile
    sf=shapefile.Reader(str(ZONE_SHP)); fields=[f[0] for f in sf.fields[1:]]
    loc_i=fields.index("LocationID"); counts=df.PULocationID.value_counts(); vals=[]; patches=[]
    for sr in sf.iterShapeRecords():
        v=float(counts.get(int(sr.record[loc_i]),0)); shape=sr.shape
        for a,b in zip(list(shape.parts),list(shape.parts[1:])+[len(shape.points)]):
            patches.append(Polygon(shape.points[a:b],closed=True)); vals.append(np.log1p(v))
    fig,ax=plt.subplots(figsize=(9,9)); pc=PatchCollection(patches,cmap="YlOrRd",edgecolor="white",linewidth=.15)
    pc.set_array(np.array(vals)); ax.add_collection(pc); ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("纽约出租车上车需求空间分布（颜色为 log(订单量+1)）")
    fig.colorbar(pc,ax=ax,shrink=.65,label="对数订单量"); fig.tight_layout(); fig.savefig(OUTPUT_DIR/"m2_bonus_zone_map.png"); plt.close(fig)

def run_m2(df):
    setup(); demand_pattern(df); zone_heat(df); fare_factors(df); insight(df); zone_map(df)
