"""M1 数据处理：质量审计、清洗与特征工程。"""
import numpy as np
import pandas as pd
from .config import PARQUET_PATH, OUTPUT_DIR

KEEP = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
        "trip_distance", "PULocationID", "DOLocationID", "payment_type",
        "fare_amount", "tip_amount", "total_amount"]

def load_raw(path=PARQUET_PATH):
    return pd.read_parquet(path, columns=KEEP)

def make_quality_report(df):
    """同时给出缺失率及业务规则异常数，便于复核而非只做 describe。"""
    rules = {
        "passenger_count": ~df.passenger_count.between(1, 6),
        "trip_distance": ~df.trip_distance.between(0.1, 100),
        "fare_amount": ~df.fare_amount.between(0, 500),
        "total_amount": ~df.total_amount.between(0, 1000),
        "trip_duration": (df.tpep_dropoff_datetime <= df.tpep_pickup_datetime),
        "pickup_time": ~df.tpep_pickup_datetime.between("2026-01-01", "2026-01-31 23:59:59"),
    }
    rows=[]
    for col in df.columns:
        rows.append({"field": col, "dtype": str(df[col].dtype), "rows": len(df),
                     "missing_count": int(df[col].isna().sum()),
                     "missing_rate_pct": round(df[col].isna().mean()*100, 4),
                     "abnormal_count": int(rules.get(col, pd.Series(False,index=df.index)).sum())})
    for name in ("trip_duration", "pickup_time"):
        rows.append({"field": name, "dtype": "business_rule", "rows": len(df),
                     "missing_count": 0, "missing_rate_pct": 0,
                     "abnormal_count": int(rules[name].sum())})
    out=pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR/"data_quality_report.csv", index=False, encoding="utf-8-sig")
    return out

def clean_and_engineer(df):
    x=df.copy()
    # 乘客数缺失用1填补：黄色出租车常为单人出行，且该字段只参与描述性分析。
    x["passenger_count"] = x.passenger_count.fillna(1)
    # 删除时间倒置、零/极端距离和负/极端金额；阈值兼顾数据错误与机场长途订单。
    duration=(x.tpep_dropoff_datetime-x.tpep_pickup_datetime).dt.total_seconds()/60
    mask=(x.tpep_pickup_datetime.between("2026-01-01","2026-01-31 23:59:59") &
          duration.between(1,180) & x.trip_distance.between(0.1,100) &
          x.fare_amount.between(0,500) & x.total_amount.between(0,1000) &
          x.PULocationID.between(1,265) & x.DOLocationID.between(1,265))
    x=x.loc[mask].copy()
    x["trip_duration_min"]=(x.tpep_dropoff_datetime-x.tpep_pickup_datetime).dt.total_seconds()/60
    x["hour"]=x.tpep_pickup_datetime.dt.hour
    x["weekday"]=x.tpep_pickup_datetime.dt.dayofweek
    x["date"]=x.tpep_pickup_datetime.dt.date
    x["is_weekend"]=(x.weekday>=5).astype("int8")
    x["is_peak"]=x.hour.isin([7,8,9,16,17,18,19]).astype("int8")
    # 衍生1：平均速度识别交通状态；衍生2：每英里费用消除里程尺度，便于比较计价效率。
    x["speed_mph"]=(x.trip_distance/(x.trip_duration_min/60)).clip(0,80)
    x["fare_per_mile"]=(x.fare_amount/x.trip_distance).clip(0,100)
    return x

def run_m1():
    raw=load_raw(); report=make_quality_report(raw); clean=clean_and_engineer(raw)
    summary=pd.DataFrame({"stage":["raw","clean"],"rows":[len(raw),len(clean)]})
    summary.to_csv(OUTPUT_DIR/"m1_cleaning_summary.csv",index=False,encoding="utf-8-sig")
    return clean, report

if __name__ == "__main__":
    d,r=run_m1(); print(f"清洗完成：{len(d):,} 条")
