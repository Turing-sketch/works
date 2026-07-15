"""M3 PyTorch 神经网络与随机森林需求预测。"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import random, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .config import OUTPUT_DIR, SEED

def make_dataset(df):
    # 样本粒度为“区域-日期-小时”；区域刻画空间差异，小时/星期/周末/高峰刻画周期规律。
    # 选取订单量最高的80个区域，剔除极稀疏区域，降低零膨胀并提升模型稳定性。
    top=df.PULocationID.value_counts().head(80).index
    x=(df[df.PULocationID.isin(top)].groupby(["PULocationID","date","hour","weekday","is_weekend","is_peak"]).size()
         .rename("demand").reset_index())
    x["date_ordinal"]=pd.to_datetime(x.date).map(pd.Timestamp.toordinal)
    return x

def chronological_split(x):
    # 严格按时间前80%训练、后20%测试，避免随机拆分造成未来信息泄漏。
    dates=sorted(x.date.unique()); cut=dates[max(1,int(len(dates)*.8))-1]
    return x[x.date<=cut].copy(),x[x.date>cut].copy()

def run_m3(df, epochs=60):
    random.seed(SEED); np.random.seed(SEED)
    font_path=OUTPUT_DIR.parent/"assets"/"fonts"/"NotoSansSC-Regular.ttf"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path)); plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font_path)).get_name()
    data=make_dataset(df); train,test=chronological_split(data)
    feats=["PULocationID","hour","weekday","is_weekend","is_peak","date_ordinal"]
    pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),["PULocationID"]),
                           ("num",StandardScaler(),["hour","weekday","is_weekend","is_peak","date_ordinal"])])
    Xtr=pre.fit_transform(train[feats]); Xte=pre.transform(test[feats]); ytr=train.demand.to_numpy(np.float32); yte=test.demand.to_numpy(np.float32)
    rf=RandomForestRegressor(n_estimators=180,min_samples_leaf=2,n_jobs=-1,random_state=SEED)
    rf.fit(train[feats],ytr); rf_pred=rf.predict(test[feats])
    try:
        import torch
        torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)
        model=torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1],64),torch.nn.ReLU(),torch.nn.Dropout(.1),
                                  torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
        opt=torch.optim.Adam(model.parameters(),lr=.003); loss_fn=torch.nn.MSELoss()
        xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr[:,None]); losses=[]
        for _ in range(epochs):
            model.train(); opt.zero_grad(); loss=loss_fn(model(xt),yt); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad(): nn_pred=model(torch.tensor(Xte,dtype=torch.float32)).numpy().ravel()
        framework="PyTorch"
    except ImportError:
        # 仅供受限环境复现实验；正式 requirements 安装 PyTorch 后自动走上方分支。
        from sklearn.neural_network import MLPRegressor
        m=MLPRegressor((64,32),random_state=SEED,max_iter=epochs,learning_rate_init=.003,early_stopping=False)
        m.fit(Xtr,ytr); nn_pred=m.predict(Xte); losses=m.loss_curve_; framework="sklearn fallback"
    plt.figure(figsize=(8,4.5)); plt.plot(losses,color="#2E86AB"); plt.title(f"神经网络训练 Loss 曲线（{framework}）")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"m3_neural_network_loss.png"); plt.close()
    metrics=[]
    for name,pred in [("Neural Network",nn_pred),("Random Forest",rf_pred)]:
        metrics.append({"model":name,"MAE":mean_absolute_error(yte,pred),"RMSE":mean_squared_error(yte,pred)**.5,"test_samples":len(yte)})
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR/"m3_model_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame({"actual":yte,"nn_pred":nn_pred,"rf_pred":rf_pred}).head(1000).to_csv(OUTPUT_DIR/"m3_predictions_sample.csv",index=False)
    # RF擅长非线性且训练稳定；神经网络可扩展到更大数据和复杂嵌入，但需调参且可解释性较弱。
    return pre, rf, data, metrics
