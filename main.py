import argparse
from src.m1_data_processing import run_m1
from src.m2_visualization import run_m2
from src.m3_modeling import run_m3
from src.m4_qa_system import TaxiQA, launch_gradio

def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["all","qa","web"],default="all"); a=p.parse_args()
    df,_=run_m1()
    if a.mode=="all": run_m2(df); run_m3(df); print("全流程完成，结果位于 outputs/")
    elif a.mode=="web": launch_gradio(df)
    else:
        qa=TaxiQA(df); print("输入 exit 退出")
        while True:
            q=input("问题> ");
            if q.lower() in {"exit","quit","退出"}: break
            print(qa.answer(q))
if __name__=="__main__": main()
