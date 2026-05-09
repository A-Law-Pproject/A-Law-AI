import sys, json, pathlib
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_excel(ROOT / "tests/평가데이터셋/챗봇_평가용_최종자료.xlsx")
df.columns = ["주유형", "세부유형", "질문", "답변"]
df = df.dropna(subset=["질문"])
df["질문"] = df["질문"].astype(str)
df["답변"] = df["답변"].fillna("").astype(str)
df["세부유형"] = df["세부유형"].fillna("-").astype(str)

risk_kw_q = ["독소", "위험", "불법", "무효", "사기", "강제", "불리", "일방", "손해배상", "해지권", "임의"]
risk_kw_a = ["무효", "불법", "위반", "독소", "보호", "위험", "강제"]
risk_sub  = ["임대인 권한", "강제 집행", "사기 예방", "법적 쟁점", "분쟁 예방", "법적 해결", "해지 효력", "특약 사항"]

m = (
    df["질문"].str.contains("|".join(risk_kw_q), na=False) |
    df["세부유형"].str.contains("|".join(risk_sub), na=False) |
    df["답변"].str.contains("|".join(risk_kw_a), na=False)
)
risk_df = df[m].reset_index(drop=True)

print(f"risk 관련: {len(risk_df)}개 / 전체 {len(df)}개")
print("주유형 분포:", risk_df["주유형"].value_counts().to_dict())
print()
for i, r in risk_df.head(10).iterrows():
    print(f"[{r['주유형']}][{r['세부유형']}]")
    print(f"  Q: {r['질문'][:70]}")
    print(f"  A: {r['답변'][:90]}")
    print()

p = ROOT / "results/risk_chatbot_candidates.json"
p.parent.mkdir(exist_ok=True)
records = risk_df[["주유형","세부유형","질문","답변"]].to_dict("records")
p.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"저장: {p}  ({len(records)}개)")
