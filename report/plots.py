import os, joblib, pandas as pd, numpy as np, yaml, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CFG = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))
DATA_PATH  = os.getenv("DATA_PATH","data/train.csv")
MODEL_PATH = CFG.get("model_path","model/model.pkl")
TARGET_COL = CFG.get("target_col","SalePrice")

df = pd.read_csv(DATA_PATH)
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])
pipe = joblib.load(MODEL_PATH)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
pred = pipe.predict(X_te)

mae = mean_absolute_error(y_te, pred)
rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
r2 = r2_score(y_te, pred)

os.makedirs("report", exist_ok=True)

plt.figure()
plt.scatter(y_te, pred, s=8, alpha=0.6)
plt.xlabel("y_true"); plt.ylabel("y_pred"); plt.title("True vs Predicted")
plt.savefig("report/y_true_vs_pred.png", dpi=150); plt.close()

plt.figure()
res = y_te - pred
plt.hist(res, bins=50)
plt.xlabel("residual"); plt.ylabel("count"); plt.title("Residuals histogram")
plt.savefig("report/residuals_hist.png", dpi=150); plt.close()

with open("report/metrics.md","w",encoding="utf-8") as f:
    f.write("# Отчёт по качеству\n\n")
    f.write("## Итоги\n")
    f.write(f"- MAE:  {mae:.2f}\n")
    f.write(f"- RMSE: {rmse:.2f}\n")
    f.write(f"- R:   {r2:.3f}\n\n")
    f.write("## Графики\n")
    f.write("- y_true_vs_pred.png  истинные vs предсказанные\n")
    f.write("- residuals_hist.png  гистограмма остатков\n")

print("Готово: графики и report/metrics.md")