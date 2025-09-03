import numpy as np
import os, sys, argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train.csv")
    parser.add_argument("--target", default="SalePrice")
    parser.add_argument("--output", default="model/model.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Колонка {args.target} не найдена в {args.input}")

    y = df[args.target]
    X = df.drop(columns=[args.target])

    # делим признаки
    num_cols = X.select_dtypes(include=["number","float","int","bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # иммутация + кодирование + масштабирование
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])

    pipe = Pipeline([
        ("prep", pre),
        ("model", Ridge(alpha=1.0))
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2 = r2_score(y_te, pred)
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.3f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipe, args.output)
    print("Saved:", args.output)

if __name__ == "__main__":
    main()