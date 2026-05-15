import yfinance as yf
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
#MAKING THE MODEL TRAINING SCRIPT TO CREATE THE MODEL FILES NEEDED FOR THE APP
# Download S&P 500 data
sp500 = yf.download("^GSPC", start="2000-01-01", end="2026-01-01", progress=False)

# Flatten columns if yfinance returns a MultiIndex
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)

# Create target and features
sp500["Returns"] = sp500["Close"].pct_change()
sp500["Target"] = (sp500["Close"].shift(-1) > sp500["Close"]).astype(int)
sp500 = sp500.iloc[:-1]

sp500["MA5"] = sp500["Close"].rolling(5).mean()
sp500["MA20"] = sp500["Close"].rolling(20).mean()
sp500["Volatility"] = sp500["Returns"].rolling(5).std()

sp500["Lag1"] = sp500["Returns"].shift(1)
sp500["Lag2"] = sp500["Returns"].shift(2)
sp500["Momentum"] = sp500["Close"] - sp500["Close"].shift(5)
sp500["Volatility20"] = sp500["Returns"].rolling(20).std()

delta = sp500["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()

rs = average_gain / average_loss
sp500["RSI"] = 100 - (100 / (1 + rs))

ema12 = sp500["Close"].ewm(span=12, adjust=False).mean()
ema26 = sp500["Close"].ewm(span=26, adjust=False).mean()
sp500["MACD"] = ema12 - ema26

sp500 = sp500.dropna()

features = [
    "Returns",
    "MA5",
    "MA20",
    "Volatility",
    "Lag1",
    "Lag2",
    "Momentum",
    "Volatility20",
    "RSI",
    "MACD"
]

X = sp500[features]
y = sp500["Target"]

# Same chronological split approach as notebook
split = int(len(sp500) * 0.8)

X_train = X[:split]
y_train = y[:split]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train_scaled, y_train)

# Save model, scaler and feature order
joblib.dump(xgb_model, "xgboost_model.pkl")
joblib.dump(scaler, "xgboost_scaler.pkl")
joblib.dump(features, "xgboost_features.pkl")

print("Model, scaler and feature list exported successfully.")