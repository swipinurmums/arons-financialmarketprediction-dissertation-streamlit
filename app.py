import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
import numpy as np
#arons amazing code :)

st.set_page_config(
    page_title="Financial Market Prediction Dashboard",
    layout="wide"
)
with st.sidebar:
    st.header("About")
    st.write(
        "This is my prototype. It demonstrates a machine learning pipeline for short-term financial market direction prediction."
    )

    st.write(
        "The app uses live Yahoo Finance data, engineered technical and market indicators, and a trained XGBoost model to generate a directional prediction."
    )

    st.warning(
     
        "Definitely do not use for financial advice."
    )

# load saved model files created by the training script

xgb_model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("xgboost_scaler.pkl")
features = joblib.load("xgboost_features.pkl")


# load saved results used for the summary dashboard

model_results = pd.read_csv("final_model_comparison.csv")
feature_importance = pd.read_csv("xgboost_feature_importance.csv")
predictions = pd.read_csv("xgboost_predictions.csv")


def create_features(data):

    # work on a copy so the downloaded data stays unchanged
    data = data.copy()

    # calculate daily returns
    data["Returns"] = data["Close"].pct_change()

    # moving averages used to represent short and medium term trend
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA20"] = data["Close"].rolling(20).mean()

    # rolling volatility based on recent daily returns
    data["Volatility"] = data["Returns"].rolling(5).std()

    # lagged returns give the model recent market movement context
    data["Lag1"] = data["Returns"].shift(1)
    data["Lag2"] = data["Returns"].shift(2)

    # simple momentum over the previous five trading days
    data["Momentum"] = (
        data["Close"] - data["Close"].shift(5)
    )

    # longer volatility window to capture wider market instability
    data["Volatility20"] = (
        data["Returns"].rolling(20).std()
    )

    # calculate rsi from average gains and losses
    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (100 / (1 + rs))

    # calculate macd using short and longer exponential moving averages
    ema12 = data["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = ema12 - ema26

    # remove rows created by rolling calculations that contain missing values
    return data.dropna()


def make_prediction(data):

    # take the latest available row of engineered features
    latest_features = data[features].iloc[-1:]

    # apply the same scaler used during model training
    latest_scaled = scaler.transform(
        latest_features
    )

    # generate class prediction
    prediction = xgb_model.predict(
        latest_scaled
    )[0]

    # probability assigned to upward movement
    probability_up = float(
        xgb_model.predict_proba(
            latest_scaled
        )[0][1]
    )

    return prediction, probability_up, latest_features


def show_confidence_message(probability_up):

    # give a simple interpretation of model confidence
    if probability_up > 0.70:

        st.info("High confidence prediction")

    elif probability_up > 0.55:

        st.info("Moderate confidence prediction")

    else:

        st.warning(
            "Low confidence prediction. "
            "Market direction uncertainty is high."
        )

    # reminder that the model often works close to the decision boundary
    st.caption(
        "Confidence values close to 50% indicate high uncertainty. "
        "This reflects the limited predictive strength observed during model evaluation."
    )


st.title("Financial Market Prediction Prototype")

# keep this visible because model outputs could be mistaken for advice
st.warning(
    "This application is for educational and research purposes only "
    "and should not be interpreted as financial advice."
)


# live ticker input

st.header("Live Stock Ticker Input")

ticker = st.text_input(
    "Enter a stock ticker symbol:",
    value="^GSPC"
)

st.write(f"Selected ticker: {ticker}")


if ticker:

    # download one year of daily market data for the selected ticker
    stock_data = yf.download(
        ticker,
        period="1y",
        interval="1d",
        progress=False
    )

    # handle invalid or unsupported ticker symbols
    if stock_data.empty:

        st.error(
            "No data found for this ticker. "
            "Please check the symbol and try again."
        )

    else:

        # create the same features used during model training
        stock_data = create_features(
            stock_data
        )

        # run the saved xgboost model on the latest feature row
        prediction, probability_up, latest_features = (
            make_prediction(stock_data)
        )

        st.subheader(
            "Predicted Market Direction"
        )

        # show the latest available closing price
        latest_close = (
            stock_data["Close"].iloc[-1].item()
        )

        st.metric(
            "Latest Closing Price",
            f"${latest_close:.2f}"
        )

        # display the predicted direction
        if prediction == 1:

            st.success(
                f"Predicted Upward Movement "
                f"({probability_up * 100:.2f}% confidence)"
            )

        else:

            st.error(
                f"Predicted Downward Movement "
                f"({(1 - probability_up) * 100:.2f}% confidence)"
            )

        show_confidence_message(
            probability_up
        )

        # show recent price movement before the technical details
        st.subheader(
            f"Recent Closing Price for {ticker}"
        )

        st.line_chart(
            stock_data["Close"]
        )

        # show the final model inputs for transparency
        st.subheader(
            "Latest Technical Indicator Values"
        )

        st.dataframe(
            latest_features.T.rename(
                columns={
                    latest_features.index[0]: "Value"
                }
            )
        )


# model performance summary

st.header("Model Performance Summary")

st.dataframe(model_results)

col1, col2 = st.columns(2)

with col1:

    st.subheader("ROC-AUC by Model")

    fig, ax = plt.subplots()

    ax.barh(
        model_results["Model"],
        model_results["ROC-AUC"]
    )

    ax.set_xlabel("ROC-AUC")

    st.pyplot(fig)

with col2:

    st.subheader("Accuracy by Model")

    fig, ax = plt.subplots()

    ax.barh(
        model_results["Model"],
        model_results["Accuracy"]
    )

    ax.set_xlabel("Accuracy")

    st.pyplot(fig)


# feature importance

st.header("XGBoost Feature Importance")

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

ax.set_xlabel("Importance")
ax.set_ylabel("Feature")

st.pyplot(fig)


# prediction output analysis

st.header("Prediction Output Analysis")

# count how often the saved test predictions were upward or downward
up_count = (
    predictions["Predicted"] == 1
).sum()

down_count = (
    predictions["Predicted"] == 0
).sum()

total = len(predictions)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Predictions",
    f"{total:,}"
)

col2.metric(
    "Upward Predictions",
    f"{(up_count / total) * 100:.2f}%"
)

col3.metric(
    "Downward Predictions",
    f"{(down_count / total) * 100:.2f}%"
)

# short summary of saved prediction behaviour
st.write(
    "The prediction outputs show how frequently the model predicted "
    "upward or downward market movement. These results demonstrate how difficult it is to confidently predict markets short term."
    " The model's predictions were often close to the decision boundary, "
    "which is consistent with the weak predictive strength observed during evaluation. "
)