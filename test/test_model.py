import pandas as pd
import numpy as np
import joblib
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from binance.spot import Spot
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import AverageTrueRange


def fetch_binance_candles(symbol: str, interval: str, limit: int = 1000):
    client = Spot()
    raw = client.klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(
        raw,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_indicators(df):
    period = 20
    std_dev = 2
    df["bb_middle"] = df["close"].rolling(window=period).mean()
    df["bb_std"] = df["close"].rolling(window=period).std()
    df["bb_upper"] = df["bb_middle"] + std_dev * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - std_dev * df["bb_std"]
    range_bb = df["bb_upper"] - df["bb_lower"]
    range_bb = range_bb.replace(0, np.nan)
    df["position_close_vs_bb"] = ((df["close"] - df["bb_middle"]) / range_bb).fillna(0.0)
    df["is_close_above_bb"] = (df["close"] > df["bb_upper"]).astype(int)
    df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    stoch_rsi = StochRSIIndicator(close=df["close"], window=14, smooth1=3, smooth2=3)
    df["stochrsi_k"] = stoch_rsi.stochrsi_k()
    df["stochrsi_d"] = stoch_rsi.stochrsi_d()

    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx()

    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr"] = atr.average_true_range()
    return df


def prepare_input(df_1h, df_4h):
    prev_1h = df_1h.tail(20)
    prev_4h = df_4h.tail(20)

    features = []
    for row in prev_1h.itertuples():
        features.extend(
            [
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume),
                float(row.bb_upper or 0),
                float(row.bb_middle or 0),
                float(row.bb_lower or 0),
                float(row.position_close_vs_bb),
                float(row.is_close_above_bb),
                float(row.ema_20 or 0),
                float(row.rsi_14 or 0),
                float(row.macd or 0),
                float(row.macd_signal or 0),
                float(row.stochrsi_k or 0),
                float(row.stochrsi_d or 0),
                float(row.adx or 0),
                float(row.atr or 0),
            ]
        )
    for row in prev_4h.itertuples():
        features.extend([float(row.open), float(row.high), float(row.low), float(row.close), float(row.volume)])

    return np.array([features], dtype="float32")


def predict_direction(symbol="HIFIUSDT"):
    df_1h = fetch_binance_candles(symbol, "1h")
    df_4h = fetch_binance_candles(symbol, "4h")
    df_1h = add_indicators(df_1h)

    features = prepare_input(df_1h, df_4h)
    model = joblib.load(f"model/modelo_xgb_{symbol}.pkl")
    scaler = joblib.load(f"scaler/scaler_xgb_{symbol}.pkl")
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    confidence = proba[prediction] * 100
    print("\nPrevisão para o próximo movimento do mercado:")
    print("SUBIR" if prediction == 1 else "CAIR")
    print(f"Confiança: {confidence:.2f}%")


if __name__ == "__main__":
    predict_direction()
