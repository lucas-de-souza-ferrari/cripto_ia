import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mongo import db
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import AverageTrueRange
from xgboost import XGBClassifier

from binance.spot import Spot
from pymongo.errors import BulkWriteError

binance_client = Spot()
exchange_info = binance_client.exchange_info()

symbols = [s for s in exchange_info["symbols"] if s["status"] == "TRADING"]
usdt_pairs = [s["symbol"] for s in symbols if s["symbol"].endswith("USDT")]

for symbol in usdt_pairs:
    try:
        print(f"\n🔍 Processando símbolo: {symbol}")
        # === Parâmetros do modelo ===
        window_1h = 20
        window_4h = 20
        future_window = 3
        take_profit_pct = 0.015
        stop_loss_pct = 0.015

        # === Função para aplicar indicadores técnicos ===
        def add_indicators(df):
            period = 20
            std_dev = 2

            df["bb_middle"] = df["close"].rolling(window=period).mean()
            df["bb_std"] = df["close"].rolling(window=period).std()
            df["bb_upper"] = df["bb_middle"] + std_dev * df["bb_std"]
            df["bb_lower"] = df["bb_middle"] - std_dev * df["bb_std"]
            range_bb = df["bb_upper"] - df["bb_lower"]
            df["position_close_vs_bb"] = ((df["close"] - df["bb_middle"]) / range_bb.replace(0, np.nan)).fillna(0.0)
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

        # === Carrega e prepara os dados ===
        df_1h = pd.DataFrame(list(db.candles_1h.find({"symbol": symbol}).sort("open_time", 1)))
        df_4h = pd.DataFrame(list(db.candles_4h.find({"symbol": symbol}).sort("open_time", 1)))

        for col in ["open", "high", "low", "close", "volume"]:
            df_1h[col] = pd.to_numeric(df_1h[col], errors="coerce")
            df_4h[col] = pd.to_numeric(df_4h[col], errors="coerce")

        df_1h = add_indicators(df_1h)

        # === Geração de amostras ===
        samples = []

        for i in range(window_1h, len(df_1h) - future_window):
            base_time = df_1h.iloc[i - 1]["open_time"]
            prev_1h = df_1h.iloc[i - window_1h : i]
            prev_4h = df_4h[df_4h["open_time"] < base_time].tail(window_4h)

            if len(prev_4h) < window_4h:
                continue

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
                features.extend(
                    [float(row.open), float(row.high), float(row.low), float(row.close), float(row.volume)]
                )

            current_price = float(df_1h.iloc[i - 1]["close"])
            target_price = current_price * (1 + take_profit_pct)
            stop_price = current_price * (1 - stop_loss_pct)

            fut = df_1h.iloc[i : i + future_window]
            first_hit = None

            for row in fut.itertuples():
                if float(row.low) <= stop_price:
                    first_hit = "stop"
                    break
                if float(row.high) >= target_price:
                    first_hit = "take"
                    break

            if first_hit == "take":
                label = 1
            elif first_hit == "stop":
                label = 0
            else:
                continue

            samples.append((features, label))

        # === Treinamento do Modelo ===
        X = np.array([x for x, y in samples], dtype="float32")
        y = np.array([y for x, y in samples])

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

        xgb = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )

        xgb.fit(X_train, y_train)

        # === Salvamento ===
        joblib.dump(scaler, f"scaler/scaler_xgb_{symbol}.pkl")
        joblib.dump(xgb, f"model/modelo_xgb_{symbol}.pkl")

        print(f"✅ Modelo salvo com sucesso para {symbol}.")

    except Exception as e:
        print(f"❌ Erro ao processar {symbol}: {e}")
        continue
