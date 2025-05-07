import pandas as pd
from mongo import db
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
import joblib

symbol = "BTCUSDT"

# === Carrega dados ===
df_1h = pd.DataFrame(list(db.candles_1h.find({"symbol": symbol}).sort("open_time", 1)))
df_4h = pd.DataFrame(list(db.candles_4h.find({"symbol": symbol}).sort("open_time", 1)))

# === Garantir que colunas estão em float ===
for col in ["open", "high", "low", "close", "volume"]:
    df_1h[col] = pd.to_numeric(df_1h[col], errors="coerce")
    df_4h[col] = pd.to_numeric(df_4h[col], errors="coerce")

# === Indicadores Técnicos ===
period = 20
std_dev = 2

df_1h["bb_middle"] = df_1h["close"].rolling(window=period).mean()
df_1h["bb_std"] = df_1h["close"].rolling(window=period).std()
df_1h["bb_upper"] = df_1h["bb_middle"] + std_dev * df_1h["bb_std"]
df_1h["bb_lower"] = df_1h["bb_middle"] - std_dev * df_1h["bb_std"]
range_bb = df_1h["bb_upper"] - df_1h["bb_lower"]
range_bb = range_bb.replace(0, np.nan)
df_1h["position_close_vs_bb"] = ((df_1h["close"] - df_1h["bb_middle"]) / range_bb).fillna(0.0)
df_1h["is_close_above_bb"] = (df_1h["close"] > df_1h["bb_upper"]).astype(int)

df_1h["ema_20"] = EMAIndicator(close=df_1h["close"], window=20).ema_indicator()
df_1h["rsi_14"] = RSIIndicator(close=df_1h["close"], window=14).rsi()
macd = MACD(close=df_1h["close"])
df_1h["macd"] = macd.macd()
df_1h["macd_signal"] = macd.macd_signal()

# === Configurações ===
window_1h = 20
window_4h = 4
future_window = 20
take_profit_pct = 0.015
stop_loss_pct = 0.015

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
                float(row.bb_upper) if not pd.isna(row.bb_upper) else 0.0,
                float(row.bb_middle) if not pd.isna(row.bb_middle) else 0.0,
                float(row.bb_lower) if not pd.isna(row.bb_lower) else 0.0,
                float(row.position_close_vs_bb),
                float(row.is_close_above_bb),
                float(row.ema_20) if not pd.isna(row.ema_20) else 0.0,
                float(row.rsi_14) if not pd.isna(row.rsi_14) else 0.0,
                float(row.macd) if not pd.isna(row.macd) else 0.0,
                float(row.macd_signal) if not pd.isna(row.macd_signal) else 0.0,
            ]
        )

    for row in prev_4h.itertuples():
        features.extend([float(row.open), float(row.high), float(row.low), float(row.close), float(row.volume)])

    current_price = float(df_1h.iloc[i - 1]["close"])
    target_price = current_price * (1 + take_profit_pct)
    stop_price = current_price * (1 - stop_loss_pct)

    fut = df_1h.iloc[i : i + future_window]
    label = 0

    for row in fut.itertuples():
        low = float(row.low)
        high = float(row.high)
        if low <= stop_price:
            label = 0
            break
        if high >= target_price:
            label = 1
            break

    if label in (0, 1):
        samples.append((features, label))

# === Dataset final ===
X = np.array([x for x, y in samples], dtype="float32")
y = np.array([y for x, y in samples])

# === Normalização e reshape para LSTM ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_lstm = X_scaled.reshape((X_scaled.shape[0], 25, 12))

# === Split ===
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, y, test_size=0.2, random_state=42, stratify=y
)
X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

# # === Modelo LSTM leve ===
# model_lstm = keras.models.Sequential(
#     [
#         keras.Input(shape=(X_lstm.shape[1], X_lstm.shape[2])),
#         keras.layers.LSTM(32, return_sequences=True),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dropout(0.1),
#         keras.layers.LSTM(16),
#         keras.layers.Dense(16, activation="relu"),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dropout(0.1),
#         keras.layers.Dense(1, activation="sigmoid"),
#     ]
# )

# model_lstm.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])
# model_lstm.summary()

# loss_lstm, acc_lstm = model_lstm.evaluate(X_test_lstm, y_test_lstm)
# print(f"\n✅ LSTM test accuracy: {acc_lstm:.4f}")
# y_pred_probs_lstm = model_lstm.predict(X_test_lstm).flatten()
# y_pred_lstm = (y_pred_probs_lstm >= 0.5).astype(int)
# print("\n📊 [LSTM] Classification Report:")
# print(classification_report(y_test_lstm, y_pred_lstm))
# cm_lstm = confusion_matrix(y_test_lstm, y_pred_lstm)
# display_lstm = ConfusionMatrixDisplay(confusion_matrix=cm_lstm, display_labels=["caiu", "subiu"])
# display_lstm.plot(cmap=plt.cm.Blues)
# plt.title("LSTM Confusion Matrix")
# plt.show()

# # === MLP (Dense Feedforward Neural Network) ===
# model_mlp = keras.models.Sequential(
#     [
#         keras.Input(shape=(X_scaled.shape[1],)),
#         keras.layers.Dense(128, activation="relu"),
#         keras.layers.Dropout(0.2),
#         keras.layers.Dense(64, activation="relu"),
#         keras.layers.Dropout(0.2),
#         keras.layers.Dense(1, activation="sigmoid"),
#     ]
# )
# model_mlp.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# model_mlp.fit(X_train_flat, y_train_flat, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])
# model_mlp.summary()

# loss_mlp, acc_mlp = model_mlp.evaluate(X_test_flat, y_test_flat)
# print(f"\n✅ MLP test accuracy: {acc_mlp:.4f}")
# y_pred_probs_mlp = model_mlp.predict(X_test_flat).flatten()
# y_pred_mlp = (y_pred_probs_mlp >= 0.5).astype(int)
# print("\n📊 [MLP] Classification Report:")
# print(classification_report(y_test_flat, y_pred_mlp))
# cm_mlp = confusion_matrix(y_test_flat, y_pred_mlp)
# display_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=["caiu", "subiu"])
# display_mlp.plot(cmap=plt.cm.Purples)
# plt.title("MLP Confusion Matrix")
# plt.show()

# === RandomForest ===
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_train_flat, y_train_flat)
joblib.dump(scaler, f"scaler_rf_{symbol}.pkl")
joblib.dump(rf, f"modelo_rf_{symbol}.pkl")
y_pred_rf = rf.predict(X_test_flat)
print("\n🎯 [RandomForest] Classification Report:")
print(classification_report(y_test_flat, y_pred_rf))
cm_rf = confusion_matrix(y_test_flat, y_pred_rf)
display_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["caiu", "subiu"])
display_rf.plot(cmap=plt.cm.Greens)
plt.title("RandomForest Confusion Matrix")
plt.show()
