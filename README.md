# 📈 AI Crypto Price Movement Predictor

Este projeto utiliza aprendizado de máquina para prever o próximo movimento (alta ou queda) de uma criptomoeda com base em dados históricos e indicadores técnicos.

## 🔍 Objetivo

Prever se o preço de um ativo (ex: BTC/USDT) irá subir ou cair na próxima janela temporal, com base no histórico de candles 1h e 4h, usando indicadores técnicos e um modelo treinado com o algoritmo **XGBoost**.

## 🧠 Técnicas e Tecnologias

- **Fontes de dados**:
  - API oficial da Binance
  - Banco de dados MongoDB (via MongoDB Atlas)

- **Indicadores técnicos aplicados**:
  - Bollinger Bands (BB)
  - EMA 20
  - RSI 14
  - MACD (linha e sinal)
  - Stochastic RSI (K e D)
  - ADX
  - ATR

- **Machine Learning**:
  - `XGBoost Classifier` (`XGBClassifier`)
  - Normalização com `MinMaxScaler`
  - Avaliação via matriz de confusão e métricas de classificação

## 📁 Estrutura do Projeto

```
├── mongo.py           # Conexão com o MongoDB Atlas e coleta de candles
├── Ai_v3.py           # Processamento de dados e treinamento do modelo
├── test/test_model.py      # Aplicação do modelo para prever o próximo movimento
├── model/             # Modelos treinados (.pkl)
├── scaler/            # Scalers salvos (.pkl)
```

## 🚀 Como Executar

### 1. Treinamento do Modelo

> Requer Python 3.10.11+

Execute o script `Ai_v3.py`:

```bash
python Ai_v3.py
```

O script irá:

- Buscar os candles 1h e 4h do MongoDB
- Aplicar os indicadores técnicos
- Gerar amostras com base em take profit e stop loss
- Treinar o modelo com XGBoost
- Salvar os arquivos:
  - `model/modelo_xgb_{symbol}.pkl`
  - `scaler/scaler_xgb_{symbol}.pkl`

### 2. Testar o Modelo

Execute o script `test_model.py`:

```bash
python test_model.py
```

O script irá:

- Buscar os dados mais recentes da Binance
- Aplicar os mesmos indicadores
- Preparar os dados conforme o treinamento
- Carregar o modelo e o scaler salvos
- Exibir a previsão: 📈 SUBIR ou 📉 CAIR
- Mostrar a **confiança (%)** da previsão

## 💾 Pré-requisitos

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate       # Windows
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Ou, se preferir, instale manualmente:

```bash
pip install pandas numpy matplotlib scikit-learn ta xgboost pymongo joblib python-dotenv
```

## 🔐 Configuração do MongoDB

Crie um arquivo `mongo.py` com o seguinte conteúdo:

```python
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://usuario:senha@seu-cluster.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["ia"]
```

## ✅ Exemplo de Output

```text
🔮 Previsão para o próximo movimento do mercado:
📈 SUBIR
📊 Confiança: 83.45%
```

## 📌 Observações

- Os dados são transformados em amostras supervisionadas com base nas condições de take profit e stop loss.
- O modelo binário prevê: `0 = cair` e `1 = subir`.
- Pode ser adaptado para outros ativos e intervalos.

## 📜 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
