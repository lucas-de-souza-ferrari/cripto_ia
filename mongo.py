from pymongo import MongoClient
import time

mongo_client = MongoClient("localhost", 27017)
db = mongo_client.ia

if __name__ == "__main__":
    from binance.spot import Spot
    from pymongo.errors import BulkWriteError

    binance_client = Spot()
    exchange_info = binance_client.exchange_info()

    symbols = [s for s in exchange_info["symbols"] if s["status"] == "TRADING"]
    usdt_pairs = [s["symbol"] for s in symbols if s["symbol"].endswith("USDT")]

    intervals = ["1h", "4h", "1d", "3d", "1w"]
    max_candles = 5000
    limit = 1000

    for symbol in usdt_pairs:
        db.symbols.update_one(
            {"_id": symbol},
            {
                "$set": {
                    "baseAsset": next(s["baseAsset"] for s in symbols if s["symbol"] == symbol),
                    "quoteAsset": next(s["quoteAsset"] for s in symbols if s["symbol"] == symbol),
                    "timeframes": intervals,
                }
            },
            upsert=True,
        )

        for interval in intervals:
            try:
                collection_name = f"candles_{interval}"
                collection = db[collection_name]
                collection.create_index([("symbol", 1), ("open_time", 1)], unique=True)

                last = collection.find_one({"symbol": symbol}, sort=[("open_time", -1)])
                start_time = last["open_time"] + 1 if last else None
                end_time = int(time.time() * 1000)

                collected = 0
                new_candles = []

                while collected < max_candles:
                    candles = binance_client.klines(
                        symbol=symbol, interval=interval, startTime=start_time, endTime=end_time, limit=limit
                    )

                    if not candles:
                        break

                    for candle in candles:
                        new_candles.append(
                            {
                                "symbol": symbol,
                                "open_time": candle[0],
                                "open": candle[1],
                                "high": candle[2],
                                "low": candle[3],
                                "close": candle[4],
                                "volume": candle[5],
                                "close_time": candle[6],
                                "quote_asset_volume": candle[7],
                                "trades": candle[8],
                                "taker_buy_base_volume": candle[9],
                                "taker_buy_quote_volume": candle[10],
                            }
                        )

                    collected += len(candles)
                    print(f"🔄 {symbol} [{interval}] - Total coletado: {collected}")
                    start_time = candles[-1][0] + 1
                    time.sleep(0.4)

                if new_candles:
                    try:
                        collection.insert_many(new_candles, ordered=False)
                        print(f"✅ Inseridos {len(new_candles)} candles em {collection_name} para {symbol}")
                    except BulkWriteError:
                        print(f"⚠ Alguns candles já existiam para {symbol} [{interval}]")
                else:
                    print(f"ℹ Nenhum novo candle para {symbol} [{interval}]")

            except Exception as e:
                print(f"❌ Erro ao processar {symbol} [{interval}]: {e}")
