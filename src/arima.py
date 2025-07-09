import pandas as pd
import json
import os
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def run_arima_per_store(store_id=1, forecast_steps=3, train_path='datasets/Arima/train.csv', output_path='outputs/predictions/arima_output.json'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(train_path, parse_dates=['Date'], low_memory=False)

    df = df[(df['Store'] == store_id) & (df['Open'] == 1)].sort_values('Date')

    if len(df) < 10:
        raise ValueError(f"Data untuk Store {store_id} terlalu sedikit untuk ARIMA.")

    df.set_index('Date', inplace=True)

    sales_series = df['Sales'].dropna()

    model = ARIMA(sales_series, order=(1, 1, 1))
    fitted = model.fit()

    forecast = fitted.forecast(steps=forecast_steps)
    forecast_values = forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast)

    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_steps)]

    last_real = sales_series.iloc[-1]
    trend = "naik" if forecast_values[-1] > last_real else "turun"

    output = {
        "model": "arima",
        "target": "sales",
        "store": store_id,
        "forecast": forecast_values,
        "forecast_dates": [d.strftime('%Y-%m-%d') for d in future_dates],
        "trend": trend
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Output ARIMA Store {store_id} disimpan di {output_path}")
    return output

