import json
import os
from datetime import datetime

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def format_currency(value):
    try:
        return f"Rp {int(value):,}".replace(",", ".")
    except:
        return str(value)

def format_date(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%d %B %Y")
    except:
        return date_str

def normalize_outputs(arima_path, clf_path, output_path='outputs/final/normalized_output.json'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    arima = load_json(arima_path)
    clf = load_json(clf_path)

    forecast_text = []
    for date, val in zip(arima['forecast_dates'], arima['forecast']):
        forecast_text.append(f"{format_date(date)}: {format_currency(val)}")

    forecast_paragraph = (
        f"Toko {arima.get('store', 'N/A')} diprediksi akan mengalami tren penjualan *{arima.get('trend', 'tidak diketahui')}* selama "
        f"{len(arima['forecast'])} hari ke depan, dengan estimasi:\n"
        + "\n".join(f"- {t}" for t in forecast_text)
    )

    if 'prediction' in clf:
        predicted_label = clf['prediction']
    elif 'prediksi' in clf:
        predicted_label = clf['prediksi']
    elif 'label_map' in clf and 'predicted_class' in clf:
        label_map = clf['label_map']
        predicted_label = label_map.get(str(clf['predicted_class']), f"Label-{clf['predicted_class']}")
    else:
        predicted_label = "Tidak diketahui"

    clf_paragraph = (
        f"Model klasifikasi memprediksi bahwa pemilik bisnis termasuk dalam kategori *{predicted_label}*, "
        f"berdasarkan data fitur yang telah dianalisis."
    )

    final_text = f"{forecast_paragraph}\n\n{clf_paragraph}"

    output = {
        "store": arima.get("store", None),
        "trend": arima.get("trend", None),
        "forecast": arima.get("forecast", []),
        "forecast_dates": arima.get("forecast_dates", []),
        "prediction": predicted_label,
        "text": final_text
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Normalisasi output berhasil → {output_path}")
    print("\n Teks hasil akhir:\n")
    print(final_text)
