from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json

model_name = "t5-small" 
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def load_json(path):

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {path} tidak ditemukan.")
        return None
    except json.JSONDecodeError:
        print(f"Error parsing JSON dari {path}.")
        return None

def build_prompt(data):

    store = data.get("store", "N/A")
    trend = data.get("trend", "tidak diketahui")
    prediksi = data.get("prediction", "tidak diketahui")  
    forecast = data.get("forecast", [])
    forecast_dates = data.get("forecast_dates", [])
    
    prompt = f"Analisis komprehensif untuk toko {store}: "
    
    if trend == "turun":
        prompt += "Toko mengalami penurunan penjualan yang memerlukan perhatian serius dan strategi perbaikan. "
    elif trend == "naik":
        prompt += "Toko menunjukkan pertumbuhan penjualan yang positif dan perlu dipertahankan dengan strategi yang tepat. "
    else:
        prompt += "Toko memiliki tren penjualan yang stabil namun perlu optimalisasi untuk pertumbuhan. "
    
    if forecast and forecast_dates:
        prompt += "Berdasarkan analisis time series, proyeksi penjualan untuk periode mendatang adalah: "
        for i, (date, value) in enumerate(zip(forecast_dates[:3], forecast[:3])):
            prompt += f"tanggal {date} diproyeksikan mencapai Rp {value:,.0f}, "
        prompt = prompt.rstrip(", ") + ". "
    
    if prediksi == "Bersedia":
        prompt += "Profil pemilik toko menunjukkan kesiapan tinggi untuk mengadopsi inovasi dan teknologi baru dalam bisnis. "
    elif prediksi == "Tidak Bersedia":
        prompt += "Profil pemilik toko menunjukkan resistensi terhadap perubahan dan memerlukan pendekatan khusus dalam adopsi teknologi. "
    
    prompt += "Berikan analisis mendalam tentang kondisi bisnis, identifikasi masalah utama, peluang pengembangan, rekomendasi strategis yang actionable, dan langkah-langkah konkret untuk meningkatkan performa dan profitabilitas toko."
    
    return prompt

def generate_narrative(prompt):

    input_text = f"generate comprehensive business analysis and recommendations: {prompt}"
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=300,           
            min_length=120,          
            num_beams=8,             
            early_stopping=False,    
            do_sample=True,           
            temperature=0.8,         
            top_p=0.9,               
            repetition_penalty=1.3, 
            no_repeat_ngram_size=3,  
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    output_text = clean_output(output_text)
    
    return output_text

def clean_output(output_text):

    prefixes_to_remove = [
        "generate comprehensive business analysis and recommendations:",
        "generate narrative:",
        "comprehensive business analysis and recommendations:",
        "business analysis and recommendations:",
        "analysis and recommendations:"
    ]
    
    for prefix in prefixes_to_remove:
        if prefix in output_text.lower():
            import re
            output_text = re.sub(re.escape(prefix), "", output_text, flags=re.IGNORECASE).strip()
    
    if output_text and not output_text[0].isupper():
        output_text = output_text[0].upper() + output_text[1:]
    
    if output_text and not output_text.endswith(('.', '!', '?')):
        output_text += '.'

    output_text = ' '.join(output_text.split())
    
    return output_text

def enhance_narrative_with_structure(base_narrative, data):

    store = data.get("store", "N/A")
    trend = data.get("trend", "tidak diketahui")
    prediksi = data.get("prediction", "tidak diketahui")
    forecast = data.get("forecast", [])
    forecast_dates = data.get("forecast_dates", [])
    
    enhanced_narrative = f"LAPORAN ANALISIS BISNIS - TOKO {store}\n"
    enhanced_narrative += "=" * 50 + "\n\n"
    
    enhanced_narrative += "RINGKASAN EKSEKUTIF:\n"
    enhanced_narrative += base_narrative + "\n\n"
    
    enhanced_narrative += "ANALISIS DETAIL:\n"
    
    enhanced_narrative += f"1. ANALISIS TREN PENJUALAN ({trend.upper()}):\n"
    if trend == "turun":
        enhanced_narrative += "   • Identifikasi penyebab: Analisis faktor internal dan eksternal\n"
        enhanced_narrative += "   • Dampak: Penurunan revenue dan potensi kerugian\n"
        enhanced_narrative += "   • Urgensi: Diperlukan tindakan segera untuk recovery\n\n"
    elif trend == "naik":
        enhanced_narrative += "   • Faktor pendorong: Identifikasi elemen yang berkontribusi positif\n"
        enhanced_narrative += "   • Potensi: Peluang untuk scaling dan ekspansi\n"
        enhanced_narrative += "   • Momentum: Manfaatkan tren positif untuk growth\n\n"
    
    if forecast and forecast_dates:
        enhanced_narrative += "2. PROYEKSI PENJUALAN:\n"
        for i, (date, value) in enumerate(zip(forecast_dates[:3], forecast[:3])):
            enhanced_narrative += f"   • {date}: Rp {value:,.0f}\n"
        enhanced_narrative += "\n"
    
    enhanced_narrative += "3. ASSESSMENT KESIAPAN TEKNOLOGI:\n"
    if prediksi == "Bersedia":
        enhanced_narrative += "   • Status: SIAP untuk adopsi teknologi\n"
        enhanced_narrative += "   • Karakteristik: Terbuka terhadap inovasi\n"
        enhanced_narrative += "   • Peluang: Implementasi solusi digital advanced\n\n"
    elif prediksi == "Tidak Bersedia":
        enhanced_narrative += "   • Status: MEMERLUKAN pendekatan khusus\n"
        enhanced_narrative += "   • Karakteristik: Resistensi terhadap perubahan\n"
        enhanced_narrative += "   • Strategi: Pendekatan bertahap dan edukatif\n\n"
    
    enhanced_narrative += "REKOMENDASI STRATEGIS:\n"
    if prediksi == "Bersedia":
        enhanced_narrative += "• Implementasi sistem POS terintegrasi\n"
        enhanced_narrative += "• Pengembangan e-commerce dan digital marketing\n"
        enhanced_narrative += "• Adopsi analytics untuk business intelligence\n"
        enhanced_narrative += "• Training tim untuk skill digital\n"
    else:
        enhanced_narrative += "• Edukasi bertahap tentang manfaat teknologi\n"
        enhanced_narrative += "• Pilot project dengan solusi sederhana\n"
        enhanced_narrative += "• Pendampingan intensif selama transisi\n"
        enhanced_narrative += "• Demonstrasi ROI yang jelas dan terukur\n"
    
    return enhanced_narrative

def process_comprehensive_narrative(data):

    if not data:
        return "Error: Data tidak dapat dimuat."
    
    prompt = build_prompt(data)
    
    base_narrative = generate_narrative(prompt)

    final_narrative = enhance_narrative_with_structure(base_narrative, data)
    
    return final_narrative