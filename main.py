from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import math
import joblib
import random
import re
import time
import google.generativeai as genai
import whisper  
import shutil   
import os      

app = FastAPI(title="ИИ-Сервер Безопасности")


GEMINI_API_KEY = ""

try:
    genai.configure(api_key=GEMINI_API_KEY)
    chat_model = None
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            chat_model = genai.GenerativeModel(m.name)
            print(f"✅ Генеративный ИИ подключен! Модель: {m.name}")
            break
            
    if chat_model is None:
        print("⚠️ Ошибка: Для этого ключа нет доступных моделей.")
except Exception as e:
    print(f"⚠️ Ошибка подключения Gemini: {e}")
    chat_model = None


try:
    try:
        phishing_model = joblib.load('phishing_model.pkl')
        print("✅ Локальная модель фишинга загружена!")
    except:
        print("⚠️ Модель не найдена. Пожалуйста, запустите train_model.py или power_up_ai.py")
        phishing_model = None
except Exception as e:
    print(f"⚠️ Критическая ошибка загрузки модели: {e}")
    phishing_model = None


print("⏳ Загрузка ИИ-модели Whisper для голоса (подождите пару секунд)...")
try:
    audio_model = whisper.load_model("base")
    print("✅ Модель Whisper успешно загружена!")
except Exception as e:
    print(f"⚠️ Ошибка загрузки Whisper: {e}")
    audio_model = None


class LinkRequest(BaseModel):
    url: str

def extract_url_from_text(text: str) -> str | None:
    url_pattern = re.compile(r'(https?://\S+|www\.\S+)')
    match = url_pattern.search(text)
    if match:
        return match.group(0)
    return None

def handle_chat_with_llm(text: str) -> str:
    if chat_model is None:
        return "Извините, модуль разговорного ИИ сейчас недоступен. Но я всё еще могу проверять ссылки, если вы отправите их отдельно."
    try:
        prompt = f"Тебя зовут Qanysh (Каныш). Ты умный и вежливый ИИ-ассистент по кибербезопасности. Твоя задача — защищать пользователей от фишинга иногда шутить. Отвечай от имени Qanysh , дружелюбно  на языке пользователя. Пользователь пишет тебе: {text}"
        response = chat_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Произошла ошибка при обращении к нейросети Google: {e}"

def extract_features(url_str: str) -> dict:
    url_lower = str(url_str).lower()
    if not url_lower.startswith(('http://', 'https://')):
         url_lower = 'http://' + url_lower

    features = {}
    features['length'] = len(url_lower)
    features['num_dots'] = url_lower.count('.')
    features['num_hyphens'] = url_lower.count('-')
    features['num_slashes'] = url_lower.count('/')
    features['has_login'] = 1 if 'login' in url_lower else 0
    features['has_free'] = 1 if 'free' in url_lower else 0
    features['has_admin'] = 1 if 'admin' in url_lower else 0
    try:
        prob = [float(url_lower.count(c)) / len(url_lower) for c in dict.fromkeys(list(url_lower))]
        features['entropy'] = round(- sum([p * math.log(p) / math.log(2.0) for p in prob]), 3)
    except:
        features['entropy'] = 0
    return features




@app.post("/analyze")
def analyze_input(request: LinkRequest):
    user_input = request.url.strip()
    time.sleep(0.5) 

    extracted_url = extract_url_from_text(user_input)

   
    if extracted_url is None:
        chat_reply = handle_chat_with_llm(user_input)
        return {
            "is_chat": True,
            "message": chat_reply
        }
        
  
    if phishing_model is None:
        return {"is_chat": True, "message": "Ошибка: Локальная модель анализа ссылок недоступна. Обучите её."}
        
    features_dict = extract_features(extracted_url) 
    feature_list = [
        features_dict['length'], features_dict['num_dots'], features_dict['num_hyphens'],
        features_dict['num_slashes'], features_dict['has_login'], features_dict['has_free'],
        features_dict['has_admin'], features_dict['entropy']
    ]
    
    probabilities = phishing_model.predict_proba([feature_list])[0]
    threat_prob = round(probabilities[1] * 100, 2)
    
    if threat_prob < 20:
        status = "Безопасно"
        advices = ["Отличная ссылка, можно смело переходить!", "Угроз не обнаружено. Qanysh одобряет."]
    elif threat_prob < 50:
        status = "Подозрительно"
        advices = ["Риск минимален, но будьте осторожны.", "Сайт выглядит нормально, но сохраняйте бдительность."]
    elif threat_prob < 80:
        status = "Опасно"
        advices = ["Осторожно! Я заметил подозрительные признаки.", "Лучше не вводить там свои данные. Qanysh предупреждает!"]
    else:
        status = "КРИТИЧЕСКАЯ УГРОЗА"
        advices = ["Это фишинг! Ни в коем случае не переходите!", "Крайне высокая вероятность кражи данных. Блокируйте отправителя!"]
        
    return {
        "is_chat": False,
        "status": status,
        "threat_probability": threat_prob,
        "advice": random.choice(advices)
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if audio_model is None:
        return {"error": "Модель распознавания голоса не загружена."}

    temp_file_path = f"temp_{file.filename}"
    try:
        
        with open(temp_file_path, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"🎙️ Анализирую аудио: {temp_file_path}...")
        
        
        result = audio_model.transcribe(temp_file_path, fp16=False)
        recognized_text = result["text"].strip()

        print(f"📝 Распознанный текст: {recognized_text}")
        
        
        return {"text": recognized_text}

    except Exception as e:
        print(f"❌ Ошибка распознавания: {e}")
        return {"error": str(e)}
        
    finally:
     
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)