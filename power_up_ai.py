import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import math
import requests
import io

print("📥 Загружаю базу данных из репозитория...")
url = "https://raw.githubusercontent.com/lucas-m-lopes/phishing-links-detection/master/dataset.csv"

try:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
except Exception as e:
    print(f"⚠️ Не удалось скачать файл по ссылке. Использую расширенный локальный набор.")
    data = {
        'url': [
            'https://google.com', 'https://yandex.ru', 'https://github.com', 'https://apple.com',
            'http://free-iphone-now.com', 'http://login-verify-account.net', 'http://secure-bank-update.com',
            'https://microsoft.com', 'https://stackoverflow.com', 'http://win-money-fast.biz',
            'http://paypal-security-check.com', 'https://amazon.com', 'http://netflix-free-trial.org'
        ] * 100, 
        'label': [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1] * 100
    }
    df = pd.DataFrame(data)

print(f"✅ Данные готовы! Найдено строк для анализа: {len(df)}")

def extract_features(url_str):
    url_str = str(url_str).lower()
    if not url_str: return [0]*8
    
    length = len(url_str)
    dots = url_str.count('.')
    hyphens = url_str.count('-')
    slashes = url_str.count('/')
    login = 1 if 'login' in url_str else 0
    free = 1 if 'free' in url_str else 0
    admin = 1 if 'admin' in url_str else 0
    
    
    try:
        prob = [float(url_str.count(c)) / len(url_str) for c in dict.fromkeys(list(url_str))]
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    except:
        entropy = 0
        
    return [length, dots, hyphens, slashes, login, free, admin, round(entropy, 3)]


print("🧠 Превращаю ссылки в математические признаки...")
X = []
y = []

for i in range(len(df)):
    row_url = df.iloc[i, 0] 
    label = df.iloc[i, 1]   
    
    X.append(extract_features(row_url))
    y.append(label)


print("🔥 Запускаю обучение Случайного Леса...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


joblib.dump(model, 'phishing_model.pkl')
print("🚀 ГОТОВО! Модель 'phishing_model.pkl' успешно обновлена.")