import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import math


data = {
    'url': [
        'http://google.com', 
        'https://yandex.ru', 
        'http://free-iphone-login.com/bonus', 
        'http://vk.com', 
        'http://admin-bonus-win.net/login',  
        'https://github.com',
        'http://secure-update-account.com',   
        'http://mail.ru'
    ],
    'label': [0, 0, 1, 0, 1, 0, 1, 0] 
}
df = pd.DataFrame(data)


def extract_features(url: str) -> list:
    url_lower = url.lower()
    
    length = len(url)
    num_dots = url.count('.')
    num_hyphens = url.count('-')
    num_slashes = url.count('/')
    has_login = 1 if 'login' in url_lower else 0
    has_free = 1 if 'free' in url_lower else 0
    has_admin = 1 if 'admin' in url_lower else 0
    
    prob = [float(url_lower.count(c)) / len(url_lower) for c in dict.fromkeys(list(url_lower))]
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    
    
    return [length, num_dots, num_hyphens, num_slashes, has_login, has_free, has_admin, round(entropy, 3)]
print("Извлекаем признаки из ссылок...")

X = df['url'].apply(extract_features).tolist()
y = df['label'].tolist()

print("Обучаем Искусственный Интеллект...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, 'phishing_model.pkl')
print("Готово! Обученная модель сохранена в файл 'phishing_model.pkl'")