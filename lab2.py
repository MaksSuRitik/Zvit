import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

#  Завантаження даних
file_name = 'Lab2_data_1.csv'

try:
    # Зчитуємо варіанти даних
    df = pd.read_csv(file_name, header=None, sep=',') 
    
    
    df = df.replace(',', '.', regex=True).astype(float)

    X = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values               
    print(f"Дані успішно завантажено з файлу {file_name}.")
    
except FileNotFoundError:
    print(f"Помилка: Файл '{file_name}' не знайдено.")
    exit()
except Exception as e:
    print(f"Помилка при зчитуванні: {e}")
    exit()

# Розділення даних ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Сортуємо для красивого графіка
sort_idx = X_test[:, 0].argsort()
X_test_sorted = X_test[sort_idx]

# Побудова моделей 
degrees = [1, 2, 3, 6, 12] 

plt.figure(figsize=(12, 8))
plt.scatter(X, y, color='gray', s=10, alpha=0.3, label='Експериментальні дані')

results = []

print(f"{'Ступінь':<10} | {'MSE Train':<15} | {'MSE Test':<15} | {'R2 Test':<10}")
print("-" * 60)

for degree in degrees:
    # Модель
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    # Навчання
    model.fit(X_train, y_train)
    
    # Прогноз
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Метрики
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    results.append({'deg': degree, 'r2': r2})
    
    print(f"{degree:<10} | {mse_train:<15.4f} | {mse_test:<15.4f} | {r2:<10.4f}")
    
    # Графік
    y_plot = model.predict(X_test_sorted)
    plt.plot(X_test_sorted, y_plot, linewidth=2, label=f'Ступінь {degree} ($R^2$={r2:.3f})')

plt.title('Поліноміальна регресія (Лаб 2)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

#Висновок
best_model = max(results, key=lambda x: x['r2'])
print("\n" + "="*30)
print(f"Рекомендований ступінь полінома: {best_model['deg']}")