import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

file_name = 'Lab2_data_1.csv'

try:
    df = pd.read_csv(file_name, header=None, sep=',')
    df = df.replace(',', '.', regex=True).astype(float)
    X = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values
    print(f"Дані успішно завантажено з {file_name}")
except FileNotFoundError:
    print(f"Помилка: Файл {file_name} не знайдено.")
    exit()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Сортування для графіків
sort_idx = X_test[:, 0].argsort()
X_test_sorted = X_test[sort_idx]

max_degree = 15
n = len(y_test) # Розмір тестової вибірки

results = []
print(f"\n{'Ступінь (k)':<12} | {'MSE Тест (sigma^2)':<20} | {'Інформ. Критерій (I)':<22} | {'R2 Оцінка':<10}")
print("-" * 75)

best_ic = float('inf')
best_degree = 0
best_model = None


lab2_best_degree = 6
lab2_metrics = {}

for degree in range(1, max_degree + 1):
    k = degree + 1  # кількість коефіцієнтів 
    
    # Якщо k наближається до n, критерій ламається (ділення на 0), пропускаємо
    if k >= n:
        break

    # Побудова моделі
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    
    # Оцінка на тестовій вибірці
    y_pred = model.predict(X_test)
    sigma2 = mean_squared_error(y_test, y_pred) 
    
    # Розрахунок Інформаційного Критерію 
    ic = sigma2 / (1 - k / n)
    r2 = r2_score(y_test, y_pred)
    
    # Зберігаємо результати
    results.append({
        'degree': degree,
        'mse': sigma2,
        'ic': ic,
        'r2': r2,
        'model': model
    })
    
    print(f"{degree:<12} | {sigma2:<20.4f} | {ic:<22.4f} | {r2:<10.4f}")

    # Пошук мінімуму критерію
    if ic < best_ic:
        best_ic = ic
        best_degree = degree
        best_model = model

  
    if degree == lab2_best_degree:
        lab2_metrics = {'mse': sigma2, 'r2': r2}

print("-" * 75)
print(f"\nЗА МЕТОДОМ МГУА (мінімум IC):")
print(f"Найкращий ступінь: {best_degree}")
print(f"Значення критерію: {best_ic:.4f}")

# Побудова графіків 
plt.figure(figsize=(12, 7))

# Точки даних
plt.scatter(X, y, color='lightgray', s=10, label='Експериментальні дані')


y_plot_gmdh = best_model.predict(X_test_sorted)
plt.plot(X_test_sorted, y_plot_gmdh, color='red', linewidth=2.5, 
         label=f'МГУА Оптимум (ступінь={best_degree}, IC={best_ic:.2f})')

if best_degree != lab2_best_degree:
    model_lab2 = make_pipeline(PolynomialFeatures(lab2_best_degree), LinearRegression())
    model_lab2.fit(X_train, y_train)
    y_plot_lab2 = model_lab2.predict(X_test_sorted)
    plt.plot(X_test_sorted, y_plot_lab2, color='blue', linestyle='--', 
             label=f'Лаб 2 вибір (ступінь={lab2_best_degree})')

plt.title(f'МГУА: Вибір оптимальної складності моделі (Варіант 21)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# Графік залежності критерію від ступеня
degrees_list = [res['degree'] for res in results]
ic_list = [res['ic'] for res in results]

plt.figure(figsize=(10, 5))
plt.plot(degrees_list, ic_list, marker='o', color='green')
plt.axvline(x=best_degree, color='red', linestyle='--', label=f'Мінімум IC (ступінь={best_degree})')
plt.title('Зміна Інформаційного Критерію залежно від ступеня')
plt.xlabel('Ступінь полінома')
plt.ylabel('Значення IC')
plt.legend()
plt.grid(True)
plt.show()