import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#y = n * x + sin(x/n) + шум
n = 21
points_count = 100

# Генеруємо X (наприклад, від 0 до 10)
X = np.linspace(0, 10, points_count).reshape(-1, 1)
# Генеруємо шум (випадкові значення)
noise = np.random.normal(0, 0.5, points_count).reshape(-1, 1)
# Розраховуємо Y за формулою
y = n * X + np.sin(X / n) + noise

#Розділення даних на навчальну (70%) і тестову (30%) вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Побудова лінійної регресійної моделі та пошук коефіцієнтів a і b 
model = LinearRegression()
model.fit(X_train, y_train)

a = model.coef_[0][0] # Коефіцієнт нахилу
b = model.intercept_[0] # Точка перетину

print(f"Рівняння регресії: y = {a:.4f} * x + {b:.4f}")
print(f"Коефіцієнт a: {a:.4f}")
print(f"Коефіцієнт b: {b:.4f}")
print("-" * 30)

#Оцінка якості моделі 
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)     
mae = mean_absolute_error(y_test, y_pred)    
r2 = r2_score(y_test, y_pred)                

print(f"Середньоквадратична похибка (MSE): {mse:.4f}")
print(f"Середня абсолютна помилка (MAE): {mae:.4f}")
print(f"Коефіцієнт детермінації (R^2): {r2:.4f}")
print("-" * 30)

#Побудова графіка точок даних та лінії регресії 
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Експериментальні дані', alpha=0.6)
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Лінія регресії')
plt.title(f'Лінійна регресія (Варіант {n})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

#Висновок про точність та застосовність
print("Висновок:")
if r2 > 0.9:
    print(f"Модель має високу точність (R^2 = {r2:.2f}). Лінійна регресія добре описує дані.")
elif r2 > 0.7:
    print(f"Модель має прийнятну точність (R^2 = {r2:.2f}).")
else:
    print(f"Модель має низьку точність (R^2 = {r2:.2f}). Можливо, залежність не є лінійною.")