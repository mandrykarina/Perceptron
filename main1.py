import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=200):
        self.weights = np.random.rand(input_size + 1) - 0.5  # +1 для bias
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, z):
        return 1 if z >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Добавляем x0 = 1 (bias)
        z = np.dot(self.weights, x)
        return self.activation(z)

        def train(self, X, d):
        self.errors = []  # Список для хранения ошибок по эпохам
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                x = X[i]
                y = self.predict(x)
                error = d[i] - y
                self.weights += self.lr * error * np.insert(x, 0, 1)
                total_error += abs(error)
            self.errors.append(total_error)
            if epoch % 20 == 0:
                print(f"Эпоха {epoch}, Ошибка: {total_error}")


# Генерация синтетического датасета (300x10)
X, d = make_classification(
    n_samples=300,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    random_state=42
)
d = np.where(d == 0, 0, 1)  # Преобразуем метки в 0 и 1

# Нормализация
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Сохраняем датасет в CSV (опционально)
df = pd.DataFrame(np.column_stack((X, d)),
                 columns=[f'x{i+1}' for i in range(10)] + ['d'])
df.to_csv('large_dataset.csv', index=False)

# Создаём и обучаем перцептрон
perceptron = Perceptron(input_size=10, learning_rate=0.01, epochs=200)
perceptron.train(X, d)

# Выводим первые 20 примеров для проверки
print("\nПервые 20 примеров:")
print("Ожидаемое | Предсказание")
print("-----------------------")
for i in range(20):
    print(f"{d[i]}        | {perceptron.predict(X[i])}")

# Отчёт по точности
correct = sum(1 for i in range(len(X)) if perceptron.predict(X[i]) == d[i])
print(f"\nТочность: {correct / len(X) * 100:.2f}%")
