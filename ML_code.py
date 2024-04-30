import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

chess = pd.read_csv("games.csv")
chess.columns

chess = chess.drop(labels="id", axis=1)
chess = chess.drop(labels="rated", axis=1)
chess = chess.drop(labels="created_at", axis=1)
chess = chess.drop(labels="last_move_at", axis=1)
chess = chess.drop(labels="white_id", axis=1)
chess = chess.drop(labels="black_id", axis=1)
chess = chess.drop(labels="moves", axis=1)

chess.info()

new = chess.increment_code.str.split("+", n=5, expand=True)  # разбиваем increment_code
chess["time"] = new[0].astype(int)  # первую часть добавляем в основное время
chess["additional"] = new[1].astype(int)  # вторую в добавочное
chess = chess.drop(labels="increment_code", axis=1)  # избавляемся от переменной
chess.victory_status.unique()  # создадим иной датафрейм для изменения данных
df = chess.victory_status.map(
    {"outoftime": 0, "resign": 1, "mate": 2, "draw": 3}
)  # добавим туда кодировки
chess["victory_status"] = pd.concat([df], axis=1)  # добавим в основной датафрейм
le = LabelEncoder()  # from sklearn.preprocessing import LabelEncoder
chess["opening_name"] = le.fit_transform(
    chess["opening_name"]
)  # трансформирует переменную
chess["opening_eco"] = le.fit_transform(chess["opening_eco"])
chess["winner"].unique()
df = chess["winner"].map({"white": 0, "black": 1, "draw": 2})
chess["winner"] = pd.concat([df], axis=1)

X = chess.drop(labels="winner", axis=1)  # запишем все кроме переменные кроме таргета
y = chess["winner"]  # наш таргет
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)  # разделим выборку

ratings = chess["white_rating"].tolist()  # создаем первый лист с рейтингом за белых
b_ratings = chess["black_rating"].tolist()  # создаем второй с рейтингом за черных
ratings.extend(b_ratings)  # объединяем

mean_rating = round(np.mean(ratings), 2)
max_rating = max(ratings)
min_rating = min(ratings)
std_rating = round(np.std(ratings), 2)
print("Средний рейтинг :", mean_rating)
print("Максимальный рейтинг :", max_rating)
print("Минимальный рейтинг :", min_rating)
print("Std Rating :", std_rating)

plt.hist(ratings, histtype="bar", rwidth=1, color="black")
plt.title("Распределение рейтинга игроков")
plt.xlabel("Рейтинг игрока")
plt.ylabel("Количество игроков")
plt.show()

size = (21, 6)
fig, ax = plt.subplots(1, 2, figsize=size)

# Распределение рейтинга белых
sns.distplot(chess["white_rating"], ax=ax[0])
# Распределение рейтинга черных
sns.distplot(chess["black_rating"], ax=ax[1])

ax_attr = ax[0].set(title="Распределение рейтинга белых")
ax_attr = ax[1].set(title="Распределение рейтинга черных")

# average rating of white
print("Средний рейтинг белых", chess["white_rating"].mean())

# average rating of black
print("Средний рейтинг черных", chess["black_rating"].mean())

ax = sns.distplot(chess["black_rating"], color="r")
ax1 = sns.distplot(chess["white_rating"], color="b")
ax_attr = ax.set(title="Сравнение рейтингов")

fig, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x=chess["time"], y=chess["turns"], palette="deep")
plt.show()

model = RandomForestClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(len(y_train))
print(len(y_test))
baseline_train = np.ones(14040)
baseline_test = np.ones(6018)

print("F1_score train: ", f1_score(y_train, baseline_train, average="micro"))
print("F1_score test: ", f1_score(y_test, baseline_test, average="micro"))

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predict = knn.predict(X_test)
print(knn.score(X_test, y_test))

knn = KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
