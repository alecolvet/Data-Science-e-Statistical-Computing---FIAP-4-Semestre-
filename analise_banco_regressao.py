import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, mean_squared_error, r2_score
)

# ============================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ============================================================
df = pd.read_csv('/content/bank-additional-full.csv', sep=';')
print("Colunas disponíveis:")
print('\n'.join(df.columns))

# Codificação das variáveis categóricas
le = LabelEncoder()
df_encoded = df.copy()
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df[col])

# Variável alvo binária (y: yes=1, no=0)
df_encoded['y_bin'] = (df['y'] == 'yes').astype(int)

# ============================================================
# 1. PROBABILIDADE GERAL DE ACEITAR O DEPÓSITO
# (Regressão Logística — variável binária)
# ============================================================
print("\n--- 1. Probabilidade de aceitar o depósito ---")

features_log = ['age', 'duration', 'campaign', 'pdays', 'previous',
                'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                'euribor3m', 'nr.employed']

X1 = df_encoded[features_log]
y1 = df_encoded['y_bin']

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

scaler1 = StandardScaler()
X1_train_sc = scaler1.fit_transform(X1_train)
X1_test_sc  = scaler1.transform(X1_test)

clf = LogisticRegression(max_iter=500, random_state=42)
clf.fit(X1_train_sc, y1_train)

y1_pred = clf.predict(X1_test_sc)
prob_geral = clf.predict_proba(X1_test_sc)[:, 1].mean()

print(f"Probabilidade média prevista de aceitar: {prob_geral * 100:.2f}%")
print(f"Acurácia do modelo: {accuracy_score(y1_test, y1_pred) * 100:.2f}%")
print("\nRelatório de Classificação:")
print(classification_report(y1_test, y1_pred, target_names=['Não', 'Sim']))

# ============================================================
# 2. PROBABILIDADE DE TER EMPRÉSTIMO PESSOAL ATIVO
# (Regressão Logística — variável binária)
# ============================================================
print("\n--- 2. Probabilidade de ter empréstimo pessoal ativo ---")

df_encoded['loan_bin'] = (df['loan'] == 'yes').astype(int)

features_loan = ['age', 'job', 'marital', 'education', 'housing']
X2 = df_encoded[features_loan]
y2 = df_encoded['loan_bin']

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

scaler2 = StandardScaler()
X2_train_sc = scaler2.fit_transform(X2_train)
X2_test_sc  = scaler2.transform(X2_test)

clf2 = LogisticRegression(max_iter=500, random_state=42)
clf2.fit(X2_train_sc, y2_train)

y2_pred = clf2.predict(X2_test_sc)
prob_loan = clf2.predict_proba(X2_test_sc)[:, 1].mean()

print(f"Probabilidade média prevista de ter empréstimo: {prob_loan * 100:.2f}%")
print(f"Acurácia do modelo: {accuracy_score(y2_test, y2_pred) * 100:.2f}%")
print(classification_report(y2_test, y2_pred, target_names=['Sem Empréstimo', 'Com Empréstimo']))

# ============================================================
# 3. PROBABILIDADE DE ACEITAR O DEPÓSITO DADO ENSINO SUPERIOR
# (Regressão Logística — subconjunto filtrado)
# ============================================================
print("\n--- 3. Aceitação condicionada a ensino superior ---")

df_superior = df_encoded[df['education'] == 'university.degree'].copy()

X3 = df_superior[features_log]
y3 = df_superior['y_bin']

X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y3, test_size=0.2, random_state=42
)

scaler3 = StandardScaler()
X3_train_sc = scaler3.fit_transform(X3_train)
X3_test_sc  = scaler3.transform(X3_test)

clf3 = LogisticRegression(max_iter=500, random_state=42)
clf3.fit(X3_train_sc, y3_train)

y3_pred = clf3.predict(X3_test_sc)
prob_superior = clf3.predict_proba(X3_test_sc)[:, 1].mean()

print(f"Probabilidade média prevista (ensino superior): {prob_superior * 100:.2f}%")
print(f"Acurácia do modelo: {accuracy_score(y3_test, y3_pred) * 100:.2f}%")

# ============================================================
# 4. REGRESSÃO LINEAR CONTÍNUA: DURAÇÃO DA CHAMADA ~ EURIBOR
#    (substitui a questão sem dados disponíveis)
# ============================================================
print("\n--- 4. Regressão Linear: Duração da chamada ~ Euribor3m ---")

X4 = df_encoded[['euribor3m', 'emp.var.rate', 'cons.conf.idx', 'age']].values
y4 = df_encoded['duration'].values

X4_train, X4_test, y4_train, y4_test = train_test_split(
    X4, y4, test_size=0.2, random_state=42
)

reg = LinearRegression()
reg.fit(X4_train, y4_train)

y4_pred = reg.predict(X4_test)
mse  = mean_squared_error(y4_test, y4_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y4_test, y4_pred)

print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.2f} segundos")
print(f"Coeficientes: euribor3m={reg.coef_[0]:.4f}, emp.var.rate={reg.coef_[1]:.4f}, "
      f"cons.conf.idx={reg.coef_[2]:.4f}, age={reg.coef_[3]:.4f}")
print(f"Intercepto   : {reg.intercept_:.4f}")

# ============================================================
# 5. PROBABILIDADE DE ACEITAR O DEPÓSITO — CLIENTES > 40 ANOS
# (Regressão Logística — subconjunto filtrado)
# ============================================================
print("\n--- 5. Aceitação entre clientes com mais de 40 anos ---")

df_40 = df_encoded[df['age'] > 40].copy()

X5 = df_40[features_log]
y5 = df_40['y_bin']

X5_train, X5_test, y5_train, y5_test = train_test_split(
    X5, y5, test_size=0.2, random_state=42
)

scaler5 = StandardScaler()
X5_train_sc = scaler5.fit_transform(X5_train)
X5_test_sc  = scaler5.transform(X5_test)

clf5 = LogisticRegression(max_iter=500, random_state=42)
clf5.fit(X5_train_sc, y5_train)

y5_pred = clf5.predict(X5_test_sc)
prob_40 = clf5.predict_proba(X5_test_sc)[:, 1].mean()

print(f"Probabilidade média prevista (>40 anos): {prob_40 * 100:.2f}%")
print(f"Acurácia do modelo: {accuracy_score(y5_test, y5_pred) * 100:.2f}%")
print(classification_report(y5_test, y5_pred, target_names=['Não', 'Sim']))
