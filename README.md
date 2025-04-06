# 🔍 Employee Churn Prediction

This project aims to analyze and predict **employee churn** based on historical company data. By applying data science techniques, feature engineering, machine learning, and cross-validation, the project seeks to identify patterns that contribute to employee turnover and assist in making strategic retention decisions.

---

## 🧠 Technologies Used

- **Language:** Python 3.11+
- **Data Analysis and Visualization:** Pandas, NumPy, Matplotlib, Plotly, Scipy
- **Preprocessing:** Scikit-learn (ColumnTransformer, StandardScaler, OneHotEncoder)
- **Modeling:** RandomForestClassifier
- **Validation and Optimization:** GridSearchCV, StratifiedKFold, Optuna
- **Explainability:** SHAP
- **Virtual Environment:** Pipenv

---

## 📁 Project Structure

- `datasets/employees_churn_dataset.csv` — Employee dataset
- `main.ipynb` — Main notebook containing the entire analysis, feature engineering, modeling, and evaluation pipeline
- `README.md` — General project description

---

## 📊 Steps Performed

### 1. **Data Exploration and Cleaning**

- Reading the dataset with proper date handling
- Null value and data structure analysis
- Visualizations using Plotly (boxplots, histograms, scatter matrix)

### 2. **Feature Engineering**

- Calculation of company tenure, time since last feedback/raise/promotion
- Removal of irrelevant columns (e.g., ID)

### 3. **Exploratory Data Analysis (EDA)**

- Churn distribution
- Statistical tests (Chi-Square) between categorical variables and churn
- Correlation matrix for numerical variables

### 4. **Modeling Preparation**

- Splitting features (`X`) and target (`y`)
- Dropping datetime and irrelevant columns
- Preprocessing with `StandardScaler` and `OneHotEncoder`
- Splitting data into training and testing sets (50% each)

### 5. **Modeling**

- Training a baseline Random Forest model
- Evaluation using metrics such as AUC, F1-score, precision, and recall
- ROC curve and confusion matrix visualization

### 6. **Cross-Validation + Hyperparameter Tuning**

- Applying `GridSearchCV` with `StratifiedKFold` (5 folds)
- Tuning parameters like `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- Returning the best model, best parameters, and recall metric

---

## 🧪 Results

- **AUC ROC:** ~0.89
- **Optimized Recall:** after cross-validation
- **Feature Importance:** interpreted with SHAP (in progress)

---

## 🛠 How to Run

```bash
# Clone the repository

# Install Pipenv (if not already installed)
pip install pipenv

# Create the virtual environment and install dependencies
pipenv install pandas scipy plotly scikit-learn optuna shap ipykernel ipywidgets nbformat numpy

# Activate the virtual environment
pipenv shell

# Open the notebook in Jupyter
jupyter notebook main.ipynb
```

---

# 🔍 Predicción de Rotación de Empleados

Este proyecto tiene como objetivo analizar y predecir la **rotación de empleados** a partir de datos históricos de la empresa. Aplicando técnicas de ciencia de datos, ingeniería de características, aprendizaje automático y validación cruzada, el proyecto busca identificar patrones que contribuyen a la salida de empleados y ayudar en la toma de decisiones estratégicas de retención.

---

## 🧠 Tecnologías Utilizadas

- **Lenguaje:** Python 3.11+
- **Análisis y Visualización de Datos:** Pandas, NumPy, Matplotlib, Plotly, Scipy
- **Preprocesamiento:** Scikit-learn (ColumnTransformer, StandardScaler, OneHotEncoder)
- **Modelado:** RandomForestClassifier
- **Validación y Optimización:** GridSearchCV, StratifiedKFold, Optuna
- **Interpretabilidad:** SHAP
- **Entorno Virtual:** Pipenv

---

## 📁 Estructura del Proyecto

- `datasets/employees_churn_dataset.csv` — Conjunto de datos de empleados
- `main.ipynb` — Notebook principal con todo el pipeline de análisis, ingeniería de características, modelado y evaluación
- `README.md` — Descripción general del proyecto

---

## 📊 Etapas Realizadas

### 1. **Exploración y Limpieza de Datos**

- Lectura del conjunto de datos con manejo adecuado de fechas
- Análisis de valores nulos y estructura de los datos
- Visualizaciones con Plotly (diagramas de caja, histogramas, matriz de dispersión)

### 2. **Ingeniería de Características**

- Cálculo del tiempo en la empresa, tiempo desde el último feedback/aumento/promoción
- Eliminación de columnas irrelevantes (como el ID)

### 3. **Análisis Exploratorio de Datos (EDA)**

- Distribución de la rotación
- Pruebas estadísticas (Chi-cuadrado) entre variables categóricas y la rotación
- Matriz de correlación para variables numéricas

### 4. **Preparación para el Modelado**

- Separación entre características (`X`) y objetivo (`y`)
- Eliminación de columnas de tipo datetime e irrelevantes
- Preprocesamiento con `StandardScaler` y `OneHotEncoder`
- División de los datos en entrenamiento y prueba (50% cada uno)

### 5. **Modelado**

- Entrenamiento de un modelo base con Random Forest
- Evaluación con métricas como AUC, F1-score, precisión y recall
- Visualización de la curva ROC y matriz de confusión

### 6. **Validación Cruzada + Ajuste de Hiperparámetros**

- Aplicación de `GridSearchCV` con `StratifiedKFold` (5 particiones)
- Optimización de parámetros como `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- Retorno del mejor modelo, mejores parámetros y métrica de recall

---

## 🧪 Resultados

- **AUC ROC:** ~0.89
- **Recall Optimizado:** tras validación cruzada
- **Importancia de Características:** interpretada con SHAP (en desarrollo)

---

## 🛠 Cómo Ejecutar

```bash
# Clonar el repositorio

# Instalar Pipenv (si aún no está instalado)
pip install pipenv

# Crear el entorno virtual e instalar las dependencias
pipenv install pandas scipy plotly scikit-learn optuna shap ipykernel ipywidgets nbformat numpy

# Activar el entorno virtual
pipenv shell

# Abrir el notebook en Jupyter
jupyter notebook main.ipynb
```

---

# 🔍 Employee Churn Prediction

Este projeto tem como objetivo analisar e prever o **Churn (rotatividade)** de funcionários com base em dados históricos da empresa. Utilizando técnicas de ciência de dados, engenharia de features, aprendizado de máquina e validação cruzada, o projeto busca identificar padrões que contribuam para a saída de colaboradores e auxiliar na tomada de decisões estratégicas de retenção.

---

## 🧠 Tecnologias Utilizadas

- **Linguagem:** Python 3.11+
- **Análise e Visualização de Dados:** Pandas, NumPy, Matplotlib, Plotly, Scipy
- **Pré-processamento:** Scikit-learn (ColumnTransformer, StandardScaler, OneHotEncoder)
- **Modelagem:** RandomForestClassifier
- **Validação e Otimização:** GridSearchCV, StratifiedKFold, Optuna
- **Explicabilidade:** SHAP
- **Ambiente Virtual:** Pipenv

---

## 📁 Estrutura do Projeto

- `datasets/employees_churn_dataset.csv` — Base de dados de funcionários
- `main.ipynb` — Código principal com todo o pipeline de análise, engenharia de features, modelagem e avaliação
- `README.md` — Descrição geral do projeto

---

## 📊 Etapas Realizadas

### 1. **Exploração e Limpeza dos Dados**

- Leitura da base com tratamento de datas
- Análise de valores nulos e estrutura dos dados
- Visualizações com Plotly (boxplots, histogramas, matriz de dispersão)

### 2. **Engenharia de Atributos**

- Cálculo de tempo de empresa, tempo desde último feedback/aumento/mudança de cargo
- Exclusão de colunas irrelevantes (como ID)

### 3. **Análise Exploratória (EDA)**

- Distribuição do churn
- Testes estatísticos (Qui-Quadrado) entre variáveis categóricas e o churn
- Matriz de correlação para variáveis numéricas

### 4. **Preparação para Modelagem**

- Separação entre features (`X`) e target (`y`)
- Exclusão de colunas datetime e variáveis irrelevantes
- Pré-processamento com `StandardScaler` e `OneHotEncoder`
- Separação em treino e teste (50% cada)

### 5. **Modelagem**

- Treinamento de um modelo base com Random Forest
- Avaliação com métricas como AUC, F1-score, precision, recall
- Visualização da curva ROC e matriz de confusão

### 6. **Validação Cruzada + Tuning de Hiperparâmetros**

- Aplicação de `GridSearchCV` com `StratifiedKFold` (5 folds)
- Otimização de parâmetros como `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- Retorno do melhor modelo, melhores parâmetros e métrica de recall

---

## 🧪 Resultados

- **AUC ROC:** ~0.89
- **Recall Otimizado:** após validação cruzada
- **Importância de Features:** interpretadas com SHAP (em desenvolvimento)

---

## 🛠 Como Executar

```bash
# Clone o repositório

# Instale o Pipenv (se ainda não tiver)
pip install pipenv

# Crie o ambiente virtual e instale as dependências
pipenv install pandas scipy plotly scikit-learn optuna shap ipykernel ipywidgets nbformat numpy

# Ative o ambiente virtual
pipenv shell

# Abra o notebook no Jupyter
jupyter notebook main.ipynb
```
