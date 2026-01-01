import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open('models/model_pipeline.pickle', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    return train

artifacts = load_model()
model = artifacts['model']
scaler = artifacts['scaler']
encoder = artifacts['encoder']
numeric_features = artifacts['numeric_features']
categorical_features = artifacts['categorical_features']
medians = artifacts['medians']
metrics = artifacts['metrics']
feature_names = artifacts['feature_names']
coefficients = artifacts['coefficients']

df_train = load_data()

st.title("Предсказание цены автомобиля")
st.markdown("---")

page = st.sidebar.selectbox(
    "Выберите раздел",
    ["Главная", "EDA", "Предсказание", "Анализ модели"]
)

# ГЛАВНАЯ 
if page == "Главная":    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Train R²", f"{metrics['train_r2']:.4f}")
    with col2:
        st.metric("Test R²", f"{metrics['test_r2']:.4f}")
    with col3:
        st.metric("Test RMSE", f"{np.sqrt(metrics['test_mse']):,.0f}")
    with col4:
        st.metric("Business Metric", f"{metrics['business_metric']:.2%}")
    
    st.markdown("---")
    st.markdown("""
    ### О проекте
    
    Это приложение предсказывает цену подержанного автомобиля на основе его характеристик.
    
    **Используемая модель:** Ridge Regression с категориальными признаками
    """)

# EDA
elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Распределения", "Корреляции", "Категории"])
    
    with tab1:
        st.subheader("Распределение цен")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df_train['selling_price'], kde=True, ax=ax)
        ax.set_xlabel('Цена продажи')
        ax.set_ylabel('Частота')
        st.pyplot(fig)
        
        st.subheader("Зависимость цены от года выпуска")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df_train, x='year', y='selling_price', alpha=0.5, ax=ax)
        ax.set_xlabel('Год выпуска')
        ax.set_ylabel('Цена продажи')
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Корреляционная матрица")
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
        corr_matrix = df_train[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
        st.pyplot(fig)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Цена по типу топлива")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df_train, x='fuel', y='selling_price', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Цена по трансмиссии")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df_train, x='transmission', y='selling_price', ax=ax)
            st.pyplot(fig)

# ПРЕДСКАЗАНИЕ
elif page == "Предсказание":
    st.header("Предсказание цены автомобиля")
    
    input_method = st.radio("Способ ввода данных:", ["Ручной ввод", "Загрузка CSV"])
    
    if input_method == "Ручной ввод":
        st.subheader("Введите характеристики автомобиля")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year = st.number_input("Год выпуска")
            km_driven = st.number_input("Пробег")
            mileage = st.number_input("Расход топлива")
        
        with col2:
            engine = st.number_input("Объём двигателя")
            max_power = st.number_input("Мощность")
            seats = st.selectbox("Количество мест", [2, 4, 5, 6, 7, 8, 9, 10], index=2)
        
        with col3:
            fuel = st.selectbox("Тип топлива", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
            seller_type = st.selectbox("Тип продавца", ['Individual', 'Dealer', 'Trustmark Dealer'])
            transmission = st.selectbox("Трансмиссия", ['Manual', 'Automatic'])
            owner = st.selectbox("Количество владельцев", 
                               ['First Owner', 'Second Owner', 'Third Owner', 
                                'Fourth & Above Owner', 'Test Drive Car'])
        
        if st.button("Предсказать цену", type="primary"):
            input_data = pd.DataFrame({
                'year': [year],
                'km_driven': [km_driven],
                'mileage': [mileage],
                'engine': [engine],
                'max_power': [max_power],
                'fuel': [fuel],
                'seller_type': [seller_type],
                'transmission': [transmission],
                'owner': [owner],
                'seats': [seats]
            })
            
            X_numeric = input_data[numeric_features].values
            
            X_categorical = encoder.transform(input_data[categorical_features])
            
            X_full = np.hstack([X_numeric, X_categorical])
            
            X_scaled = scaler.transform(X_full)
            
            prediction = model.predict(X_scaled)[0]
            
            st.success(f"### Предсказанная цена: {prediction:.0f}")
    
    else:  
        st.subheader("Загрузите CSV файл")
        
        uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.write("Загруженные данные:")
            st.dataframe(input_df.head())
            
            if st.button("Предсказать цены", type="primary"):
                try:
                    for col, median_val in medians.items():
                        if col in input_df.columns:
                            input_df[col].fillna(median_val, inplace=True)
                    
                    X_numeric = input_df[numeric_features].values
                    X_categorical = encoder.transform(input_df[categorical_features])
                    X_full = np.hstack([X_numeric, X_categorical])
                    X_scaled = scaler.transform(X_full)
                    
                    predictions = model.predict(X_scaled)
                    
                    result_df = input_df.copy()
                    result_df['predicted_price'] = predictions
                    
                    st.write("Результаты предсказаний:")
                    st.dataframe(result_df)
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label=" Скачать результаты",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Ошибка при обработке файла: {e}")

# АНАЛИЗ
elif page == "Анализ модели":
    st.header("Анализ модели")
    
    tab1, tab2 = st.tabs(["Веса признаков", "Метрики"])
    
    with tab1:
        st.subheader("Коэффициенты модели (важность признаков)")
        
        coef_df = pd.DataFrame({
            'Признак': feature_names,
            'Коэффициент': coefficients,
            'Абс. значение коэф.': np.abs(coefficients)
        }).sort_values('Абс. значение коэф.', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['green' if c > 0 else 'red' for c in coef_df['Коэффициент']]
        ax.barh(coef_df['Признак'], coef_df['Коэффициент'], color=colors)
        ax.set_xlabel('Коэффициент')
        ax.set_title('Веса признаков в модели Ridge')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Интерпретация:**
        - 🟢 **увеличивают** предсказанную цену
        - 🔴 **уменьшают** предсказанную цену
        - Чем длиннее столбик, тем сильнее влияние признака
        """)
        
        st.subheader("Таблица коэффициентов")
        st.dataframe(coef_df.sort_values('Абс. значение коэф.', ascending=False))
    
    with tab2:
        st.subheader("Метрики качества модели")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Регрессионные метрики")
            metrics_df = pd.DataFrame({
                'Метрика': ['R² (Train)', 'R² (Test)', 'RMSE (Test)'],
                'Значение': [
                    f"{metrics['train_r2']:.4f}",
                    f"{metrics['test_r2']:.4f}",
                    f"{np.sqrt(metrics['test_mse']):,.0f}"
                ]
            })
            st.table(metrics_df)
        
        with col2:
            st.markdown("### Бизнес-метрика")
            st.metric(
                "Доля точных предсказаний (±10%)",
                f"{metrics['business_metric']:.2%}"
            )
            st.progress(metrics['business_metric'])
