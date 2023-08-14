import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn import datasets

# Tải mô hình
@st.cache_data
def load_model(model_name):
    with open(model_name, 'rb') as file:
        model = pickle.load(file)
    return model

# Danh sách mô hình
models = {
    "Decision Tree": load_model("Decision_Tree.pkl"),
    "Logistic Regression": load_model("Logistic_Regression.pkl"),
    "Naive Bayes": load_model("Naive_Bayes.pkl"),
    "SVM": load_model("SVM.pkl")
}


st.markdown("""
    <style>
        .reportview-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .main .block-container {
            width: 65%;
        }
        h1 {
            color: #ff6347;
            font-family: 'Courier New', Courier, monospace;
        }
        .prediction {
            padding: 20px;
            border: 2px solid #ff6347;
            border-radius: 5px;
            font-size: 20px;
            text-align: center;
            background-color: #e6e6e6;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("Phân loại hoa Iris")


model_choice = st.sidebar.selectbox("Chọn mô hình:", list(models.keys()))
st.sidebar.subheader("Nhập dữ liệu:")
data_input_method = st.sidebar.radio("Lựa chọn phương thức nhập dữ liệu:", ["Nhập thủ công", "Tải file CSV"])

if data_input_method == "Tải file CSV":
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV:", type=['csv'])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.sidebar.write(data)
        user_data = data.values
    else:
        user_data = None
else:
    use_slider = st.sidebar.checkbox("Sử dụng thanh kéo")

    if use_slider:
        sepal_length = st.sidebar.slider("Chiều dài lá đài (cm)", 4.0, 8.0, 5.5)
        sepal_width = st.sidebar.slider("Chiều rộng lá đài (cm)", 2.0, 4.5, 3.5)
        petal_length = st.sidebar.slider("Chiều dài cánh hoa (cm)", 1.0, 7.0, 4.5)
        petal_width = st.sidebar.slider("Chiều rộng cánh hoa (cm)", 0.1, 2.5, 1.5)
    else:
        sepal_length = st.sidebar.number_input("Chiều dài lá đài (cm)", min_value=4.0, max_value=8.0, value=5.5)
        sepal_width = st.sidebar.number_input("Chiều rộng lá đài (cm)", min_value=2.0, max_value=4.5, value=3.5)
        petal_length = st.sidebar.number_input("Chiều dài cánh hoa (cm)", min_value=1.0, max_value=7.0, value=4.5)
        petal_width = st.sidebar.number_input("Chiều rộng cánh hoa (cm)", min_value=0.1, max_value=2.5, value=1.5)
    
    user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if user_data is not None:
    model = models[model_choice]
    prediction = model.predict(user_data)
    predicted_class = datasets.load_iris().target_names[prediction[0]]
    st.subheader(f"Kết quả dự đoán từ thuật toán {model_choice}:")
    st.markdown(f"<div class='prediction'>{predicted_class}</div>", unsafe_allow_html=True)

    # Hiển thị hình ảnh tương ứng
    image_map = {
        "setosa": "setosa.jpg",
        "versicolor": "versicolor.jpg",
        "virginica": "virginica.jpg"
    }
    if predicted_class in image_map:
        st.image(image_map[predicted_class], caption=f'Hoa Iris {predicted_class}', use_column_width=True)
