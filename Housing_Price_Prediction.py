import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide")
st.title('Prediksi Harga Rumah 🏠')

# ——————————————————————————————————————————————
# 1) Load artifacts
@st.cache_resource
def load_artifacts():
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path='enhanced_housing_price_model.tflite')
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load scaler & feature selector
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_selector.pkl', 'rb') as f:
        selector = pickle.load(f)

    return interpreter, input_details, output_details, scaler, selector

interp, in_det, out_det, scaler, selector = load_artifacts()
feature_names = list(scaler.feature_names_in_)

# ——————————————————————————————————————————————
# 2) Sidebar: input mentah
st.sidebar.header('Masukkan Fitur Dasar')

area      = st.sidebar.number_input('Luas Area (sq ft)', 1000.0, 20000.0, 5000.0)
bedrooms  = st.sidebar.slider('Jumlah Kamar Tidur',       1,      6,      3)
bathrooms = st.sidebar.slider('Jumlah Kamar Mandi',       1,      4,      2)
stories   = st.sidebar.slider('Jumlah Lantai (stories)',  1,      4,      2)
mainroad  = st.sidebar.selectbox('Dekat Jalan Utama',     ['Ya','Tidak'])
guestroom = st.sidebar.selectbox('Memiliki Kamar Tamu',   ['Ya','Tidak'])
basement  = st.sidebar.selectbox('Memiliki Basement',     ['Ya','Tidak'])
hotwater  = st.sidebar.selectbox('Pemanas Air',           ['Ya','Tidak'])
aircond   = st.sidebar.selectbox('AC',                    ['Ya','Tidak'])
parking   = st.sidebar.slider('Tempat Parkir',            0,      5,      1)
prefarea  = st.sidebar.selectbox('Area Preferensi',       ['Ya','Tidak'])
furn      = st.sidebar.selectbox('Status Perabotan',      ['Furnished','Semi-Furnished','Unfurnished'])

raw = pd.DataFrame([{
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'mainroad': mainroad,
    'guestroom': guestroom,
    'basement': basement,
    'hotwaterheating': hotwater,
    'airconditioning': aircond,
    'parking': parking,
    'prefarea': prefarea,
    'furnishingstatus': furn
}])

# Debug: tampilkan feature_names yang diharapkan scaler
with st.sidebar.expander("🔍 Debug: Fitur yang dibutuhkan scaler"):
    st.write(feature_names)

# ——————————————————————————————————————————————
# 3) Pipeline feature engineering
def make_features(df):
    df = df.copy()
    # Rasio fitur
    df['area_per_bedroom']       = df['area']    / df['bedrooms']
    df['bathroom_bedroom_ratio'] = df['bathrooms']/ df['bedrooms']
    df['parking_per_area']       = df['parking'] / df['area']

    # Flag biner (Ya=1, Tidak=0)
    for col in ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']:
        df[f'{col}_yes'] = (df[col] == 'Ya').astype(int)

    # Dummy untuk furnishingstatus
    furn_dummies = pd.get_dummies(df['furnishingstatus'], prefix='furnishingstatus')
    # Rename kolom dummy agar cocok persis dengan scaler.feature_names_in_
    furn_rename = {
        'furnishingstatus_Furnished':       'furnishingstatus_furnished',
        'furnishingstatus_Semi-Furnished': 'furnishingstatus_semi-furnished',
        'furnishingstatus_Unfurnished':    'furnishingstatus_unfurnished'
    }
    furn_dummies.rename(columns=furn_rename, inplace=True)
    # Pastikan semua level hadir
    for col in ['furnishingstatus_furnished','furnishingstatus_semi-furnished','furnishingstatus_unfurnished']:
        if col not in furn_dummies:
            furn_dummies[col] = 0

    df = pd.concat([df, furn_dummies], axis=1)

    # Drop kolom mentah non-numerik
    df.drop([
        'mainroad','guestroom','basement','hotwaterheating',
        'airconditioning','prefarea','furnishingstatus'
    ], axis=1, inplace=True)

    return df

feat = make_features(raw)

# ——————————————————————————————————————————————
# 4) Reorder kolom sesuai scaler.feature_names_in_
feat = feat[feature_names]

# ——————————————————————————————————————————————
# 5) Fungsi prediksi
def predict_price(df):
    # Scale → select → reshape → infer
    X_scaled = scaler.transform(df)
    X_sel    = selector.transform(X_scaled).astype(np.float32)

    shape_in = in_det[0]['shape']
    if len(shape_in) == 3:
        X_input = X_sel.reshape(1, X_sel.shape[1], 1)
    else:
        X_input = X_sel.reshape(1, -1)

    interp.set_tensor(in_det[0]['index'], X_input)
    interp.invoke()
    result = interp.get_tensor(out_det[0]['index'])
    return result[0][0]

# ——————————————————————————————————————————————
# 6) UI: tombol prediksi & tampilkan hasil
if st.button('Prediksi Harga'):
    price   = predict_price(feat)

    price_full   = price * 1_000

    # quick‑hack: perkalian konstanta
    adj_factor   = 50  # sesuaikan untuk “mengangkat” skala
    price_scaled = price_full * adj_factor

    st.subheader('Hasil Prediksi (Skala Diubah)')
    st.success(f"🏷️ Rp {price_scaled:,.0f}")

# ——————————————————————————————————————————————
# 7) Info Dataset & Model (opsional)
with st.expander('ℹ️ Tentang Dataset'):
    st.write(pd.read_csv('Housing.csv').head(3))
    st.markdown("""
- **price**: target  
- Fitur numerik: `area`, `bedrooms`, `bathrooms`, `stories`, `parking`  
- Fitur rasio dan flag biner, serta dummy `furnishingstatus`  
""")

with st.expander('ℹ️ Tentang Model'):
    st.markdown("""
- Model di-export ke TensorFlow Lite untuk efisiensi.  
- Preprocessing: `scaler.pkl` + `feature_selector.pkl`.  
""")
