import streamlit as st
import pandas as pd         
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import base64

st.set_page_config(page_title="custom practice", page_icon=":bar_chart:", layout="centered", initial_sidebar_state="expanded")
st.title("Flower ML Predictor")
st.markdown("This is a simple ML predictor app built with Streamlit. You can select a model, input feature values, and get predictions.")
models= "MODEL_FILES"

# back groun video 
def set_video_bg(video_file):
    # 1. ویڈیو فائل کو پڑھنا
    with open(video_file, "rb") as f:
        video_bytes = f.read()

    # 2. Base64 میں تبدیل کرنا
    b64_video = base64.b64encode(video_bytes).decode()

    # 3. HTML اور CSS کے ذریعے ویڈیو کو لائیو پس منظر میں لگانا
    video_html = f"""
    <style>
    #bg-video {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        z-index: -1;
        object-fit: cover;
        filter: brightness(0.6); /* تاکہ تحریر صاف نظر آئے */
    }}
    .stApp {{
        background: transparent;
    }}
    </style>
    <video autoplay loop muted playsinline id="bg-video">
        <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)


set_video_bg("bg_video.mp4")

# 1. Load the model files from the specified directory
available = [i.replace("_schema.json","") for i in os.listdir(models) if i.endswith ("_schema.json")]
if not available:
    st.warning("No model files found in the specified directory.")
    st.stop()
choice = st.selectbox("Select a model",available,format_func= lambda x: x.replace('_', ' ').title())

# 2 Load the selected model's schema and model file

@st.cache_data
def load_model(name):
    model=joblib.load(f"{models}\{name}_model.pkl")
    scaler=joblib.load(f"{models}\{name}_scaler.pkl")
    schema=json.load(open(f"{models}\{name}_schema.json"))    
    return model,scaler,schema
model,scaler,schema=load_model(choice)

# 3  build input from schema
st.markdown("## Enter Features values")
user_input={}

# slider kay irdgid aik box bunana
st.markdown(
    """
    <style>
    [data-testid="stForm"] {
        border: 2px solid rgba(255, 255, 255, 0.5) !important; /* یہاں سے بارڈر کو جتنا چاہیں ڈارک یا بولڈ کر لیں */
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(0.3, 0.3, 0.3, 0.3); /* ہلکا ڈارک بیک گراؤنڈ */
        box-shadow: 0 4px 15px rgba(0.4, 0.4, 0, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.form("Prediction form"):
    col=st.columns(2)
    for i,feature in enumerate(schema["features"]):
        target_col=col[i%2]
        user_input[feature["name"]]=target_col.slider(
            feature["name"],
            min_value=float(feature["min"]),
            max_value=float(feature["max"]),
            value=float(feature['mean']),
            step=float(feature['step'])
               )
    submitted=st.form_submit_button("Predict")

# 4 Make prediction and display results
if submitted:
    feature_order=[feature["name"] for feature in schema["features"]]
    input_df=pd.DataFrame([user_input])[feature_order]
    input_scaled=scaler.transform(input_df) if scaler is not None else input_df.values
    # Regression task
    if schema["task_type"]=="regression":
        prediction=model.predict(input_scaled)[0]
        st.success(f"The predicted value is: {prediction:.2f}")
    # classification task
    elif schema["task_type"] == "classification":
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        class_names = ["Setosa", "Versicolor", "Virginica"]
        st.success(f"The predicted class is: **{class_names[int(prediction)]}**")
        
        fig, ax = plt.subplots(figsize=(6, 3)); fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
        sns.barplot(x=class_names, y=prediction_proba, ax=ax, palette="viridis", alpha=0.8); plt.ylim(0, 1.15)
        for p in ax.patches: ax.annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', color='white', fontsize=10, xytext=(0, 3), textcoords='offset points')
        ax.tick_params(colors='white');_= [s.set_color('white') for s in ax.spines.values()]
        st.pyplot(fig)
    else:
        st.error("Unknown task type specified in the schema.")

