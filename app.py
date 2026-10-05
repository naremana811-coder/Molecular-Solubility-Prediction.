import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors

st.set_page_config(page_title="Molecular Solubility Predictor", layout="wide")

st.title("🧪 تنبؤ الذوبانية الجزيئية (Molecular Solubility Prediction)")
st.write("تطبيق مخصص لتوقع ذوبانية المركبات الكيميائية باستخدام نماذج التعلم الآلي و RDkit")

def generate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        Descriptors.MolLogP(mol),
        Descriptors.MolWt(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.TPSA(mol)
    ]

@st.cache_resource
def load_model():
    return joblib.load('solubility_model.pkl')

try:
    model = load_model()
except Exception:
    st.error("لم يتم العثور على ملف النموذج `solubility_model.pkl`. يرجى حفظه أولاً.")
    st.stop()

tab1, tab2 = st.tabs(["🔍 تجربة مركب منفرد", "📁 معالجة ملف بيانات كبير"])

with tab1:
    st.subheader("توقع ذوبانية مركب بواسطة SMILES")
    single_smiles = st.text_input("أدخل كود SMILES للمركب:", "CC(=O)OC1=CC=CC=C1C(=O)O")
    
    if st.button("توقع الذوبانية"):
        desc = generate_descriptors(single_smiles)
        if desc is None:
            st.error("صيغة SMILES غير صحيحة.")
        else:
            df_features = pd.DataFrame([desc], columns=['MolLogP', 'MolWt', 'NumRotatableBonds', 'TPSA'])
            prediction = model.predict(df_features)[0]
            st.success(f"الذوبانية المتوقعة (Log Solubility): **{prediction:.3f}**")
            st.dataframe(df_features)

with tab2:
    st.subheader("رفع ملف يحتوي على بيانات كثيرة")
    uploaded_file = st.file_uploader("قم برفع ملف CSV يحتوي على عمود باسم SMILES:", type=['csv'])
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write(f"عدد الصفوف: **{len(input_df)}**")
        
        if 'SMILES' in input_df.columns:
            if st.button("بدء التنبؤ للبيانات كاملة"):
                with st.spinner("جاري معالجة المركبات..."):
                    results = []
                    valid_mask = []
                    for smiles in input_df['SMILES']:
                        desc = generate_descriptors(str(smiles))
                        if desc is None:
                            results.append([np.nan]*4)
                            valid_mask.append(False)
                        else:
                            results.append(desc)
                            valid_mask.append(True)
                    
                    features_df = pd.DataFrame(results, columns=['MolLogP', 'MolWt', 'NumRotatableBonds', 'TPSA'])
                    
                    preds = []
                    for idx, row in features_df.iterrows():
                        if valid_mask[idx]:
                            preds.append(model.predict([row.values])[0])
                        else:
                            preds.append(np.nan)
                    
                    input_df['Predicted_Log_Solubility'] = preds
                    st.success("تم التنبؤ لجميع البيانات بنجاح!")
                    st.dataframe(input_df.head(20))
                    
                    csv = input_df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ تنزيل النتائج (CSV)", data=csv, file_name='solubility_predictions.csv', mime='text/csv')
        else:
            st.error("تأكد أن الملف يحتوي على عمود باسم 'SMILES'")
