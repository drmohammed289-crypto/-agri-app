import streamlit as st
import pandas as pd
import statsmodels.api as sm
from google import genai
from io import StringIO

st.set_page_config(page_title="منصة الاقتصاد الزراعي الذكية", page_icon="🌾", layout="wide")

st.title("🌾 منصة تحليل الاقتصاد الزراعي والقياس القياسي")
st.markdown("منصة بحثية متقدمة لتحليل بيانات الإنتاج الزراعي، تشغيل النماذج القياسية (Econometric Models)، واستخراج التقارير الاقتصادية المدعومة بالذكاء الاصطناعي.")

st.sidebar.header("⚙️ إعدادات النموذج والبيانات")
api_key = st.sidebar.text_input("أدخل مفتاح Gemini API", type="password")

default_data = "Yield,Water,Fertilizer\n3.5,100,50\n4.0,120,60\n5.2,150,75\n6.0,180,90\n7.1,200,100"
data_input = st.sidebar.text_area("أدخل بيانات الجدول (CSV format)", default_data)

try:
    df = pd.read_csv(StringIO(data_input))
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 جدول البيانات المدخلة:")
        st.dataframe(df, use_container_width=True)

    with col2:
        st.subheader("🛠️ لوحة التحكم:")
        run_analysis = st.button("تشغيل التحليل القياسي وتقرير الذكاء الاصطناعي", type="primary")

    if run_analysis:
        X = df[['Water', 'Fertilizer']]
        X = sm.add_constant(X)
        y = df['Yield']
        model = sm.OLS(y, X).fit()
        
        st.divider()
        st.subheader("📈 نتائج التحليل الإحصائي والقياسي (OLS Summary):")
        st.text(str(model.summary()))
        
        if api_key:
            st.divider()
            st.subheader("🤖 التقرير الاقتصادي التحليلي (مُولد بالذكاء الاصطناعي):")
            with st.spinner("جاري تحليل المعلمات الاقتصادية وكتابة التقرير الأكاديمي..."):
                client = genai.Client(api_key=api_key)
                prompt = f"""بصفتك باحثاً وخبيراً في الاقتصاد الزراعي، قم بتحليل مخرجات نموذج القياس الاقتصادي التالي واكتب تقريراً أكاديمياً مبسطاً يفسر أثر مياه الري والأسمدة على الإنتاجية الزراعية:\n{model.summary()}"""
                response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                st.success("تم توليد التقرير بنجاح!")
                st.write(response.text)
        else:
            st.warning("⚠️ أدخل مفتاح Gemini API في الشريط الجانبي لتوليد التقرير تلقائياً.")
except Exception as e:
    st.error(f"حدث خطأ: {e}")
