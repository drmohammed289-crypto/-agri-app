import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import linprog
import statsmodels.api as sm
import streamlit as st

# إعدادات صفحة التطبيق
st.set_page_config(
    page_title="منصة الخبير الاقتصادي والقياسي الذكي",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# 1. نظام الحماية بكلمة المرور
# ---------------------------------------------------------
def check_password():
  """التحقق من كلمة المرور"""

  def password_entered():
    # 🔑 يمكنك تغيير كلمة المرور هنا بين علامتي التنصيص بدلاً من '12345'
    if st.session_state["password"] == "12345":
      st.session_state["password_correct"] = True
      del st.session_state["password"]
    else:
      st.session_state["password_correct"] = False

  if "password_correct" not in st.session_state:
    st.markdown("## 🔒 المنصة محمية بكلمة مرور")
    st.write("الرجاء إدخال كود المرور للوصول إلى المنصة:")
    st.text_input(
        "كلمة المرور:",
        type="password",
        on_change=password_entered,
        key="password",
    )
    return False
  elif not st.session_state["password_correct"]:
    st.markdown("## 🔒 المنصة محمية بكلمة مرور")
    st.text_input(
        "كلمة المرور:",
        type="password",
        on_change=password_entered,
        key="password",
    )
    st.error("😕 كلمة المرور غير صحيحة، يرجى المحاولة مرة أخرى.")
    return False
  else:
    return True


if not check_password():
  st.stop()

# ---------------------------------------------------------
# 🚀 واجهة التطبيق الرئيسية
# ---------------------------------------------------------
st.title("🌾 منصة الخبير الاقتصادي والقياسي الذكي")
st.write(
    "منصة بحثية واحترافية متكاملة للتحليلات القياسية، الإحصاء الوصفي،"
    " الاستشارات الاقتصادية المتقدمة، وجلب البيانات الموثقة."
)

# القائمة الجانبية لتحديد أقسام المنصة (تم فصل الأقسام بوضوح)
st.sidebar.header("⚙️ إعدادات المنصة")
app_mode = st.sidebar.radio(
    "اختر قسم العمل الأساسي:",
    [
        "📊 تحليل البيانات والنماذج القياسية",
        "🌐 بوابة جمع البيانات والمؤشرات",
        "👨‍🏫 المستشار الاقتصادي والقياسي",
    ],
)

# =========================================================
# القسم الأول: تحليل البيانات والنماذج القياسية والإحصاء الوصفي
# =========================================================
if app_mode == "📊 تحليل البيانات والنماذج القياسية":
  st.sidebar.header("📁 تحميل بياناتك الخاصة")
  uploaded_file = st.sidebar.file_uploader(
      "قم برفع ملف البيانات (Excel أو CSV):", type=["xlsx", "xls", "csv"]
  )

  if uploaded_file is not None:
    try:
      if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
      else:
        df = pd.read_excel(uploaded_file)
      st.sidebar.success("تم تحميل الملف بنجاح! 🎉")

      st.subheader("📊 جدول البيانات النشط للتحليل:")
      st.dataframe(df, use_container_width=True)

      columns_list = df.columns.tolist()

      st.sidebar.header("🎛️ لوحة التحكم والاختيار")
      model_choice = st.sidebar.selectbox(
          "اختر نموذج التحليل أو الإحصاء:",
          [
              "0. مقاييس النزعة المركزية والتشتت الإحصائي",
              "1. دالة الإنتاج الخطية (OLS)",
              "2. دالة إنتاج كوب-دوجلاس (Cobb-Douglas)",
              "3. دالة الإنتاج التربيعية (Quadratic - تناقص الغلة)",
              "4. تحليل الاتجاه العام (Trend Analysis)",
              (
                  "5. تحليل الكفاءة باستخدام مغلف البيانات (DEA - Data Envelopment"
                  " Analysis)"
              ),
              "6. حساب الهوامش التسويقية (Marketing Margins)",
              "7. تحليل حد الإنتاج القياسي (Frontier Analysis - COLS)",
              "8. تحليل التكاليف وصافي العائد (Cost & Profitability Analysis)",
          ],
      )

      # 0. مقاييس النزعة المركزية والتشتت
      if model_choice == "0. مقاييس النزعة المركزية والتشتت الإحصائي":
        st.subheader("📈 تقرير مقاييس النزعة المركزية والتشتت الإحصائي")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_stats_cols = st.multiselect(
            "اختر المتغيرات الرقمية لحساب المقاييس الإحصائية:", num_cols
        )

        if selected_stats_cols:
          try:
            sub_df = df[selected_stats_cols]
            stats_df = sub_df.describe().T
            stats_df["التباين"] = sub_df.var()
            stats_df["معامل الاختلاف (%)"] = (
                sub_df.std() / sub_df.mean()
            ) * 100
            stats_df["المنوال"] = sub_df.mode().iloc[0]

            stats_df = stats_df.rename(
                columns={
                    "count": "عدد المشاهدات",
                    "mean": "المتوسط الحسابي",
                    "std": "الانحراف المعياري",
                    "min": "الحد الأدنى",
                    "25%": "الربيع الأول (25%)",
                    "50%": "الوسيط (50%)",
                    "75%": "الربيع الثالث (75%)",
                    "max": "الحد الأقصى",
                }
            )

            ordered_cols = [
                "عدد المشاهدات",
                "المتوسط الحسابي",
                "الوسيط",
                "المنوال",
                "الانحراف المعياري",
                "التباين",
                "معامل الاختلاف (%)",
                "الحد الأدنى",
                "الحد الأقصى",
            ]
            stats_df = stats_df[
                [c for c in ordered_cols if c in stats_df.columns]
            ]
            st.dataframe(stats_df, use_container_width=True)

            st.markdown("### 📊 تمثيل بياني للمتوسطات الحسابية:")
            st.bar_chart(sub_df.mean())

          except Exception as ex:
            st.error(f"حدث خطأ أثناء حساب الإحصاءات: {ex}")
        else:
          st.warning("يرجى اختيار متغير واحد على الأقل.")

      # 1. دالة الإنتاج الخطية (OLS)
      elif model_choice == "1. دالة الإنتاج الخطية (OLS)":
        st.subheader("📈 نتائج دالة الإنتاج الخطية (OLS)")
        col1, col2 = st.columns(2)
        with col1:
          y_col = st.selectbox(
              "اختر المتغير التابع (الإنتاج / العائد):",
              columns_list,
              key="ols_y",
          )
        with col2:
          x_cols = st.multiselect(
              "اختر المتغيرات المستقلة (المدخلات):",
              [c for c in columns_list if c != y_col],
              key="ols_x",
          )

        if st.button("🚀 تشغيل التحليل وإصدار التقرير", key="btn_ols"):
          if y_col and x_cols:
            try:
              temp_df = df[[y_col] + x_cols].apply(
                  pd.to_numeric, errors="coerce"
              )
              temp_df = temp_df.dropna()
              y = temp_df[y_col]
              X = sm.add_constant(temp_df[x_cols])
              model = sm.OLS(y, X).fit()
              st.text(model.summary().as_text())

              st.markdown("### 📉 الرسم البياني: القيم الفعلية مقابل القيم المقدرة")
              fig, ax = plt.subplots(figsize=(8, 5))
              ax.scatter(y, model.fittedvalues, color="blue", alpha=0.7)
              ax.plot(
                  [y.min(), y.max()],
                  [y.min(), y.max()],
                  "r--",
                  lw=2,
                  label="الخط المثالي (مطابقة تامة)",
              )
              ax.set_xlabel(f"القيم الفعليـة لـ ({y_col})")
              ax.set_ylabel("القيم المقدرة (Fitted Values)")
              ax.set_title("مقارنة القيم الفعلية والمقدرة لنموذج OLS")
              ax.legend()
              st.pyplot(fig)

            except Exception as ex:
              st.error(f"حدث خطأ: {ex}")
          else:
            st.warning("يرجى اختيار المتغيرات المطلوبة.")

      # 2. دالة كوب دوجلاس
      elif model_choice == "2. دالة إنتاج كوب-دوجلاس (Cobb-Douglas)":
        st.subheader("📉 نتائج دالة كوب-دوجلاس اللوغاريتمية (Log-Log Model)")
        col1, col2 = st.columns(2)
        with col1:
          y_col = st.selectbox(
              "اختر المتغير التابع (الإنتاج / العائد):", columns_list, key="cd_y"
          )
        with col2:
          x_cols = st.multiselect(
              "اختر المتغيرات المستقلة (المدخلات):",
              [c for c in columns_list if c != y_col],
              key="cd_x",
          )

        if st.button("🚀 تشغيل التحليل وإصدار التقرير", key="btn_cd"):
          if y_col and x_cols:
            try:
              temp_df = df[[y_col] + x_cols].apply(
                  pd.to_numeric, errors="coerce"
              )
              temp_df = temp_df.dropna()
              if (temp_df <= 0).any().any():
                temp_df = temp_df[(temp_df > 0).all(axis=1)]
              df_log = np.log(temp_df)
              X = sm.add_constant(df_log[x_cols])
              y = df_log[y_col]
              model = sm.OLS(y, X).fit()
              st.text(model.summary().as_text())
              returns_to_scale = model.params[x_cols].sum()
              st.metric("إجمالي عوائد الحجم", f"{returns_to_scale:.4f}")

              st.markdown("### 📊 الرسم البياني: مطابقة النموذج اللوغاريتمي")
              fig, ax = plt.subplots(figsize=(8, 5))
              ax.scatter(y, model.fittedvalues, color="green", alpha=0.7)
              ax.plot(
                  [y.min(), y.max()],
                  [y.min(), y.max()],
                  "r--",
                  lw=2,
                  label="المطابقة المثالية",
              )
              ax.set_xlabel("القيم الفعلية اللوغاريتمية")
              ax.set_ylabel("القيم المقدرة اللوغاريتمية")
              ax.set_title("تقييم دالة إنتاج كوب-دوجلاس")
              ax.legend()
              st.pyplot(fig)

            except Exception as ex:
              st.error(f"حدث خطأ: {ex}")
          else:
            st.warning("يرجى اختيار المتغيرات.")

      # 3. دالة الإنتاج التربيعية
      elif model_choice == "3. دالة الإنتاج التربيعية (Quadratic)":
        st.subheader("📐 نتائج دالة الإنتاج التربيعية (لقياس تناقص الغلة)")
        col1, col2 = st.columns(2)
        with col1:
          y_col = st.selectbox(
              "اختر المتغير التابع (الإنتاج):", columns_list, key="q_y"
          )
        with col2:
          x_col = st.selectbox(
              "اختر المتغير المراد اختبار تناقص غلته:",
              [c for c in columns_list if c != y_col],
              key="q_x",
          )

        if st.button("🚀 تشغيل التحليل وإصدار التقرير", key="btn_q"):
          if y_col and x_col:
            try:
              temp_df = df[[y_col, x_col]].apply(
                  pd.to_numeric, errors="coerce"
              )
              temp_df = temp_df.dropna()
              temp_df["X_sq"] = temp_df[x_col] ** 2
              X = sm.add_constant(temp_df[[x_col, "X_sq"]])
              y = temp_df[y_col]
              model = sm.OLS(y, X).fit()
              st.text(model.summary().as_text())

              st.markdown("### 📈 منحنى الإنتاج التربيعي وتناقص الغلة:")
              fig, ax = plt.subplots(figsize=(9, 5))
              sorted_idx = np.argsort(temp_df[x_col])
              ax.scatter(
                  temp_df[x_col],
                  y,
                  color="purple",
                  alpha=0.6,
                  label="البيانات الفعلية",
              )
              ax.plot(
                  temp_df[x_col].iloc[sorted_idx],
                  model.fittedvalues.iloc[sorted_idx],
                  color="red",
                  lw=2.5,
                  label="منحنى الإنتاج التقديري",
              )
              ax.set_xlabel(x_col)
              ax.set_ylabel(y_col)
              ax.set_title("منحنى دالة الإنتاج التربيعية")
              ax.legend()
              st.pyplot(fig)

            except Exception as ex:
              st.error(f"حدث خطأ: {ex}")

      # 4. تحليل الاتجاه العام
      elif model_choice == "4. تحليل الاتجاه العام (Trend Analysis)":
        st.subheader("📅 تحليل الاتجاه العام للمتغيرات عبر الزمن")
        col1, col2 = st.columns(2)
        with col1:
          year_col = st.selectbox(
              "اختر عمود الزمن أو السنوات:", columns_list, key="t_yr"
          )
        with col2:
          target_var = st.selectbox(
              "اختر المتغير المراد دراسة اتجاهه العام:",
              [c for c in columns_list if c != year_col],
              key="t_var",
          )

        if st.button("🚀 تشغيل التحليل وإصدار التقرير", key="btn_t"):
          if year_col and target_var:
            try:
              temp_df = df[[year_col, target_var]].apply(
                  pd.to_numeric, errors="coerce"
              )
              temp_df = temp_df.dropna()
              X_trend = sm.add_constant(temp_df[year_col])
              y_trend = temp_df[target_var]
              trend_model = sm.OLS(y_trend, X_trend).fit()
              st.text(trend_model.summary().as_text())

              st.markdown("### 📈 الرسم البياني لخط الاتجاه الزمني:")
              fig, ax = plt.subplots(figsize=(10, 5))
              ax.plot(
                  temp_df[year_col],
                  temp_df[target_var],
                  marker="o",
                  label="القيم الفعلية",
                  color="blue",
              )
              ax.plot(
                  temp_df[year_col],
                  trend_model.fittedvalues,
                  color="red",
                  linestyle="--",
                  lw=2,
                  label="خط الاتجاه العام",
              )
              ax.set_xlabel(year_col)
              ax.set_ylabel(target_var)
              ax.set_title("تحليل الاتجاه العام عبر الزمن")
              ax.legend()
              st.pyplot(fig)

            except Exception as ex:
              st.error(f"حدث خطأ: {ex}")

      # 5. تحليل الكفاءة DEA
      elif model_choice == (
          "5. تحليل الكفاءة باستخدام مغلف البيانات (DEA - Data Envelopment"
          " Analysis)"
      ):
        st.subheader("📐 تحليل الكفاءة الفنية باستخدام مغلف البيانات (DEA)")
        col1, col2 = st.columns(2)
        with col1:
          y_col = st.selectbox(
              "اختر متغير المخرج (الإنتاج):", columns_list, key="dea_y"
          )
        with col2:
          x_cols = st.multiselect(
              "اختر متغيرات المدخلات:",
              [c for c in columns_list if c != y_col],
              key="dea_x",
          )

        if st.button("🚀 تشغيل التحليل وإصدار التقرير", key="btn_dea"):
          if y_col and x_cols:
            try:
              temp_df = df[[y_col] + x_cols].apply(
                  pd.to_numeric, errors="coerce"
              )
              temp_df = temp_df.dropna()
              inputs = temp_df[x_cols].values
              outputs = temp_df[y_col].values
              n_dmu = len(temp_df)
              X_mat = inputs.T
              Y_mat = outputs.reshape(1, n_dmu)
              eff_list = []
              for k in range(n_dmu):
                x_k = inputs[k]
                y_k = outputs[k]
                c_lp = np.array([1.0] + [0.0] * n_dmu)
                A_inputs = np.column_stack((-x_k, X_mat))
                b_inputs = np.zeros(X_mat.shape[0])
                A_outputs = np.column_stack((np.zeros(Y_mat.shape[0]), -Y_mat))
                b_outputs = np.array([-y_k])
                A_ub = np.vstack((A_inputs, A_outputs))
                b_ub = np.concatenate((b_inputs, b_outputs))
                bounds = [(0, None)] + [(0, None)] * n_dmu
                res = linprog(
                    c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
                )
                eff_list.append(res.x[0] if res.success else np.nan)
              temp_df["Technical_Efficiency (DEA)"] = eff_list
              st.dataframe(temp_df, use_container_width=True)

              st.markdown("### 📊 توزيع درجات الكفاءة الفنية للوحدات:")
              fig, ax = plt.subplots(figsize=(10, 5))
              ax.bar(
                  range(len(temp_df)),
                  temp_df["Technical_Efficiency (DEA)"],
                  color="teal",
              )
              ax.axhline(
                  1.0,
                  color="red",
                  linestyle="--",
                  label="الكفاءة التامة (100%)",
              )
              ax.set_xlabel("وحدات اتخاذ القرار (DMUs)")
              ax.set_ylabel("درجة الكفاءة")
              ax.set_title("تحليل الكفاءة الفنية (DEA)")
              ax.legend()
              st.pyplot(fig)

            except Exception as ex:
              st.error(f"حدث خطأ: {ex}")

      # 6. الهوامش التسويقية
      elif model_choice == "6. حساب الهوامش التسويقية (Marketing Margins)":
        st.subheader("💰 تحليل الهوامش التسويقية ونصيب المزارع")
        col1, col2 = st.columns(2)
        with col1:
          farm_col = st.selectbox(
              "اختر عمود سعر المزرعة (Farm Price):", columns_list, key="m_farm"
          )
        with col2:
          retail_col = st.selectbox(
              "اختر عمود سعر التجزئة / المستهلك (Retail Price):",
              columns_list,
              key="m_retail",
          )

        if st.button("🚀 تشغيل التحليل وإصدار التقرير", key="btn_m"):
          if farm_col and retail_col:
            try:
              temp_df = df[[farm_col, retail_col]].apply(
                  pd.to_numeric, errors="coerce"
              )
              temp_df = temp_df.dropna()
              temp_df["Absolute_Margin"] = (
                  temp_df[retail_col] - temp_df[farm_col]
              )
              temp_df["Percentage_Margin (%)"] = (
                  temp_df["Absolute_Margin"] / temp_df[retail_col]
              ) * 100
              temp_df["Farmer_Share (%)"] = (
                  temp_df[farm_col] / temp_df[retail_col]
              ) * 100
              st.dataframe(temp_df, use_container_width=True)

              st.markdown(
                  "### 📊 مقارنة أسعار المزرعة وأسعار التجزئة (الهوامش التسويقية):"
              )
              fig, ax = plt.subplots(figsize=(10, 5))
              x_indices = range(len(temp_df))
              ax.plot(
                  x_indices,
                  temp_df[retail_col],
                  marker="o",
                  label="سعر التجزئة",
                  color="orange",
              )
              ax.plot(
                  x_indices,
                  temp_df[farm_col],
                  marker="s",
                  label="سعر المزرعة",
                  color="green",
              )
              ax.set_xlabel("المشاهدات / الأسواق")
              ax.set_ylabel("القيمة / السعر")
              ax.set_title("تحليل الهوامش التسويقية")
              ax.legend()
              st.pyplot(fig)

            except Exception as ex:
              st.error(f"حدث خطأ: {ex}")

      # 7. تحليل حد الإنتاج القياسي (Frontier)
      elif model_choice == (
          "7. تحليل حد الإنتاج القياسي (Frontier Analysis - COLS)"
      ):
        st.subheader("⚡ تحليل حد الإنتاج وتقدير الكفاءة (COLS Frontier)")
        col1, col2 = st.columns(2)
        with col1:
          y_col = st.selectbox(
              "اختر المتغير التابع (الإنتاج):", columns_list, key="f_y"
          )
        with col2:
          x_cols = st.multiselect(
              "اختر المتغيرات المستقلة (المدخلات):",
              [c for c in columns_list if c != y_col],
              key="f_x",
          )

        if st.button("🚀 تشغيل التحليل وإصدار التقرير", key="btn_f"):
          if y_col and x_cols:
            try:
              temp_df = df[[y_col] + x_cols].apply(
                  pd.to_numeric, errors="coerce"
              )
              temp_df = temp_df.dropna()
              if (temp_df <= 0).any().any():
                temp_df = temp_df[(temp_df > 0).all(axis=1)]
              df_log = np.log(temp_df)
              X = sm.add_constant(df_log[x_cols])
              y = df_log[y_col]
              model = sm.OLS(y, X).fit()
              residuals = model.resid
              temp_df["Technical_Efficiency"] = np.exp(
                  residuals - residuals.max()
              )
              st.text(model.summary().as_text())
              st.dataframe(temp_df, use_container_width=True)

              st.markdown("### 📊 توزيع درجات الكفاءة الفنية:")
              fig, ax = plt.subplots(figsize=(9, 4))
              ax.hist(
                  temp_df["Technical_Efficiency"],
                  bins=10,
                  color="skyblue",
                  edgecolor="black",
              )
              ax.set_xlabel("درجة الكفاءة الفنية")
              ax.set_ylabel("التكرار")
              ax.set_title("توزيع الكفاءة لنموذج الحد الإنتاجي")
              st.pyplot(fig)

            except Exception as ex:
              st.error(f"حدث خطأ: {ex}")

      # 8. تحليل التكاليف وصافي العائد
      elif model_choice == (
          "8. تحليل التكاليف وصافي العائد (Cost & Profitability Analysis)"
      ):
        st.subheader("💵 تحليل التكاليف الكلية، الإيرادات، وصافي العائد الاقتصادي")
        col1, col2, col3 = st.columns(3)
        with col1:
          rev_col = st.selectbox(
              "اختر عمود إجمالي الإيرادات (Total Revenue):",
              columns_list,
              key="c_rev",
          )
        with col2:
          cost_col = st.selectbox(
              "اختر عمود إجمالي التكاليف (Total Costs):",
              columns_list,
              key="c_cost",
          )
        with col3:
          yield_col = st.selectbox(
              "اختر عمود الإنتاجية أو المساحة (اختياري):",
              columns_list,
              key="c_yield",
          )

        if st.button("🚀 تشغيل التحليل المالي واحتساب المؤشرات", key="btn_cost"):
          if rev_col and cost_col:
            try:
              temp_df = df.copy()
              temp_df[rev_col] = pd.to_numeric(
                  temp_df[rev_col], errors="coerce"
              )
              temp_df[cost_col] = pd.to_numeric(
                  temp_df[cost_col], errors="coerce"
              )
              temp_df = temp_df.dropna(subset=[rev_col, cost_col])
              temp_df["صافي العائد (Net Return)"] = (
                  temp_df[rev_col] - temp_df[cost_col]
              )
              temp_df["نسبة العائد إلى التكلفة (BCR)"] = (
                  temp_df[rev_col] / temp_df[cost_col]
              )
              temp_df["معدل الربحية (%)"] = (
                  temp_df["صافي العائد (Net Return)"] / temp_df[cost_col]
              ) * 100
              st.dataframe(temp_df, use_container_width=True)
              avg_net = temp_df["صافي العائد (Net Return)"].mean()
              avg_bcr = temp_df["نسبة العائد إلى التكلفة (BCR)"].mean()
              st.success(
                  f"• **متوسط صافي العائد:** {avg_net:,.2f}\n• **متوسط نسبة"
                  f" العائد إلى التكلفة (BCR):** {avg_bcr:,.2f}"
              )

              st.markdown("### 📊 مقارنة الإيرادات والتكاليف الكلية:")
              fig, ax = plt.subplots(figsize=(10, 5))
              width = 0.35
              x = np.arange(len(temp_df))
              ax.bar(
                  x - width / 2,
                  temp_df[rev_col],
                  width,
                  label="إجمالي الإيرادات",
                  color="green",
              )
              ax.bar(
                  x + width / 2,
                  temp_df[cost_col],
                  width,
                  label="إجمالي التكاليف",
                  color="crimson",
              )
              ax.set_xlabel("المشاهدات")
              ax.set_ylabel("القيمة النقدية")
              ax.set_title("تحليل التكاليف والإيرادات الكلية")
              ax.legend()
              st.pyplot(fig)

            except Exception as ex:
              st.error(f"حدث خطأ أثناء إجراء تحليل التكاليف: {ex}")

    except Exception as e:
      st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
  else:
    st.info("👈 يرجى رفع ملف البيانات الخاص بك من القائمة الجانبية لبدء العمل.")


# =========================================================
# القسم الثاني: بوابة جمع البيانات والمؤشرات (مستقلة تماماً)
# =========================================================
elif app_mode == "🌐 بوابة جمع البيانات والمؤشرات":
  st.subheader("🌐 بوابة جمع البيانات والمؤشرات العالمية والمحلية")
  st.write(
      "من هنا يمكنك تجميع بيانات المؤشرات الاقتصادية والزراعية المعتمدة (البنك"
      " الدولي، الفاو، البيانات الرسمية) وعرضها وتحميلها مباشرة كملف إكسيل"
      " جاهز."
  )

  col_b1, col_b2 = st.columns(2)
  with col_b1:
    country_code = st.text_input(
        "كود الدولة الثلاثي (مثال: EGY لمصر، USA لأمريكا، WLD للعالم):",
        value="EGY",
    )
  with col_b2:
    indicator_choice = st.selectbox(
        "اختر المؤشر الاقتصادي أو الزراعي المطلوب جمعه:",
        [
            (
                "الزراعة والغابات والصيد كنسبة من الناتج المحلي الإجمالي"
                " (NV.AGR.TOTL.ZS)"
            ),
            ("إجمالي إنتاج الحبوب كجم لكل هكتار (AG.YLD.CREL.KG)"),
            ("الأراضي الزراعية كنسبة من المساحة الكلية (AG.LND.AGRI.ZS)"),
            ("السكان الريفيون كنسبة من إجمالي السكان (SP.RUR.TOTL.ZS)"),
        ],
    )

  indicator_code = indicator_choice.split("(")[-1].replace(")", "")

  if st.button("📥 جلب البيانات وتجميعها كشيت إكسيل"):
    if not country_code or not indicator_code:
      st.warning("يرجى إدخال بيانات الدولة والمؤشر المطلوبة.")
    else:
      with st.spinner(
          "جاري الاتصال بقواعد البيانات وتجميع البيانات الموثقة..."
      ):
        try:
          url = f"http://api.worldbank.org/v2/country/{country_code.strip()}/indicator/{indicator_code.strip()}?format=json&per_page=100"
          response = requests.get(url)
          if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and data[1]:
              records = [
                  {
                      "الدولة": entry.get("country", {}).get(
                          "value", country_code
                      ),
                      "السنة": entry.get("date"),
                      "اسم المؤشر": entry.get("indicator", {}).get(
                          "value", indicator_code
                      ),
                      "القيمة": entry.get("value"),
                  }
                  for entry in data[1]
              ]
              df_wb = pd.DataFrame(records)
              st.success("🎉 تم تجميع البيانات بنجاح من المصادر الرسمية!")
              st.dataframe(df_wb, use_container_width=True)

              st.markdown("### 📈 تمثيل بياني لتطور المؤشر عبر الزمن:")
              df_plot = df_wb.dropna(subset=["القيمة"])
              if not df_plot.empty:
                st.line_chart(df_plot.set_index("السنة")["القيمة"])

              # توثيق المصدر والمراجع الرسمية
              st.markdown("---")
              st.markdown("### 📚 توثيق المصدر والمراجع المعتمدة:")
              st.info(
                  "• **المصدر الرئيسي:** مجموعة البنك الدولي (World Bank Open"
                  f" Data API)\n• **كود المؤشر:** `{indicator_code}`\n• **الدولة"
                  f" المستهدفة:** `{country_code.upper()}`\n• **الجهة المرجعية"
                  " المساندة:** منظمة الأغذية والزراعة (FAOSTAT)."
              )

              csv_data = df_wb.to_csv(index=False).encode("utf-8-sig")
              st.download_button(
                  label="💾 تحميل البيانات المجمعة كملف CSV (جاهز للإكسيل)",
                  data=csv_data,
                  file_name=f"DataCollection_{country_code}_{indicator_code}.csv",
                  mime="text/csv",
              )
            else:
              st.error(
                  "❌ لم يتم العثور على بيانات لهذا المؤشر أو كود الدولة غير"
                  " صحيح."
              )
          else:
            st.error("فشل الاتصال بخادم البيانات.")
        except Exception as ex:
          st.error(f"حدث خطأ أثناء جلب البيانات: {ex}")


# =========================================================
# القسم الثالث: المستشار الاقتصادي والقياسي (الاستشارات)
# =========================================================
elif app_mode == "👨‍🏫 المستشار الاقتصادي والقياسي":
  st.subheader("👨‍🏫 المستشار الاقتصادي والقياسي")
  st.write(
      "اطرح أي سؤال اقتصادي أو قياسي، اطلب شرحاً تفصيلياً لأي نظرية أو تحليل،"
      " وسيقوم الخبير بتقديم إجابة احترافية، دقيقة، وشاملة لجميع الاستفسارات (بما"
      " في ذلك أسعار الصرف والمعلومات والبيانات الحالية) مدعومة بالمراجع"
      " الرسمية."
  )

  user_question = st.text_area(
      "اكتب سؤالك أو استفسارك الاقتصادي هنا:",
      placeholder="مثلاً: ما هو سعر الصرف اليوم، اشرح دالة إنتاج كوب-دوجلاس، إلخ...",
      height=120,
  )

  if st.button("إرسال السؤال للحصول على الشرح المفصل"):
    if not user_question:
      st.warning("يرجى كتابة السؤال أولاً.")
    else:
      with st.spinner("جاري إعداد الشرح والتحليل العلمي المفصل مع المراجع..."):
        ai_response = ""
        api_key = None

        try:
          if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
          elif "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
          pass

        if not api_key:
          api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
              "GEMINI_API_KEY"
          )

        if api_key and len(api_key) > 5:
          try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "أنت خبير اقتصادي واقتصادي قياسي محترف ودقيق للغاية."
                            " مهمتك هي الإجابة على كافة استفسارات المستخدم بدقة"
                            " واحترافية عالية وبأسلوب علمي ومنهجي. يُرجى عدم"
                            " الامتناع عن الإجابة على أي سؤال، سواء كان يتعلق"
                            " بالنظريات الاقتصادية، النماذج القياسية، أو"
                            " المعلومات والبيانات المالية والمعطيات الحالية (مثل"
                            " أسعار الصرف اليومية والبيانات العامة). قدم دائماً"
                            " إجابات وافية، مفصلة، ومدعومة بالتحليل الدقيق والمراجع"
                            " الرسمية المعتمدة لكل ما يطلبه المستخدم."
                        ),
                    },
                    {"role": "user", "content": user_question},
                ],
            )
            ai_response = completion.choices[0].message.content
          except Exception:
            ai_response = ""

        if not ai_response:
          q_lower = user_question.strip().lower()
          if "صرف" in q_lower or "سعر" in q_lower or "عملة" in q_lower:
            ai_response = """### 💱 تقرير أسعار الصرف والمعطيات المالية:
- **التحليل المالي:** تتحدد أسعار الصرف بناءً على قوى العرض والطلب في الأسواق النقدية، ومعدلات التضخم وأسعار الفائدة.
---
📚 **المراجع والمصادر الرسمية:**
1. البنك المركزي المصري / البنوك المركزية الوطنية.
2. صندوق النقد الدولي (IMF) - قاعدة بيانات أسعار الصرف الرسمية."""
          elif "كوب" in q_lower or "cobb" in q_lower or "دوجلاس" in q_lower:
            ai_response = """### 📈 شرح دالة إنتاج كوب-دوجلاس (Cobb-Douglas Production Function):
1. **الصيغة الرياضية الأساسية:** تُكتب في الصورة الخطية اللوغاريتمية بالشكل التالي:
   $\\ln(Y) = \\beta_0 + \\beta_1 \\ln(X_1) + \\beta_2 \\ln(X_2) + \\dots + \\epsilon$
2. **المميزات:** تُثل المعلمات ($\\beta_1, \\beta_2$) مباشرةً **مرونات الإنتاج** للمدخلات.
3. **عوائد الحجم:** يتم تحديدها بجمع قيم المرونات ($\\sum \\beta_i$).
---
📚 **المراجع والمصادر الرسمية:**
1. مرجع الاقتصاد القياسي (Damodar Gujarati).
2. تقارير منظمة الفاو (FAO Economic Papers)."""
          else:
            ai_response = f"""### 💡 الاستشارة والتحليل الاقتصادي والقياسي حول: "{user_question}"
- **التحليل المنهجي:** يتطلب التعامل مع هذه الظاهرة دراسة شاملة لكافة العوامل الاقتصادية والقياسية المرتبطة بها لضمان دقة النتائج.
- **التوجيه التطبيقي:** يمكنك تطبيق النماذج المرتبطة مباشرة من قسم "تحليل البيانات والنماذج القياسية".
---
📚 **المراجع والمصادر الرسمية:**
1. المراجع الأكاديمية المعتمدة في الاقتصاد الزراعي.
2. تقارير البنك الدولي ومنظمة الفاو."""

        st.markdown("---")
        st.markdown(ai_response)
