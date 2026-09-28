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
    # 🔑 كلمة المرور الافتتاحية للمنصة
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
    "منصة بحثية وأكاديمية متكاملة للتحليلات القياسية، دالة الإنتاج، اتجاهات"
    " النمو، كفاءة الأداء، مؤشرات الأمن الغذائي والفجوات، التجارة الخارجية،"
    " والقدرة التنافسية."
)

# القائمة الجانبية لتحديد أقسام المنصة
st.sidebar.header("⚙️ إعدادات المنصة")
app_mode = st.sidebar.radio(
    "اختر قسم العمل الأساسي:",
    [
        "📊 تحليل البيانات والنماذج القياسية",
        "🌐 بوابة جمع البيانات والمؤشرات",
        "👨‍🏫 المستشار الاقتصادي والقياسي (برنامج الذكاء الاصطناعي)",
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
          "اختر نموذج التحليل أو المؤشر الاقتصادي:",
          [
              "0. مقاييس النزعة المركزية والتشتت الإحصائي",
              "1. دالة الإنتاج الخطية (OLS)",
              "2. دالة إنتاج كوب-دوجلاس (Cobb-Douglas)",
              "3. دالة الإنتاج التربيعية (Quadratic - تناقص الغلة)",
              "4. تحليل الاتجاه العام بصيغه المختلفة (خطي، أسي/نمو)",
              (
                  "5. تحليل الكفاءة باستخدام مغلف البيانات (DEA - Data Envelopment"
                  " Analysis)"
              ),
              "6. حساب الهوامش التسويقية (Marketing Margins)",
              "7. تحليل حد الإنتاج القياسي (Frontier Analysis - COLS)",
              "8. تحليل التكاليف وصافي العائد (Cost & Profitability Analysis)",
              (
                  "9. مؤشرات الأمن الغذائي والفجوات (اكتفاء، فجوة ظاهرية/حقيقية،"
                  " فترة الكفاية)"
              ),
              "10. مؤشرات التجارة الخارجية والتبعية الاقتصادية",
              (
                  "11. مؤشرات القدرة التنافسية الشاملة (RCA, النصيب السوقي,"
                  " الاختراق, السعر النسبي)"
              ),
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

      # 4. تحليل الاتجاه العام بصيغه المختلفة
      elif model_choice == "4. تحليل الاتجاه العام بصيغه المختلفة (خطي، أسي/نمو)":
        st.subheader(
            "📅 تحليل الاتجاه العام وصيغ النمو الزمنية (الخطية والأسية/اللوغاريتمية)"
        )
        col1, col2, col3 = st.columns(3)
        with col1:
          year_col = st.selectbox(
              "عمود الزمن / السنوات (Time/Year):", columns_list, key="t_yr"
          )
        with col2:
          target_var = st.selectbox(
              "المتغير المراد دراسة اتجاهه:",
              [c for c in columns_list if c != year_col],
              key="t_var",
          )
        with col3:
          trend_form = st.selectbox(
              "اختر صيغة الاتجاه العام:",
              [
                  "الخطية (Linear: Y = a + bT)",
                  "الأسية / النمو (Exponential: ln(Y) = a + bT)",
              ],
              key="t_form",
          )

        if st.button("🚀 تشغيل تحليل الاتجاه وإصدار التقرير", key="btn_t"):
          if year_col and target_var:
            try:
              temp_df = df[[year_col, target_var]].apply(
                  pd.to_numeric, errors="coerce"
              )
              temp_df = temp_df.dropna()
              T = temp_df[year_col]
              Y = temp_df[target_var]

              if "الأسية" in trend_form:
                if (Y <= 0).any():
                  st.error(
                      "❌ عذراً، الصيغة الأسية تتطلب أن تكون جميع قيم المتغير"
                      " موجبة (> 0)."
                  )
                else:
                  Y_transformed = np.log(Y)
                  X_trend = sm.add_constant(T)
                  model = sm.OLS(Y_transformed, X_trend).fit()
                  st.text(model.summary().as_text())
                  b1 = model.params.iloc[1]
                  growth_rate = (np.exp(b1) - 1) * 100
                  st.metric(
                      "معدل النمو السنوي المركب التقديري (CAGR %)",
                      f"{growth_rate:.2f}%",
                  )
                  fitted_orig = np.exp(model.fittedvalues)

                  st.markdown(
                      "### 📈 الرسم البياني لخط الاتجاه الأسي / النمو:"
                  )
                  fig, ax = plt.subplots(figsize=(10, 5))
                  ax.plot(
                      T, Y, marker="o", label="القيم الفعلية", color="blue"
                  )
                  ax.plot(
                      T,
                      fitted_orig,
                      color="red",
                      linestyle="--",
                      lw=2.5,
                      label="الاتجاه الأسي التقديري",
                  )
                  ax.set_xlabel(year_col)
                  ax.set_ylabel(target_var)
                  ax.set_title("تحليل الاتجاه العام (الصيغة الأسية/النمو)")
                  ax.legend()
                  st.pyplot(fig)
              else:
                X_trend = sm.add_constant(T)
                model = sm.OLS(Y, X_trend).fit()
                st.text(model.summary().as_text())
                b1 = model.params.iloc[1]
                st.metric(
                    "مقدار التغير السنوي المطلق (الميل b1)", f"{b1:,.4f}"
                )

                st.markdown("### 📈 الرسم البياني لخط الاتجاه الخطي:")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(T, Y, marker="o", label="القيم الفعلية", color="blue")
                ax.plot(
                    T,
                    model.fittedvalues,
                    color="red",
                    linestyle="--",
                    lw=2.5,
                    label="الاتجاه الخطي العام",
                )
                ax.set_xlabel(year_col)
                ax.set_ylabel(target_var)
                ax.set_title("تحليل الاتجاه العام (الصيغة الخطية)")
                ax.legend()
                st.pyplot(fig)

            except Exception as ex:
              st.error(f"حدث خطأ أثناء تنفيذ تحليل الاتجاه: {ex}")

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

      # 9. مؤشرات الأمن الغذائي والفجوات
      elif (
          model_choice
          == "9. مؤشرات الأمن الغذائي والفجوات (اكتفاء، فجوة ظاهرية/حقيقية، فترة الكفاية)"
      ):
        st.subheader(
            "🌾 تحليل مؤشرات الأمن الغذائي، الفجوة الظاهرية والحقيقية، وفترة كفاية"
            " الإنتاج"
        )
        c1, c2 = st.columns(2)
        with c1:
          fs_prod = st.selectbox(
              "عمود الإنتاج المحلي (Production):", columns_list, key="fs_p"
          )
          fs_imp = st.selectbox(
              "عمود الواردات (Imports):", columns_list, key="fs_i"
          )
          fs_exp = st.selectbox(
              "عمود الصادرات (Exports):", columns_list, key="fs_e"
          )
        with c2:
          fs_waste = st.selectbox(
              "عمود الفاقد والتلف (Waste/Losses - اختياري للحقيقية):",
              [None] + columns_list,
              key="fs_w",
          )
          fs_pop = st.selectbox(
              "عمود عدد السكان (Population - اختياري):",
              [None] + columns_list,
              key="fs_pop",
          )

        if st.button(
            "🚀 احتساب مؤشرات الأمن الغذائي والفجوات وفترة الكفاية",
            key="btn_fs_all",
        ):
          try:
            temp_df = df.copy()
            temp_df[fs_prod] = pd.to_numeric(
                temp_df[fs_prod], errors="coerce"
            )
            temp_df[fs_imp] = pd.to_numeric(temp_df[fs_imp], errors="coerce")
            temp_df[fs_exp] = pd.to_numeric(temp_df[fs_exp], errors="coerce")

            temp_df["إجمالي الاستهلاك المتاح"] = (
                temp_df[fs_prod] + temp_df[fs_imp] - temp_df[fs_exp]
            )
            temp_df["نسبة الاكتفاء الذاتي (SSR %)"] = np.where(
                temp_df["إجمالي الاستهلاك المتاح"] > 0,
                (temp_df[fs_prod] / temp_df["إجمالي الاستهلاك المتاح"]) * 100,
                np.nan,
            )
            temp_df["نسبة الاعتماد الاستيرادي (IDR %)"] = np.where(
                temp_df["إجمالي الاستهلاك المتاح"] > 0,
                (temp_df[fs_imp] / temp_df["إجمالي الاستهلاك المتاح"]) * 100,
                np.nan,
            )
            temp_df["الفجوة الغذائية الظاهرية"] = (
                temp_df["إجمالي الاستهلاك المتاح"] - temp_df[fs_prod]
            )

            if fs_waste and fs_waste != "None":
              temp_df[fs_waste] = pd.to_numeric(
                  temp_df[fs_waste], errors="coerce"
              )
              net_prod = temp_df[fs_prod] - temp_df[fs_waste]
              temp_df["الفجوة الغذائية الحقيقية"] = (
                  temp_df["إجمالي الاستهلاك المتاح"] - net_prod
              )
            else:
              temp_df["الفجوة الغذائية الحقيقية"] = (
                  temp_df["إجمالي الاستهلاك المتاح"]
                  - (temp_df[fs_prod] * 0.90)
              )

            temp_df["فترة كفاية الإنتاج (يوم)"] = np.where(
                temp_df["إجمالي الاستهلاك المتاح"] > 0,
                (temp_df[fs_prod] / temp_df["إجمالي الاستهلاك المتاح"])
                * 365.25,
                np.nan,
            )

            if fs_pop and fs_pop != "None":
              temp_df[fs_pop] = pd.to_numeric(
                  temp_df[fs_pop], errors="coerce"
              )
              temp_df["نصيب الفرد من الاستهلاك المتاح"] = np.where(
                  temp_df[fs_pop] > 0,
                  temp_df["إجمالي الاستهلاك المتاح"] / temp_df[fs_pop],
                  np.nan,
              )

            st.dataframe(temp_df, use_container_width=True)

            avg_app_gap = temp_df["الفجوة الغذائية الظاهرية"].mean()
            avg_real_gap = temp_df["الفجوة الغذائية الحقيقية"].mean()
            avg_adequacy = temp_df["فترة كفاية الإنتاج (يوم)"].mean()

            st.success(
                f"• **متوسط الفجوة الظاهرية:** {avg_app_gap:,.2f}\n• **متوسط"
                f" الفجوة الحقيقية:** {avg_real_gap:,.2f}\n• **متوسط فترة كفاية"
                f" الإنتاج المحلي:** {avg_adequacy:.1f} يوم"
            )

            st.markdown(
                "### 📊 تمثيل بياني للفجوات الغذائية وفترة كفاية الإنتاج:"
            )
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(
                range(len(temp_df)),
                temp_df["الفجوة الغذائية الظاهرية"],
                marker="o",
                color="red",
                label="الفجوة الظاهرية",
            )
            ax1.plot(
                range(len(temp_df)),
                temp_df["الفجوة الغذائية الحقيقية"],
                marker="s",
                color="darkorange",
                label="الفجوة الحقيقية",
            )
            ax1.set_xlabel("المشاهدات / السنوات")
            ax1.set_ylabel("الكمية")
            ax1.set_title("مقارنة الفجوة الظاهرية والفجوة الحقيقية")
            ax1.legend()

            ax2.bar(
                range(len(temp_df)),
                temp_df["فترة كفاية الإنتاج (يوم)"],
                color="forestgreen",
            )
            ax2.set_xlabel("المشاهدات / السنوات")
            ax2.set_ylabel("الأيام")
            ax2.set_title("فترة كفاية الإنتاج المحلي (يوم)")
            st.pyplot(fig)

          except Exception as ex:
            st.error(f"حدث خطأ أثناء الاحتساب: {ex}")

      # 10. مؤشرات التجارة الخارجية والتبعية الاقتصادية
      elif model_choice == "10. مؤشرات التجارة الخارجية والتبعية الاقتصادية":
        st.subheader(
            "🌐 تحليل مؤشرات التجارة الخارجية (أهمية الصادرات والواردات للناتج"
            " ومؤشر التبعية)"
        )
        col1, col2, col3 = st.columns(3)
        with col1:
          ft_exp = st.selectbox(
              "اختر عمود إجمالي الصادرات (Exports):", columns_list, key="ft_ex"
          )
        with col2:
          ft_imp = st.selectbox(
              "اختر عمود إجمالي الواردات (Imports):", columns_list, key="ft_im"
          )
        with col3:
          ft_gdp = st.selectbox(
              "اختر عمود الناتج المحلي الإجمالي (GDP):", columns_list, key="ft_g"
          )

        if st.button("🚀 احتساب مؤشرات التجارة الخارجية والتبعية", key="btn_ft"):
          if ft_exp and ft_imp and ft_gdp:
            try:
              temp_df = df.copy()
              temp_df[ft_exp] = pd.to_numeric(temp_df[ft_exp], errors="coerce")
              temp_df[ft_imp] = pd.to_numeric(temp_df[ft_imp], errors="coerce")
              temp_df[ft_gdp] = pd.to_numeric(temp_df[ft_gdp], errors="coerce")
              temp_df = temp_df.dropna(subset=[ft_exp, ft_imp, ft_gdp])

              temp_df["الميزان التجاري"] = temp_df[ft_exp] - temp_df[ft_imp]
              temp_df["درجة أهمية الصادرات للناتج (%)"] = (
                  temp_df[ft_exp] / temp_df[ft_gdp]
              ) * 100
              temp_df["درجة أهمية الواردات للناتج (%)"] = (
                  temp_df[ft_imp] / temp_df[ft_gdp]
              ) * 100
              temp_df["مؤشر التبعية الاقتصادية (إجمالي التجارة/الناتج %)"] = (
                  (temp_df[ft_exp] + temp_df[ft_imp]) / temp_df[ft_gdp]
              ) * 100

              st.dataframe(temp_df, use_container_width=True)

              avg_exp_gdp = temp_df["درجة أهمية الصادرات للناتج (%)"].mean()
              avg_imp_gdp = temp_df["درجة أهمية الواردات للناتج (%)"].mean()
              avg_dep = temp_df[
                  "مؤشر التبعية الاقتصادية (إجمالي التجارة/الناتج %)"
              ].mean()

              st.success(
                  f"• **متوسط درجة أهمية الصادرات للناتج المحلي:**"
                  f" {avg_exp_gdp:.2f}%\n• **متوسط درجة أهمية الواردات للناتج"
                  f" المحلي:** {avg_imp_gdp:.2f}%\n• **متوسط مؤشر التبعية"
                  f" الاقتصادية والتجارية:** {avg_dep:.2f}%"
              )

              st.markdown(
                  "### 📊 تمثيل بياني لأهمية الصادرات والواردات للناتج المحلي:"
              )
              fig, ax = plt.subplots(figsize=(10, 5))
              x = np.arange(len(temp_df))
              width = 0.35
              ax.bar(
                  x - width / 2,
                  temp_df["درجة أهمية الصادرات للناتج (%)"],
                  width,
                  label="أهمية الصادرات للناتج (%)",
                  color="teal",
              )
              ax.bar(
                  x + width / 2,
                  temp_df["درجة أهمية الواردات للناتج (%)"],
                  width,
                  label="أهمية الواردات للناتج (%)",
                  color="coral",
              )
              ax.set_xlabel("المشاهدات / السنوات")
              ax.set_ylabel("النسبة المئوية من الناتج المحلي (%)")
              ax.set_title("درجة أهمية الصادرات والواردات للناتج المحلي")
              ax.legend()
              st.pyplot(fig)

            except Exception as ex:
              st.error(f"حدث خطأ أثناء الحساب: {ex}")
          else:
            st.warning(
                "يرجى اختيار أعمدة الصادرات والواردات والناتج المحلي (GDP)."
            )

      # 11. مؤشرات القدرة التنافسية الشاملة (مع إدراج السعر النسبي صراحة)
      elif (
          model_choice
          == "11. مؤشرات القدرة التنافسية الشاملة (RCA, النصيب السوقي, الاختراق, السعر النسبي)"
      ):
        st.subheader(
            "🏆 تحليل القدرة التنافسية الشاملة (الميزة النسبية الظاهرة، النصيب"
            " السوقي، معامل الاختراق، والسعر النسبي)"
        )

        c1, c2 = st.columns(2)
        with c1:
          comp_x_ij = st.selectbox(
              "صادرات الدولة من السلعة (X_ij):", columns_list, key="c_xij"
          )
          comp_x_it = st.selectbox(
              "إجمالي صادرات الدولة (X_it):", columns_list, key="c_xit"
          )
          comp_imp = st.selectbox(
              "إجمالي الواردات المحلية من السلعة (Imports):",
              columns_list,
              key="c_imp",
          )
        with c2:
          comp_x_wj = st.selectbox(
              "الصادرات العالمية للسلعة (X_wj):", columns_list, key="c_xwj"
          )
          comp_x_wt = st.selectbox(
              "إجمالي الصادرات العالمية (X_wt):", columns_list, key="c_xwt"
          )
          comp_prod = st.selectbox(
              "الإنتاج المحلي للسلعة (Production):", columns_list, key="c_prod"
          )

        st.markdown("---")
        st.markdown("#### 💲 متغيرات الأسعار لحساب السعر النسبي (Relative Price):")
        c3, c4 = st.columns(2)
        with c3:
          comp_local_price = st.selectbox(
              "سعر التصدير المحلي / السعر المحلي (Local Price):",
              columns_list,
              key="c_lpr",
          )
        with c4:
          comp_ref_price = st.selectbox(
              "السعر العالمي أو المرجعي المنافس (Reference/World Price):",
              columns_list,
              key="c_rpr",
          )

        if st.button("🚀 احتساب منظومة القدرة التنافسية المتكاملة", key="btn_comp_all"):
          try:
            temp_df = df.copy()
            for col in [
                comp_x_ij,
                comp_x_it,
                comp_x_wj,
                comp_x_wt,
                comp_imp,
                comp_prod,
                comp_local_price,
                comp_ref_price,
            ]:
              if col:
                temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")

            # 1. الميزة النسبية الظاهرة (RCA)
            temp_df["الميزة النسبية الظاهرة (RCA)"] = np.where(
                (temp_df[comp_x_it] > 0)
                & (temp_df[comp_x_wj] > 0)
                & (temp_df[comp_x_wt] > 0),
                (temp_df[comp_x_ij] / temp_df[comp_x_it])
                / (temp_df[comp_x_wj] / temp_df[comp_x_wt]),
                np.nan,
            )

            # 2. النصيب السوقي (Market Share %)
            temp_df["النصيب السوقي (%)"] = np.where(
                temp_df[comp_x_wj] > 0,
                (temp_df[comp_x_ij] / temp_df[comp_x_wj]) * 100,
                np.nan,
            )

            # 3. معامل الاختراق الاستيرادي (Import Penetration Rate %)
            cons = (
                temp_df[comp_prod] + temp_df[comp_imp] - temp_df[comp_x_ij]
            )
            temp_df["معامل الاختراق الاستيرادي (%)"] = np.where(
                cons > 0, (temp_df[comp_imp] / cons) * 100, np.nan
            )

            # 4. السعر النسبي (Relative Price = Local Price / Reference Price)
            temp_df["السعر النسبي"] = np.where(
                temp_df[comp_ref_price] > 0,
                temp_df[comp_local_price] / temp_df[comp_ref_price],
                np.nan,
            )

            st.dataframe(temp_df, use_container_width=True)

            avg_rca = temp_df["الميزة النسبية الظاهرة (RCA)"].mean()
            avg_rp = temp_df["السعر النسبي"].mean()
            st.success(
                f"• **متوسط مؤشر الميزة النسبية (RCA):** {avg_rca:.4f}\n• **متوسط"
                f" السعر النسبي:** {avg_rp:.4f}"
            )

            st.markdown("### 📊 تمثيل بياني لمؤشرات التنافسية (RCA والسعر النسبي):")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(
                range(len(temp_df)),
                temp_df["الميزة النسبية الظاهرة (RCA)"],
                marker="o",
                color="purple",
                lw=2,
            )
            ax1.axhline(
                1.0, color="red", linestyle="--", label="حد الميزة النسبية (1.0)"
            )
            ax1.set_title("مؤشر الميزة النسبية الظاهرة (RCA)")
            ax1.legend()

            ax2.plot(
                range(len(temp_df)),
                temp_df["السعر النسبي"],
                marker="s",
                color="crimson",
                lw=2,
            )
            ax2.axhline(
                1.0, color="black", linestyle="--", label="التكافؤ السعري (1.0)"
            )
            ax2.set_title("مؤشر السعر النسبي (Relative Price)")
            ax2.legend()
            st.pyplot(fig)

          except Exception as ex:
            st.error(f"حدث خطأ أثناء احتساب مؤشرات التنافسية: {ex}")

    except Exception as e:
      st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
  else:
    st.info("👈 يرجى رفع ملف البيانات الخاص بك من القائمة الجانبية لبدء العمل.")


# =========================================================
# القسم الثاني: بوابة جمع البيانات والمؤشرات
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
# القسم الثالث: المستشار الاقتصادي والقياسي (برنامج الذكاء الاصطناعي التفاعلي)
# =========================================================
elif app_mode == "👨‍🏫 المستشار الاقتصادي والقياسي (برنامج الذكاء الاصطناعي)":
  st.subheader(
      "🤖 المستشار الاقتصادي والقياسي الذكي (شات تفاعلي مدعوم بالذكاء الاصطناعي)"
  )
  st.write(
      "أهلاً بك في غرفة الحوار مع الخبير الأكاديمي الرقمي. اطرح أي سؤال اقتصادي"
      " أو استفسار قياسي وسيقوم النظام بالرد عليك بشكل فوري وتفاعلي متكامل."
  )

  # إعدادات مفتاح API في الشريط الجانبي لتفعيل الذكاء الحي
  st.sidebar.markdown("---")
  st.sidebar.subheader("🔑 إعدادات ذكاء المستشار الآلي")
  ai_provider = st.sidebar.selectbox(
      "اختر محرك الذكاء الاصطناعي:",
      ["Google Gemini (Recommended)", "OpenAI GPT-4o"],
  )
  user_api_key = st.sidebar.text_input(
      "أدخل مفتاح الـ API الخاص بك (Gemini / OpenAI):", type="password"
  )

  active_api_key = user_api_key
  if not active_api_key:
    try:
      if "GEMINI_API_KEY" in st.secrets:
        active_api_key = st.secrets["GEMINI_API_KEY"]
        ai_provider = "Google Gemini (Recommended)"
      elif "OPENAI_API_KEY" in st.secrets:
        active_api_key = st.secrets["OPENAI_API_KEY"]
        ai_provider = "OpenAI GPT-4o"
    except Exception:
      pass

  # تهيئة سجل المحادثة في الذاكرة المؤقتة (Session State)
  if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "أهلاً بك أيها الباحث والزميل العزيز. أنا خبيرك الاقتصادي والقياسي"
            " الذكي، جاهز للإجابة على كافة تساؤلاتك حول الاقتصاد الزراعي،"
            " مؤشرات الأمن الغذائي، الفجوات، دالات الإنتاج، ومؤشرات القدرة"
            " التنافسية (مثل RCA والسعر النسبي). كيف يمكنني مساعدتك اليوم؟"
        ),
    }]

  # عرض رسائل الشات السابقة
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  # استقبال السؤال الجديد عبر صندوق المحادثة الحقيقي (Chat Input)
  if prompt := st.chat_input(
      "اكتب سؤالك أو استفسارك الاقتصادي هنا (مثل: اشرح مؤشر السعر النسبي)..."
  ):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    with st.chat_message("assistant"):
      with st.spinner("جاري التفكير وصياغة التحليل الأكاديمي العميق..."):
        response_text = ""
        system_instruction = (
            "أنت أستاذ أكاديمي مرموق وخبير دولي رفيع المستوى في الاقتصاد"
            " الزراعي، الاقتصاد القياسي، والتجارة الدولية. مهمتك هي تقديم إجابات"
            " وافية، عميقة، ومفصلة للغاية تتطابق مع أرقى المعايير الأكاديمية"
            " العالمية والمقالات البحثية المحكمة. يجب أن تتضمن إجابتك بشكل موسع:"
            " 1. الإطار النظري المفصل للمفهوم أو المؤشر الاقتصادي."
            " 2. الصيغ الرياضية والاقتصادية الدقيقة بدلالة الرموز مع الشرح التفصيلي"
            " لكل رمز ومعامل."
            " 3. التفسير الاقتصادي والقياسي العميق للنتائج وآثارها."
            " 4. الآثار السياساتية والتوصيات الاستراتيجية لصناع القرار."
            " 5. المراجع العلمية والمصادر الأكاديمية المعتمدة (مثل FAO, World"
            " Bank, Balassa, Gujarati)."
        )

        # محاولة الاتصال الفعلي بالذكاء الاصطناعي إذا وُجد المفتاح
        if active_api_key and len(active_api_key) > 5:
          if "Gemini" in ai_provider:
            try:
              import google.generativeai as genai

              genai.configure(api_key=active_api_key)
              model = genai.GenerativeModel(
                  model_name="gemini-1.5-pro",
                  system_instruction=system_instruction,
              )
              chat = model.start_chat(history=[])
              response = chat.send_message(prompt)
              response_text = response.text
            except Exception as e:
              response_text = (
                  f"⚠️ تعذر الاتصال بـ Google Gemini API: {e}\n\nيرجى التحقق من"
                  " صحة المفتاح."
              )
          else:
            try:
              from openai import OpenAI

              client = OpenAI(api_key=active_api_key)
              formatted_msgs = [{"role": "system", "content": system_instruction}]
              for m in st.session_state.messages:
                formatted_msgs.append({"role": m["role"], "content": m["content"]})
              completion = client.chat.completions.create(
                  model="gpt-4o", messages=formatted_msgs
              )
              response_text = completion.choices[0].message.content
            except Exception as e:
              response_text = f"⚠️ تعذر الاتصال بـ OpenAI API: {e}"

        # رد افتراضي احترافي ومتقدم في حال عدم إدخال المفتاح
        if not response_text or "⚠️" in response_text:
          p_lower = prompt.strip().lower()
          if (
              "تنافسية" in p_lower
              or "سعر" in p_lower
              or "rca" in p_lower
              or "ميزة" in p_lower
          ):
            response_text = r"""### 🏛️ الدراسة الأكاديمية الموسعة: تحليل القدرة التنافسية التجارية والسعر النسبي

#### 1. الإطار النظري والمفهوم الاقتصادي:
تُعرّف **القدرة التنافسية (Competitiveness)** في الأدبيات الاقتصادية الدولية بأنها قدرة الدولة على إنتاج وتصدير السلع والخدمات بكفاءة تنافسية في الأسواق العالمية مقارنة بالمنافسين الأجانب، مع الحفاظ على معدلات نمو مستدامة ومستويات معيشية مرتفعة.

#### 2. المكونات الأساسية ومؤشر السعر النسبي:
* **مؤشر السعر النسبي (Relative Price - RP):**
  - **مفهومه:** يُعد من أهم مؤشرات التنافسية السعرية، حيث يعكس مدى قدرة المنتج المحلي على المنافسة في الأسواق مقارنة بالأسعار العالمية للمنافسين.
  - **الصيغة الرياضية:**
    $$RP = \frac{P_{local}}{P_{reference}}$$
    حيث ($P_{local}$) هو سعر التصدير المحلي، و($P_{reference}$) هو السعر العالمي أو المرجعي للمنافس.
  - **التفسير:** إذا كان $RP < 1$ فهذا يعكس **ميزة سعرية تنافسية** للمنتج المحلي، أما إذا كان $RP > 1$ فيعكس ارتفاع السعر المحلي مقارنة بالعالمي مما يضعف التنافسية.

* **مؤشر الميزة النسبية الظاهرة (RCA):**
  $$RCA_{ij} = \frac{X_{ij} / X_{it}}{X_{wj} / X_{wt}}$$
  إذا كان الناتج أكبر من الواحد ($RCA > 1$) فهذا يدل على وجود ميزة نسبية تصديرية قوية.

---
📚 **المراجع والمصادر الأكاديمية المعتمدة:**
1. Balassa, B. (1965). *Trade Liberalisation and “Revealed” Comparative Advantage*. The Manchester School.
2. Porter, M. E. (1990). *The Competitive Advantage of Nations*.
3. FAO (2022). *Agricultural Trade Competitiveness Methodologies*."""
          elif (
              "أمن" in p_lower
              or "غذائي" in p_lower
              or "اكتفاء" in p_lower
              or "فجوة" in p_lower
          ):
            response_text = r"""### 🌾 الدراسة الأكاديمية الموسعة: منظومة الأمن الغذائي والفجوات

#### 1. الإطار النظري والمفاهيمي:
يُعد الأمن الغذائي ركيزة أساسية للأمن القومي. ويتم قياسه عبر مؤشرات كمية دقيقة:
- **نسبة الاكتفاء الذاتي (SSR):** $(الإنتاج / الاستهلاك المتاح) \times 100$.
- **الفجوة الغذائية (الظاهرية والحقيقية):**
  - الظاهرية: الفرق الإجمالي بين الاستهلاك والإنتاج.
  - الحقيقية: تُحسب بعد استبعاد الفاقد والتلف المتراكم في سلاسل الإمداد.
- **فترة كفاية الإنتاج المحلي (يوم):** 
  $$\text{فترة الكفاية} = \left( \frac{\text{الإنتاج المحلي}}{\text{إجمالي الاستهلاك المتاح}} \right) \times 365.25$$

---
📚 **المراجع:** FAO (2021) & World Bank (2020)."""
          else:
            response_text = f"""💡 **تحليل الخبير الأكاديمي حول استفسارك:**

بناءً على النظريات المتقدمة في الاقتصاد الزراعي والقياسي، يتطلب التعامل مع هذا الموضوع دمج النظريات الاقتصادية الجزئية والكلية، وتحليل سلاسل الإمداد، وتقدير معالم النماذج القياسية بدقة.

*(ملاحظة: لتشغيل الذكاء الاصطناعي التفاعلي الحي بالكامل والرد على أي سؤال مفتوح، يرجى إدخال مفتاح Google Gemini API أو OpenAI API في الشريط الجانبي).*

---
📚 **المراجع المعتمدة:** Gujarati & Porter (2009) - Basic Econometrics."""

        st.markdown(response_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )
