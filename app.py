import numpy as np
import pandas as pd
import requests
from scipy.optimize import linprog
import statsmodels.api as sm
import streamlit as st

# إعدادات صفحة التطبيق
st.set_page_config(
    page_title="منصة تحليل الاقتصاد الزراعي والاقتصاد القياسي",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# نظام الحماية بكلمة المرور
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
    st.write("الرجاء إدخال كود المرور أو الاشتراك للوصول إلى المنصة:")
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
st.title("🌾 منصة تحليل الاقتصاد الزراعي والاقتصاد القياسي الشاملة")
st.write(
    "منصة بحثية متقدمة للتحليلات القياسية، الإحصاء الوصفى، الاستشارات"
    " الذكية، وجلب بيانات البنك الدولي المفتوحة."
)

# الإعدادات العامة في القائمة الجانبية
st.sidebar.header("⚙️ إعدادات المنصة")
app_mode = st.sidebar.radio(
    "اختر قسم العمل الأساسي:",
    ["📊 تحليل البيانات والنماذج القياسية", "🤖 المستشار الاقتصادي الذكي (AI)", "🌐 بوابة بيانات البنك الدولي والفاو"]
)

# إعدادات مفتاح الذكاء الاصطناعي في الجانب
st.sidebar.markdown("---")
st.sidebar.header("🔑 إعدادات الذكاء الاصطناعي")
ai_api_key = st.sidebar.text_input(
    "مفتاح API (Gemini/OpenAI):",
    type="password",
    help="مطلوب فقط لاستخدام قسم المستشار الاقتصادي الذكي.",
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
            # حساب الإحصاءات الأساسية
            stats_df = sub_df.describe().T
            # إضافة التباين ومعامل الاختلاف
            stats_df["التباين"] = sub_df.var()
            stats_df["معامل الاختلاف (%)"] = (
                sub_df.std() / sub_df.mean()
            ) * 100
            stats_df["المنوال"] = sub_df.mode().iloc[0]

            # إعادة تسمية الأعمدة للعربية
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

            # إعادة ترتيب الأعمدة لشكل احترافي
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

            st.markdown("---")
            st.markdown("### 📝 التقرير التحليلي الإحصائي")
            st.success(
                "• **النزعة المركزية:** يوضح المتوسط الحسابي والوسيط والمنوال مركز"
                " تجميع البيانات لكل متغير زراعي أو اقتصادي مختار."
            )
            st.info(
                "• **التشتت:** يعكس الانحراف المعياري والتباين مدى انتشار"
                " البيانات حول قيمتها المتوسطة، بينما يوضح معامل الاختلاف درجة"
                " التشتت النسبي للمقارنة بين المتغيرات."
            )
          except Exception as ex:
            st.error(f"حدث خطأ أثناء حساب الإحصاءات الوصفية: {ex}")
        else:
          st.warning("يرجى اختيار متغير واحد على الأقل لحساب المقاييس.")

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

              if len(temp_df) < 3:
                st.error("البيانات الصالحة غير كافية لإجراء التحليل.")
              else:
                y = temp_df[y_col]
                X = sm.add_constant(temp_df[x_cols])
                model = sm.OLS(y, X).fit()
                st.text(model.summary().as_text())

                st.markdown("---")
                st.markdown("### 📝 التقرير التحليلي باللغة العربية")
                r2 = model.rsquared * 100
                st.success(
                    f"• **معامل التحديد ($R^2$):** بلغ {r2:.2f}%، مما يشير إلى أن"
                    " المتغيرات المستقلة تفسر هذه النسبة من التغيرات في"
                    f" المتغير التابع ({y_col})."
                )
            except Exception as ex:
              st.error(f"حدث خطأ أثناء تنفيذ نموذج OLS: {ex}")
          else:
            st.warning("يرجى اختيار المتغير التابع والمتغيرات المستقلة.")

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

              if len(temp_df) < 3:
                st.error("البيانات الصالحة غير كافية.")
              else:
                df_log = np.log(temp_df)
                X = sm.add_constant(df_log[x_cols])
                y = df_log[y_col]
                model = sm.OLS(y, X).fit()
                st.text(model.summary().as_text())

                returns_to_scale = model.params[x_cols].sum()
                st.markdown("### 📊 ملخص المرونات وعوائد الحجم:")
                cols = st.columns(len(x_cols) + 1)
                for i, col_name in enumerate(x_cols):
                  cols[i].metric(
                      f"مرونة ({col_name})", f"{model.params[col_name]:.4f}"
                  )
                cols[-1].metric("إجمالي عوائد الحجم", f"{returns_to_scale:.4f}")
            except Exception as ex:
              st.error(f"حدث خطأ أثناء التنفيذ: {ex}")
          else:
            st.warning("يرجى اختيار المتغيرات المطلوبة.")

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
            except Exception as ex:
              st.error(f"حدث خطأ: {ex}")
          else:
            st.warning("يرجى اختيار المتغيرات.")

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
              st.line_chart(
                  pd.DataFrame(
                      {
                          "القيم الفعلية": temp_df[target_var],
                          "خط الاتجاه العام": trend_model.fittedvalues,
                      },
                      index=temp_df[year_col],
                  )
              )
            except Exception as ex:
              st.error(f"حدث خطأ: {ex}")
          else:
            st.warning("يرجى اختيار الأعمدة المطلوبة.")

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
                if res.success:
                  eff_list.append(res.x[0])
                else:
                  eff_list.append(np.nan)

              temp_df["Technical_Efficiency (DEA)"] = eff_list
              st.dataframe(temp_df, use_container_width=True)
            except Exception as ex:
              st.error(f"حدث خطأ: {ex}")
          else:
            st.warning("يرجى اختيار المتغيرات.")

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
            except Exception as ex:
              st.error(f"حدث خطأ: {ex}")
          else:
            st.warning("يرجى اختيار أعمدة الأسعار.")

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
              max_resid = residuals.max()
              te = np.exp(residuals - max_resid)
              temp_df["Technical_Efficiency"] = te
              st.text(model.summary().as_text())
              st.dataframe(temp_df, use_container_width=True)
            except Exception as ex:
              st.error(f"حدث خطأ: {ex}")
          else:
            st.warning("يرجى اختيار المتغيرات.")

    except Exception as e:
      st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
  else:
    st.info("👈 يرجى رفع ملف البيانات الخاص بك من القائمة الجانبية لبدء العمل.")


# =========================================================
# القسم الثاني: المستشار الاقتصادي الذكي (AI)
# =========================================================
elif app_mode == "🤖 المستشار الاقتصادي الذكي (AI)":
  st.subheader("🤖 المستشار الاقتصادي والقياسي (مدعوم بالذكاء الاصطناعي)")
  st.write(
      "اطرح أي سؤال اقتصادي، اطلب شرحاً تفصيلياً لنظرية اقتصادية (مثل قانون"
      " تناقص الغلة، مروناة الطلب، توازن السوق)، أو استفسر عن أي مفهوم قياسي وسيجيبك"
      " الخبير بأسلوب بشري مفسر ودقيق."
  )

  user_question = st.text_area(
      "اكتب سؤالك أو استفسارك الاقتصادي هنا:",
      placeholder="مثلاً: اشرح لي بالتفصيل نظرية دالة إنتاج كوب-دوجلاس وأهميتها في الاقتصاد الزراعي...",
      height=120,
  )

  if st.button("💬 إرسال السؤال للحصول على الشرح المفصل"):
    if not user_question:
      st.warning("يرجى كتابة السؤال أولاً.")
    elif not ai_api_key:
      st.warning(
          "⚠️ يرجى إدخال مفتاح API الخاص بك في خانة الإعدادات بالقائمة الجانبية"
          " أولاً."
      )
    else:
      with st.spinner("جاري صياغة الإجابة العلمية المفصلة..."):
        try:
          import google.generativeai as genai

          genai.configure(api_key=ai_api_key)
          # استخدام نموذج جيميناي
          model = genai.GenerativeModel("gemini-1.5-flash")
          prompt = f"أنت أستاذ وخبير محترف في الاقتصاد والاقتصاد القياسي الزراعي. أجب عن السؤال التالي بأسلوب علمي دقيق، مفسر، ومكتوب بطريقة بشرية منظمة وبسيطة:\n\n{user_question}"
          response = model.generate_content(prompt)

          st.markdown("---")
          st.markdown("### 💡 إجابة المستشار الاقتصادي:")
          st.markdown(response.text)
        except Exception as e:
          st.error(
              f"حدث خطأ أثناء الاتصال بمحرك الذكاء الاصطناعي: {e}. تأكد من صحة"
              " مفتاح الـ API."
          )


# =========================================================
# القسم الثالث: بوابة بيانات البنك الدولي والفاو المفتوحة
# =========================================================
elif app_mode == "🌐 بوابة بيانات البنك الدولي والفاو":
  st.subheader("🌐 بوابة البيانات المفتوحة (البنك الدولي & الفاو)")
  st.write(
      "هنا يمكنك جلب بيانات أي مؤشر اقتصادي أو زراعي عالمي مباشرة من قواعد"
      " بيانات البنك الدولي المفتوحة، وعرض الجدول، وتوثيق المصدر، وتحميل البيانات"
      " كملف إكسيل/CSV."
  )

  col_b1, col_b2 = st.columns(2)
  with col_b1:
    country_code = st.text_input(
        "كود الدولة الثلاثي (مثال: EGY لمصر، USA لأمريكا، WLD للعالم):",
        value="EGY",
    )
  with col_b2:
    indicator_choice = st.selectbox(
        "اختر المؤشر الاقتصادي والزراعي الجاهز:",
        [
            (
                "الزراعة والغابات والصيد كنسبة من الناتج المحلي الإجمالي"
                " (NV.AGR.TOTL.ZS)"
            ),
            ("إجمالي إنتاج الحبوب كجم لكل هكتار (AG.YLD.CREL.KG)"),
            ("الأراضي الزراعية كنسبة من المساحة الكلية (AG.LND.AGRI.ZS)"),
            ("السكان الريفيون كنسبة من إجمالي السكان (SP.RUR.TOTL.ZS)"),
            ("مؤشر آخر (أدخل الكود يدوياً)"),
        ],
    )

  if indicator_choice.startswith("مؤشر آخر"):
    indicator_code = st.text_input("أدخل كود مؤشر البنك الدولي يدوياً:")
  else:
    indicator_code = indicator_choice.split("(")[-1].replace(")", "")

  if st.button("📥 جلب البيانات وتحميلها كشيت"):
    if not country_code or not indicator_code:
      st.warning("يرجى التأكد من إدخال كود الدولة وكود المؤشر.")
    else:
      with st.spinner("جاري الاتصال بقاعدة بيانات البنك الدولي وسحب البيانات..."):
        try:
          url = f"http://api.worldbank.org/v2/country/{country_code.strip()}/indicator/{indicator_code.strip()}?format=json&per_page=100"
          response = requests.get(url)

          if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and data[1]:
              records = []
              for entry in data[1]:
                yr = entry.get("date")
                val = entry.get("value")
                c_name = entry.get("country", {}).get("value", country_code)
                ind_name = entry.get("indicator", {}).get(
                    "value", indicator_code
                )
                records.append(
                    {
                        "الدولة": c_name,
                        "السنة": yr,
                        "اسم المؤشر": ind_name,
                        "القيمة": val,
                    }
                )

              df_wb = pd.DataFrame(records)
              st.success(
                  "🎉 تم جلب البيانات بنجاح من قاعدة بيانات البنك الدولي الرسمية!"
              )

              # عرض المصدر بوضوح
              st.markdown("### 📚 توثيق المصدر والبيانات:")
              st.info(
                  f"• **المصدر الرسمي:** البنك الدولي (World Bank Open Data"
                  f" API)\n• **كود المؤشر:** `{indicator_code}`\n• **الدولة المستهدفة:**"
                  f" `{country_code.upper()}`"
              )

              st.dataframe(df_wb, use_container_width=True)

              # زر التحميل كملف CSV (يفتح مباشرة في إكسيل)
              csv_data = df_wb.to_csv(index=False).encode("utf-8-sig")
              st.download_button(
                  label="💾 تحميل البيانات كملف CSV (جاهز للإكسيل)",
                  data=csv_data,
                  file_name=f"WorldBank_Data_{country_code}_{indicator_code}.csv",
                  mime="text/csv",
              )
            else:
              st.error(
                  "❌ لم يتم العثور على بيانات لهذا المؤشر أو أن كود الدولة غير"
                  " صحيح."
              )
          else:
            st.error(
                f"فشل الاتصال بالخادم. رمز الخطأ: {response.status_code}"
            )
        except Exception as ex:
          st.error(f"حدث خطأ أثناء جلب البيانات: {ex}")
