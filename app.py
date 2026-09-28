import numpy as np
import pandas as pd
from scipy.optimize import linprog
import statsmodels.api as sm
import streamlit as st

# إعدادات صفحة التطبيق
st.set_page_config(
    page_title="منصة تحليل الاقتصاد الزراعي والاقتصاد القياسي",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌾 منصة تحليل الاقتصاد الزراعي والاقتصاد القياسي الشاملة")
st.write(
    "منصة بحثية متقدمة تتضمن النماذج الاقتصادية والقياسية مع تقارير تحليلية"
    " تلقائية."
)

# ---------------------------------------------------------
# الجزء الخاص برفع ملف البيانات
# ---------------------------------------------------------
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

    # عرض جدول البيانات النشط
    st.subheader("📊 جدول البيانات النشط للتحليل:")
    st.dataframe(df, use_container_width=True)

    # ---------------------------------------------------------
    # لوحة التحكم واختيار النماذج
    # ---------------------------------------------------------
    st.sidebar.header("🎛️ لوحة التحكم والاختيار")
    model_choice = st.sidebar.selectbox(
        "اختر نموذج التحليل الاقتصادي والقياسي:",
        [
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

    if st.button("🚀 تشغيل التحليل وإصدار التقرير"):

      # 1. دالة الإنتاج الخطية (OLS)
      if model_choice == "1. دالة الإنتاج الخطية (OLS)":
        st.subheader("📈 نتائج دالة الإنتاج الخطية (OLS)")
        if {"Water", "Fertilizer", "Yield"}.issubset(df.columns):
          X = sm.add_constant(df[["Water", "Fertilizer"]])
          y = df["Yield"]
          model = sm.OLS(y, X).fit()
          st.text(model.summary().as_text())

          # التقرير التحليلي العربي
          st.markdown("---")
          st.markdown("### 📝 التقرير التحليلي باللغة العربية")
          r2 = model.rsquared * 100
          st.success(
              f"• **معامل التحديد ($R^2$):** بلغ {r2:.2f}%، مما يشير إلى أن"
              " المتغيرات المستقلة (المياه والسماد) تفسر هذه النسبة من التغيرات"
              " في الإنتاجية."
          )
          st.info(
              "• **التفسير الاقتصادي:** توضح المعاملات التغير المطلق في الإنتاج"
              " الناتج عن زيادة وحدة واحدة من كل مدخل مع ثبات العوامل الأخرى."
          )
        else:
          st.error(
              "يرجى التأكد من وجود أعمدة (Yield, Water, Fertilizer) في الملف"
              " المرفوع."
          )

      # 2. دالة كوب دوجلاس
      elif model_choice == "2. دالة إنتاج كوب-دوجلاس (Cobb-Douglas)":
        st.subheader("📉 نتائج دالة كوب-دوجلاس اللوغاريتمية (Log-Log Model)")
        if {"Water", "Fertilizer", "Yield"}.issubset(df.columns):
          df_log = np.log(df[["Yield", "Water", "Fertilizer"]])
          X = sm.add_constant(df_log[["Water", "Fertilizer"]])
          y = df_log["Yield"]
          model = sm.OLS(y, X).fit()
          st.text(model.summary().as_text())

          beta_water = model.params["Water"]
          beta_fert = model.params["Fertilizer"]
          returns_to_scale = beta_water + beta_fert

          col1, col2, col3 = st.columns(3)
          col1.metric("مرونة المياه", f"{beta_water:.4f}")
          col2.metric("مرونة السماد", f"{beta_fert:.4f}")
          col3.metric("إجمالي عوائد الحجم", f"{returns_to_scale:.4f}")

          # التقرير التحليلي العربي
          st.markdown("---")
          st.markdown("### 📝 التقرير التحليلي باللغة العربية")
          scale_desc = (
              "متزايدة (Increasing)"
              if returns_to_scale > 1
              else (
                  "ثابتة (Constant)"
                  if abs(returns_to_scale - 1) < 0.001
                  else "متناقصة (Decreasing)"
              )
          )
          st.success(
              f"• **مرونة الإنتاج:** مرونة المياه تقدر بـ ({beta_water:.4f})"
              f" ومرونة السماد بـ ({beta_fert:.4f}). تعني المرونة نسبة تغير"
              " الإنتاج عند تغير المدخل بنسبة 1%."
          )
          st.info(
              f"• **عوائد الحجم:** إجمالي عوائد الحجم بلغ ({returns_to_scale:.4f})"
              f" وهو ما يدل على أن الإنتاج يمر بحالة **{scale_desc}**."
          )
        else:
          st.error(
              "يرجى التأكد من وجود أعمدة (Yield, Water, Fertilizer) في الملف"
              " المرفوع."
          )

      # 3. دالة الإنتاج التربيعية
      elif model_choice == "3. دالة الإنتاج التربيعية (Quadratic)":
        st.subheader("📐 نتائج دالة الإنتاج التربيعية (لقياس تناقص الغلة)")
        if {"Water", "Yield"}.issubset(df.columns):
          df_quad = df.copy()
          df_quad["Water_sq"] = df_quad["Water"] ** 2
          X = sm.add_constant(df_quad[["Water", "Water_sq"]])
          y = df_quad["Yield"]
          model = sm.OLS(y, X).fit()
          st.text(model.summary().as_text())

          # التقرير التحليلي العربي
          st.markdown("---")
          st.markdown("### 📝 التقرير التحليلي باللغة العربية")
          st.success(
              "• **قانون تناقص الغلة:** إشارة المعامل التربيعي السالب للمدخل تدل"
              " على صحة قانون تناقص الغلة، حيث يبدأ العائد الحدي في التناقص"
              " مع التوسع المستمر في استخدام المدخل."
          )
        else:
          st.error(
              "يرجى التأكد من وجود أعمدة (Yield, Water) في الملف المرفوع."
          )

      # 4. تحليل الاتجاه العام
      elif model_choice == "4. تحليل الاتجاه العام (Trend Analysis)":
        st.subheader("📅 تحليل الاتجاه العام للمتغيرات عبر الزمن")
        if "Year" in df.columns:
          numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
          numeric_cols.remove("Year")

          target_var = st.selectbox(
              "اختر المتغير المراد دراسة اتجاهه العام:", numeric_cols
          )

          X_trend = sm.add_constant(df["Year"])
          y_trend = df[target_var]
          trend_model = sm.OLS(y_trend, X_trend).fit()
          st.text(trend_model.summary().as_text())

          st.write(
              f"**رسم بياني يوضح مسار واتجاه ({target_var}) عبر السنوات:**"
          )
          chart_data = pd.DataFrame(
              {
                  "القيم الفعلية": df[target_var],
                  "خط الاتجاه العام": trend_model.fittedvalues,
              },
              index=df["Year"],
          )
          st.line_chart(chart_data)

          # التقرير التحليلي العربي
          st.markdown("---")
          st.markdown("### 📝 التقرير التحليلي باللغة العربية")
          slope = trend_model.params["Year"]
          direction = (
              "تصاعدي (موجب)" if slope > 0 else "تنازلي (سالب)"
          )
          st.success(
              f"• **معدل التغير السنوي:** الميل الزمني للمتغير ({target_var})"
              f" بلغ ({slope:.4f}) سنوياً، وهو اتجاه عام **{direction}** خلال"
              " فترة الدراسة."
          )
        else:
          st.error(
              "الرجاء التأكد من وجود عمود باسم (Year) في الملف لتمكين تحليل"
              " الاتجاه العام."
          )

      # 5. تحليل الكفاءة DEA
      elif model_choice == (
          "5. تحليل الكفاءة باستخدام مغلف البيانات (DEA - Data Envelopment"
          " Analysis)"
      ):
        st.subheader(
            "📐 تحليل الكفاءة الفنية باستخدام مغلف البيانات (DEA - Input-Oriented"
            " CCR)"
        )
        if {"Water", "Fertilizer", "Yield"}.issubset(df.columns):
          inputs = df[["Water", "Fertilizer"]].values
          outputs = df["Yield"].values
          n_dmu = len(df)
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

          df_dea = df.copy()
          df_dea["Technical_Efficiency (DEA)"] = eff_list
          st.dataframe(df_dea, use_container_width=True)

          # التقرير التحليلي العربي
          st.markdown("---")
          st.markdown("### 📝 التقرير التحليلي باللغة العربية")
          avg_eff = np.nanmean(eff_list) * 100
          st.success(
              f"• **متوسط الكفاءة الفنية:** بلغ متوسط الكفاءة الفنية للعينة"
              f" ({avg_eff:.2f}%). الوحدات التي تصل كفاءتها إلى 1.0 (أو 100%)"
              " تعتبر كفؤة وتقع على حد الإنتاج الأمثل."
          )
        else:
          st.error(
              "يرجى التأكد من توفر أعمدة المدخلات والمخرجات المطلوبة (Yield,"
              " Water, Fertilizer)."
          )

      # 6. الهوامش التسويقية
      elif model_choice == "6. حساب الهوامش التسويقية (Marketing Margins)":
        st.subheader("💰 تحليل الهوامش التسويقية ونصيب المزارع")
        if "Farm_Price" in df.columns and "Retail_Price" in df.columns:
          df_margin = df.copy()
          df_margin["Absolute_Margin"] = (
              df_margin["Retail_Price"] - df_margin["Farm_Price"]
          )
          df_margin["Percentage_Margin (%)"] = (
              df_margin["Absolute_Margin"] / df_margin["Retail_Price"]
          ) * 100
          df_margin["Farmer_Share (%)"] = (
              df_margin["Farm_Price"] / df_margin["Retail_Price"]
          ) * 100

          st.dataframe(df_margin, use_container_width=True)

          avg_abs = df_margin["Absolute_Margin"].mean()
          avg_pct = df_margin["Percentage_Margin (%)"].mean()
          avg_share = df_margin["Farmer_Share (%)"].mean()

          col1, col2, col3 = st.columns(3)
          col1.metric("متوسط الهامش المطلق", f"{avg_abs:.2f}")
          col2.metric("متوسط الهامش النسبي", f"{avg_pct:.2f}%")
          col3.metric("متوسط نصيب المزارع", f"{avg_share:.2f}%")

          # التقرير التحليلي العربي
          st.markdown("---")
          st.markdown("### 📝 التقرير التحليلي باللغة العربية")
          st.success(
              f"• **تحليل الكفاءة التسويقية:** بلغ متوسط الهامش المطلق"
              f" ({avg_abs:.2f}) ومتوسط الهامش النسبي ({avg_pct:.2f}%)."
          )
          st.info(
              f"• **نصيب المزارع:** يحصل المزارع في المتوسط على"
              f" ({avg_share:.2f}%) من السعر النهائي المدفوع من المستهلك،"
              " وكلما زادت هذه النسبة دل ذلك على كفاءة وكبر نصيب المنتج من"
              " الأرباح التسويقية."
          )
        else:
          st.error(
              "يرجى التأكد من وجود أعمدة الأسعار (Farm_Price) و (Retail_Price)"
              " في الملف المرفوع."
          )

      # 7. تحليل حد الإنتاج القياسي (Frontier)
      elif model_choice == (
          "7. تحليل حد الإنتاج القياسي (Frontier Analysis - COLS)"
      ):
        st.subheader(
            "⚡ تحليل حد الإنتاج وتقدير الكفاءة الفنية (Corrected OLS Frontier)"
        )
        if {"Water", "Fertilizer", "Yield"}.issubset(df.columns):
          df_log = np.log(df[["Yield", "Water", "Fertilizer"]])
          X = sm.add_constant(df_log[["Water", "Fertilizer"]])
          y = df_log["Yield"]
          model = sm.OLS(y, X).fit()

          residuals = model.resid
          max_resid = residuals.max()
          te = np.exp(residuals - max_resid)

          df_frontier = df.copy()
          df_frontier["Predicted_Yield_Frontier"] = np.exp(
              model.fittedvalues + max_resid
          )
          df_frontier["Technical_Efficiency"] = te

          st.text(model.summary().as_text())
          st.markdown("### 📋 جدول الكفاءة الفنية وحد الإنتاج:")
          st.dataframe(df_frontier, use_container_width=True)

          avg_te = te.mean()
          st.metric(
              "متوسط الكفاءة الفنية على حد الإنتاج", f"{avg_te * 100:.2f}%"
          )

          # التقرير التحليلي العربي
          st.markdown("---")
          st.markdown("### 📝 التقرير التحليلي باللغة العربية")
          st.success(
              f"• **كفاءة حدود الإنتاج:** بلغ متوسط الكفاءة الفنية وفقاً لنموذج"
              f" حدود الإنتاج المصحح (COLS) نحو ({avg_te * 100:.2f}%)."
          )
          st.info(
              "• **الاستنتاج:** تعكس درجات الكفاءة مدى قدرة الوحدات الإنتاجية"
              " على تعظيم الإنتاج باستخدام نفس القدر المتاح من المدخلات"
              " مقارنة بالوحدة المعيارية المثلى على الحدود."
          )
        else:
          st.error(
              "يرجى التأكد من توفر أعمدة (Yield, Water, Fertilizer) في الملف."
          )

  except Exception as e:
    st.error(f"حدث خطأ أثناء قراءة أو معالجة الملف: {e}")
else:
  st.info(
      "👈 يرجى رفع ملف البيانات الخاص بك (Excel أو CSV) من القائمة الجانبية لبدء"
      " عرض الجدول وتشغيل النماذج والتقارير."
  ) 
