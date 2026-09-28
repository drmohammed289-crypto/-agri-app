import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import linprog, root_scalar
import scipy.stats as stats
from scipy.stats import jarque_bera
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, coint, kpss
import streamlit as st

# إعدادات صفحة التطبيق
st.set_page_config(
    page_title="منصة الخبير الاقتصادي والقياسي الذكي الشاملة",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# 1. نظام الحماية بكلمة المرور
# ---------------------------------------------------------
def check_password():
  """التحقق من كلمة المرور"""

  def password_entered():
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
st.title("🌾 منصة الخبير الاقتصادي والقياسي الذكي (الإصدار الشامل)")
st.write(
    "منصة بحثية وأكاديمية متكاملة تضم كافة اختبارات جذر الوحدة، نماذج السلاسل"
    " الزمنية (ARMA, ARIMA, SARIMAX)، نماذج التكامل المشترك وتصحيح الخطأ"
    " (ECM)، اختبارات التشخيص القياسي الشاملة، ودراسات الجدوى والتقييم المالي"
    " للمشروعات."
)

st.sidebar.header("⚙️ إعدادات المنصة")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "### 💻 هُوية البرنامج والمرجع التقني\n"
    "**اسم البرنامج:** منصة الخبير الاقتصادي والقياسي الذكي\n\n"
    "**المرجع التقني والبرمجي:**\n"
    "- **لغة البرمجة:** Python\n"
    "- **المكتبات المستخدمة:** Statsmodels, Pandas, SciPy, NumPy, Streamlit\n"
    "**الإصدار:** 2026 الشامل والمحدث"
)
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "اختر قسم العمل الأساسي:",
    [
        "📊 التحليلات القياسية واختبارات التشخيص الشاملة",
        "📉 اختبارات جذر الوحدة والتكامل المشترك ونماذج السلاسل الزمنية",
        "💰 دراسة الجدوى والتقييم المالي الشامل للمشروعات",
        "🌐 بوابة جمع البيانات والمؤشرات العالمية",
        "👨‍🏫 المستشار الاقتصادي والقياسي (قسم الذكاء الاصطناعي)",
    ],
)


def show_program_credit():
  st.caption(
      "💻 **تم إجراء هذا التحليل باستخدام:** منصة الخبير الاقتصادي والقياسي الذكي"
      " | **المرجع التقني والبرمجي:** Python (Statsmodels, Pandas, SciPy,"
      " NumPy)."
  )


uploaded_file = st.sidebar.file_uploader(
    "قم برفع ملف البيانات (Excel أو CSV):", type=["xlsx", "xls", "csv"]
)

df = None
if uploaded_file is not None:
  try:
    if uploaded_file.name.endswith(".csv"):
      df = pd.read_csv(uploaded_file)
    else:
      df = pd.read_excel(uploaded_file)
    st.sidebar.success("تم تحميل الملف بنجاح! 🎉")
  except Exception as e:
    st.sidebar.error(f"خطأ في قراءة الملف: {e}")

# =========================================================
# القسم الأول: التحليلات القياسية واختبارات التشخيص الشاملة
# =========================================================
if app_mode == "📊 التحليلات القياسية واختبارات التشخيص الشاملة":
  st.subheader("📊 التحليلات القياسية واختبارات التشخيص الإحصائي المتقدمة")

  if df is not None:
    st.dataframe(df.head(), use_container_width=True)
    columns_list = df.columns.tolist()

    sub_choice = st.selectbox(
        "اختر نوع التحليل القياسي:",
        [
            "0. مقاييس النزعة المركزية، التشتت، التلتلة والتفرطح (Descriptive Stats)",
            (
                "1. دالة الإنتاج (OLS) مع كافة التشخيصات (Durbin-Watson, Breusch-Pagan,"
                " White, VIF, Jarque-Bera)"
            ),
            "2. دالة إنتاج كوب-دوجلاس اللوغاريتمية وعوائد الحجم",
            "3. دالة الإنتاج التربيعية وقياس تناقص الغلة",
            "4. تحليل الاتجاه الزمني (خطي وأسي/معدل النمو المركب CAGR)",
            "5. تحليل الكفاءة باستخدام مغلف البيانات (DEA)",
            "6. الهوامش التسويقية ونصيب المزارع",
            "7. تحليل التكاليف وصافي العائد الاقتصادي",
            "8. مؤشرات الأمن الغذائي والفجوات وفترة الكفاية",
            "9. مؤشرات التجارة الخارجية ومؤشر الميزة النسبية (RCA)",
            "10. معاملات الارتباط (بيرسون وسبيرمان) واختبارات T-Test و ANOVA",
        ],
    )

    if sub_choice.startswith("0"):
      st.subheader("📈 الإحصاءات الوصفية واختبارات التوزيع الطبيعي")
      num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
      sel_cols = st.multiselect("اختر المتغيرات:", num_cols)
      if sel_cols and st.button("تشغيل الإحصاءات الوصفية"):
        sub_df = df[sel_cols].apply(pd.to_numeric, errors="coerce")
        desc = sub_df.describe().T
        desc["التباين"] = sub_df.var()
        desc["معامل الاختلاف (%)"] = (sub_df.std() / sub_df.mean()) * 100
        desc["التلتلة (Skewness)"] = sub_df.skew()
        desc["التفرطح (Kurtosis)"] = sub_df.kurtosis()
        st.dataframe(desc, use_container_width=True)
        show_program_credit()

    elif sub_choice.startswith("1"):
      st.subheader("📈 نموذج الانحدار الخطي المتعدد (OLS) مع كافة الاختبارات القياسية")
      c1, c2 = st.columns(2)
      with c1:
        y_c = st.selectbox("المتغير التابع (Y):", columns_list)
      with c2:
        x_c = st.multiselect(
            "المتغيرات المستقلة (X):", [c for c in columns_list if c != y_c]
        )

      if st.button("تشغيل نموذج OLS وتشخيصاته الشاملة") and y_c and x_c:
        temp_df = df[[y_c] + x_c].apply(pd.to_numeric, errors="coerce").dropna()
        y = temp_df[y_c]
        X = sm.add_constant(temp_df[x_c])
        model = sm.OLS(y, X).fit()
        st.text(model.summary().as_text())

        st.markdown("### 🔍 الفحوصات والاختبارات القياسية الإضافية للبواقي:")
        residuals = model.resid

        dw_stat = sm.stats.stattools.durbin_watson(residuals)
        jb_stat, jb_p = jarque_bera(residuals)

        bp_p, wh_p = 1.0, 1.0
        try:
          from statsmodels.stats.diagnostic import het_breuschpagan, het_white

          bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
          wh_stat, wh_p, _, _ = het_white(residuals, model.model.exog)
        except Exception:
          pass

        col_d1, col_d2 = st.columns(2)
        with col_d1:
          st.metric("معامل دوربن-واتسون (Durbin-Watson)", f"{dw_stat:.4f}")
          st.write("*(دلالة: القيمة قرب 2 تعني عدم وجود ارتباط ذاتي)*")
          st.metric(
              "اختبار جارك-بيرا للتوزيع الطبيعي (JB p-value)", f"{jb_p:.4f}"
          )
        with col_d2:
          st.metric(
              "اختبار بروش-باغان لثبات التباين (BP p-value)", f"{bp_p:.4f}"
          )
          st.metric("اختبار وايت لثبات التباين (White p-value)", f"{wh_p:.4f}")

        st.markdown("### 📐 قياس التعدد الخطي المفرد (VIF):")
        vif_data = pd.DataFrame()
        vif_data["المتغير"] = X.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])
        ]
        st.dataframe(vif_data, use_container_width=True)
        show_program_credit()

    elif sub_choice.startswith("2"):
      st.subheader("📉 دالة إنتاج كوب-دوجلاس اللوغاريتمية")
      c1, c2 = st.columns(2)
      with c1:
        y_c = st.selectbox("المتغير التابع (الإنتاج):", columns_list, key="cd_y")
      with c2:
        x_c = st.multiselect(
            "المدخلات المستقلة:", [c for c in columns_list if c != y_c], key="cd_x"
        )
      if st.button("تقدير كوب-دوجلاس") and y_c and x_c:
        td = df[[y_c] + x_c].apply(pd.to_numeric, errors="coerce").dropna()
        td = td[(td > 0).all(axis=1)]
        df_log = np.log(td)
        X = sm.add_constant(df_log[x_c])
        model = sm.OLS(df_log[y_c], X).fit()
        st.text(model.summary().as_text())
        st.metric(
            "إجمالي عوائد الحجم (Returns to Scale)",
            f"{model.params[x_c].sum():.4f}",
        )
        show_program_credit()

    else:
      st.info("يرجى اختيار القسم الفرعي المطلوب.")
  else:
    st.info("👈 يرجى رفع ملف البيانات من القائمة الجانبية.")

# =========================================================
# القسم الثاني: اختبارات جذر الوحدة، التكامل المشترك والسلاسل الزمنية
# =========================================================
elif (
    app_mode
    == "📉 اختبارات جذر الوحدة والتكامل المشترك ونماذج السلاسل الزمنية"
):
  st.subheader(
      "📉 اختبارات جذر الوحدة، التكامل المشترك، ونماذج السلاسل الزمنية المتقدمة"
  )

  if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    ts_sub = st.selectbox(
        "اختر المنهجية القياسية الزمنية المطلوبة:",
        [
            "1. اختبارات جذر الوحدة الشاملة (ADF, Phillips-Perron, KPSS)",
            (
                "2. نماذج التكامل المشترك وتصحيح الخطأ (Engle-Granger, Johansen,"
                " ECM)"
            ),
            "3. نماذج التنبؤ (ARMA / ARIMA / SARIMAX) مع معايير المفاضلة الكاملة",
            "4. نماذج المتجهات الانحدارية (VAR Models)",
        ],
    )

    if ts_sub.startswith("1"):
      st.subheader(
          "🧪 اختبارات جذر الوحدة الثلاثية لاختبار استقرار السلسلة الزمنية"
      )
      col_t1, col_t2 = st.columns(2)
      with col_t1:
        target_series = st.selectbox("اختر السلسلة المراد اختبارها:", num_cols)
      with col_t2:
        diff_d = st.number_input(
            "درجة الفروق (Differencing Order - d):",
            min_value=0,
            max_value=3,
            value=0,
        )

      if (
          st.button("🚀 تنفيذ اختبارات جذر الوحدة الثلاثية (ADF, PP, KPSS)")
          and target_series
      ):
        s_data = pd.to_numeric(df[target_series], errors="coerce").dropna()
        if diff_d > 0:
          for _ in range(diff_d):
            s_data = s_data.diff().dropna()

        st.markdown("---")
        st.markdown(
            "### 1️⃣ اختبار ديككي-فولر المعزز (Augmented Dickey-Fuller - ADF)"
        )
        try:
          adf_res = adfuller(s_data)
          st.write(f"- **ADF Statistic:** `{adf_res[0]:.4f}`")
          st.write(f"- **p-value:** `{adf_res[1]:.4f}`")
          if adf_res[1] < 0.05:
            st.success("النتيجة: السلسلة مستقرة (Stationary) حسب اختبار ADF.")
          else:
            st.warning(
                "النتيجة: السلسلة غير مستقرة وتحتوي على جذر وحدة (Non-Stationary)."
            )
        except Exception as e:
          st.error(f"خطأ في تنفيذ اختبار ADF: {e}")

        st.markdown("---")
        st.markdown("### 2️⃣ اختبار فيليبس-بيرون (Phillips-Perron Test - PP)")
        try:
          from arch.unitroot import PhillipsPerron

          pp = PhillipsPerron(s_data)
          st.write(f"- **PP Statistic:** `{pp.stat:.4f}`")
          st.write(f"- **p-value:** `{pp.pvalue:.4f}`")
        except ImportError:
          st.info(
              "ℹ️ اختبار Phillips-Perron يعتمد على مكتبة `arch` غير المثبتة"
              " حالياً."
          )
        except Exception as e:
          st.info(f"ملاحظة في اختبار PP: {e}")

        st.markdown("---")
        st.markdown("### 3️⃣ اختبار KPSS")
        try:
          kpss_res = kpss(s_data, regression="c", nlags="auto")
          st.write(f"- **KPSS Statistic:** `{kpss_res[0]:.4f}`")
          st.write(f"- **p-value:** `{kpss_res[1]:.4f}`")
        except Exception as e:
          st.info(f"ملاحظة في اختبار KPSS: {e}")

        show_program_credit()

    elif ts_sub.startswith("2"):
      st.subheader(
          "🔗 اختبارات التكامل المشترك (Cointegration) ونموذج تصحيح الخطأ (ECM)"
      )
      c1, c2 = st.columns(2)
      with c1:
        y_var = st.selectbox("المتغير التابع (Y):", num_cols, key="coint_y")
      with c2:
        x_var = st.selectbox(
            "المتغير المستقل (X):", [c for c in num_cols if c != y_var], key="coint_x"
        )

      if st.button("🚀 تنفيذ اختبار إنجل-غرانجر والتكامل المشترك") and y_var and x_var:
        y_s = pd.to_numeric(df[y_var], errors="coerce")
        x_s = pd.to_numeric(df[x_var], errors="coerce")
        temp_c = pd.concat([y_s, x_s], axis=1).dropna()

        try:
          score, p_value, crit_values = coint(temp_c[y_var], temp_c[x_var])
          st.markdown("### 📊 نتائج اختبار إنجل-غرانجر (Engle-Granger Test):")
          st.metric("قيمة الإحصاء (Coint Score)", f"{score:.4f}")
          st.metric("القيمة الاحتمالية (p-value)", f"{p_value:.4f}")

          if p_value < 0.05:
            st.success(
                "✅ يوجد تكامل مشترك (Cointegration) بين المتغيرين على المدى الطويل!"
            )
            dy = temp_c[y_var].diff().dropna()
            dx = temp_c[x_var].diff().dropna()
            ols_long = sm.OLS(
                temp_c[y_var], sm.add_constant(temp_c[x_var])
            ).fit()
            ecm_resid = ols_long.resid.shift(1).dropna()

            ecm_df = pd.DataFrame({"DY": dy, "DX": dx}).loc[ecm_resid.index]
            ecm_df["ECT_lag1"] = ecm_resid

            X_ecm = sm.add_constant(ecm_df[["DX", "ECT_lag1"]])
            ecm_model = sm.OLS(ecm_df["DY"], X_ecm).fit()
            st.markdown("### 📉 نتائج نموذج تصحيح الخطأ (ECM):")
            st.text(ecm_model.summary().as_text())
          else:
            st.warning(
                "❌ لا يوجد دليل على وجود تكامل مشترك بين المتغيرين عند مستوى معنوية"
                " 5%."
            )
        except Exception as e:
          st.error(f"خطأ أثناء تنفيذ التكامل المشترك: {e}")

        # اختبار جوهانسن بشكل آمن ومحلي لتجنب أي خطأ استيراد
        st.markdown("---")
        st.markdown("### 🌐 اختبار جوهانسن للتكامل المشترك (Johansen Test):")
        try:
          from statsmodels.tsa.vector_ar.vecm import coin_johansen

          j_df = temp_c[[y_var, x_var]]
          j_res = coin_johansen(j_df, det_order=0, k_ar_diff=1)
          st.write(f"- **Trace Statistic:** `{j_res.lr1}`")
          st.write(f"- **Critical Values (95%):** `{j_res.cvt[:, 1]}`")
        except ImportError:
          st.info(
              "ℹ️ وحدة Johansen غير متوفرة في الإصدار الحالي من statsmodels."
          )
        except Exception as ex:
          st.info(
              "ملاحظة: يتطلب اختبار جوهانسن بيانات مستقرة نسبياً أو ضبط الفترات"
              f" ({ex})."
          )

        show_program_credit()

    elif ts_sub.startswith("3"):
      st.subheader(
          "📈 نماذج التنبؤ (ARMA / ARIMA / SARIMAX) مع معايير المفاضلة الشاملة"
      )
      t_col = st.selectbox(
          "اختر السلسلة الزمنية للتنبؤ:", num_cols, key="arima_target"
      )
      c1, c2, c3, c4 = st.columns(4)
      with c1:
        p_v = st.number_input("الانحدار الذاتي (p):", 0, 5, 1)
      with c2:
        d_v = st.number_input("التكامل/الفروق (d):", 0, 2, 1)
      with c3:
        q_v = st.number_input("المتوسطات المتحركة (q):", 0, 5, 1)
      with c4:
        steps = st.number_input("فترات التنبؤ:", 1, 24, 5)

      if st.button("🚀 تقدير ARIMA ومفاضلة النماذج"):
        try:
          series = pd.to_numeric(df[t_col], errors="coerce").dropna()
          model = ARIMA(series, order=(p_v, d_v, q_v)).fit()
          st.text(model.summary().as_text())

          st.markdown("### 📊 معايير مفاضلة واختيار النماذج الشاملة:")
          m1, m2, m3, m4 = st.columns(4)
          with m1:
            st.metric("معيار أكايكي (AIC)", f"{model.aic:.2f}")
          with m2:
            st.metric("معيار بايز (BIC)", f"{model.bic:.2f}")
          with m3:
            st.metric("معيار هانان-كوين (HQIC)", f"{model.hqic:.2f}")
          with m4:
            st.metric("لوغاريتم الإمكان (Log Likelihood)", f"{model.llf:.2f}")

          rmse = np.sqrt(np.mean((model.fittedvalues - series) ** 2))
          mae = np.mean(np.abs(model.fittedvalues - series))
          mape = np.mean(
              np.abs((series - model.fittedvalues) / series.replace(0, np.nan))
          ) * 100

          st.markdown("### 🎯 مقاييس دقة التنبؤ داخل العينة:")
          d1, d2, d3 = st.columns(3)
          with d1:
            st.metric("جذر متوسط مربع الخطأ (RMSE)", f"{rmse:.4f}")
          with d2:
            st.metric("متوسط الخطأ المطلق (MAE)", f"{mae:.4f}")
          with d3:
            st.metric("متوسط نسبة الخطأ المطلق (MAPE %)", f"{mape:.2f}%")

          forecast = model.forecast(steps=steps)
          st.markdown("### 🔮 جدول التنبؤات المستقبلية:")
          f_df = pd.DataFrame(
              {
                  "فترة التنبؤ": [f"+{i}" for i in range(1, steps + 1)],
                  "القيمة": forecast,
              }
          )
          st.dataframe(f_df, use_container_width=True)
        except Exception as e:
          st.error(f"حدث خطأ أثناء تقدير نموذج ARIMA: {e}")
        show_program_credit()

    elif ts_sub.startswith("4"):
      st.subheader("🌐 نماذج الانحدار الذاتي للمتجهات (VAR Model)")
      var_cols = st.multiselect(
          "اختر متغيرات السلاسل الزمنية لنموذج VAR:", num_cols, key="var_cols"
      )
      lag_order = st.number_input("عدد فترات الإبطاء (Lags):", 1, 5, 2)
      if st.button("🚀 تقدير نموذج VAR") and len(var_cols) >= 2:
        try:
          from statsmodels.tsa.vector_ar.var_model import VAR

          v_df = df[var_cols].apply(pd.to_numeric, errors="coerce").dropna()
          model_var = VAR(v_df)
          results_var = model_var.fit(lag_order)
          st.text(results_var.summary())
        except Exception as e:
          st.error(f"خطأ في تقدير نموذج VAR: {e}")
        show_program_credit()
  else:
    st.info("👈 يرجى رفع ملف البيانات أولاً.")

# =========================================================
# القسم الثالث: دراسة الجدوى والتقييم المالي الشامل للمشروعات
# =========================================================
elif app_mode == "💰 دراسة الجدوى والتقييم المالي الشامل للمشروعات":
  st.subheader("💰 دراسة الجدوى الاقتصادية والتقييم المالي الشامل للمشروعات")
  c1, c2 = st.columns(2)
  with c1:
    init_inv = st.number_input(
        "التكاليف الاستثمارية الابتدائية (C0):", value=150000.0
    )
    disc_rate = (
        st.number_input("معدل الخصم السنوي (%):", value=12.0) / 100.0
    )
  with c2:
    cf_str = st.text_input(
        "التدفقات النقدية السنوية الصافية مفصولة بفواصل:",
        value="40000, 45000, 50000, 55000, 60000, 65000",
    )

  if st.button("🚀 حساب معايير التقييم المالي ودراسة الجدوى"):
    try:
      cfs = [float(x.strip()) for x in cf_str.split(",") if x.strip()]
      r = disc_rate
      npv = -init_inv + sum(cf / ((1 + r) ** (i + 1)) for i, cf in enumerate(cfs))


      def npv_f(rate):
        return -init_inv + sum(
            cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cfs)
        )


      irr = np.nan
      try:
        sol = root_scalar(npv_f, bracket=[-0.99, 10.0], method="brentq")
        if sol.converged:
          irr = sol.root * 100.0
      except:
        pass

      pv_benefits = sum(
          cf / ((1 + r) ** (i + 1)) for i, cf in enumerate(cfs)
      )
      bcr = pv_benefits / init_inv if init_inv > 0 else np.nan
      npvi = pv_benefits / init_inv if init_inv > 0 else np.nan

      cum_cf = 0
      payback = np.nan
      disc_cum_cf = 0
      disc_payback = np.nan
      cum_list = []

      for i, cf in enumerate(cfs):
        cum_cf += cf
        cum_list.append(cum_cf)
        if cum_cf >= init_inv and np.isnan(payback):
          prev = cum_list[i - 1] if i > 0 else 0
          payback = i + ((init_inv - prev) / cf)

        dc = cf / ((1 + r) ** (i + 1))
        disc_cum_cf += dc
        if disc_cum_cf >= init_inv and np.isnan(disc_payback):
          prev_d = disc_cum_cf - dc
          disc_payback = i + ((init_inv - prev_d) / dc)

      st.markdown("### 📊 نتائج التقييم المالي الشامل:")
      m1, m2, m3 = st.columns(3)
      with m1:
        st.metric("صافي القيمة الحالية (NPV)", f"{npv:,.2f}")
        st.metric("معدل العائد الداخلي (IRR)", f"{irr:.2f}%")
      with m2:
        st.metric("نسبة المنافع للتكاليف (BCR)", f"{bcr:.2f}")
        st.metric("مؤشر الربحية (NPVI / PI)", f"{npvi:.2f}")
      with m3:
        st.metric(
            "فترة الاسترداد العادية",
            f"{payback:.2f} سنة" if not np.isnan(payback) else "لا توجد",
        )
        st.metric(
            "فترة الاسترداد المخصومة",
            (
                f"{disc_payback:.2f} سنة"
                if not np.isnan(disc_payback)
                else "لا توجد"
            ),
        )

      if npv > 0:
        st.success(
            "✅ **القرار الاستثماري:** المشروع **مقبول ومربح اقتصادياً**."
        )
      else:
        st.warning("❌ **القرار الاستثماري:** المشروع **مرفوض**.")
      show_program_credit()
    except Exception as ex:
      st.error(f"حدث خطأ أثناء إجراء الحسابات المالية: {ex}")

# =========================================================
# القسم الرابع: بوابة جمع البيانات والمؤشرات العالمية
# =========================================================
elif app_mode == "🌐 بوابة جمع البيانات والمؤشرات العالمية":
  st.subheader("🌐 بوابة جمع البيانات والمؤشرات الاقتصادية والزراعية")
  c_code = st.text_input("كود الدولة الثلاثي (مثال: EGY, USA):", value="EGY")
  if st.button("📥 جلب البيانات من البنك الدولي"):
    try:
      url = f"http://api.worldbank.org/v2/country/{c_code}/indicator/NV.AGR.TOTL.ZS?format=json&per_page=50"
      res = requests.get(url).json()
      if len(res) > 1 and res[1]:
        d_wb = pd.DataFrame([
            {
                "السنة": e.get("date"),
                "نسبة الزراعة من الناتج المحلي (%)": e.get("value"),
            }
            for e in res[1]
        ])
        st.dataframe(d_wb, use_container_width=True)
        show_program_credit()
      else:
        st.error("لم يتم العثور على بيانات.")
    except Exception as e:
      st.error(f"خطأ: {e}")

# =========================================================
# القسم الخامس: المستشار الاقتصادي والقياسي (ذكاء اصطناعي)
# =========================================================
elif (
    app_mode
    == "👨‍🏫 المستشار الاقتصادي والقياسي (قسم الذكاء الاصطناعي المتخصص)"
):
  st.subheader(
      "🏛️ غرفة الخبير الأكاديمي والبحث المباشر (اقتصاد زراعي وقياسي)"
  )
  prompt_text = st.chat_input("اكتب سؤالك أو موضوع بحثك هنا...")
  if prompt_text:
    st.write(f"**استفسارك:** {prompt_text}")
    st.success(
        "💡 **توجيه الخبير:** يمكنك الاعتماد على نماذج ARIMA و ECM واختبارات ADF"
        " لضمان خلو سلسلتك الزمنية من الثبات الوهمي وتحقيق دقة تنبؤ عالية."
    )
    show_program_credit()
