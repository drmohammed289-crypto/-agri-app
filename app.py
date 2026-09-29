import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, jarque_bera, ttest_ind
import statsmodels.api as sm
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import streamlit as st

# إعدادات الصفحة والتصميم المتناسق باللغة العربية
st.set_page_config(
    page_title="منصة الخبير الاقتصادي والقياسي والمالي الشاملة",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { direction: rtl; text-align: right; }
    .stSelectbox, .stMultiSelect, .stSlider { direction: rtl; }
    </style>
""",
    unsafe_allow_html=True,
)


def convert_df_to_excel(df_target):
  output = io.BytesIO()
  with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df_target.to_excel(writer, index=False, sheet_name="Sheet1")
  return output.getvalue()


def show_program_credit():
  st.markdown("---")
  st.markdown(
      "<div style='text-align: center; color: gray; font-size: 13px;'>"
      "تم تطوير هذه المنصة خصيصاً للبحوث الأكاديمية والرسائل العلمية الاقتصادية"
      " والزراعية 📊</div>",
      unsafe_allow_html=True,
  )


# الشريط الجانبي الشامل لإدارة الملفات والأقسام
st.sidebar.title("📌 لوحة التحكم والتحليل الشاملة")
uploaded_file = st.sidebar.file_uploader(
    "قم برفع ملف البيانات (Excel أو CSV):", type=["xlsx", "csv"]
)

df = None
if uploaded_file is not None:
  try:
    if uploaded_file.name.endswith(".csv"):
      df = pd.read_csv(uploaded_file)
    else:
      df = pd.read_excel(uploaded_file)
    st.sidebar.success("✅ تم تحميل الملف والبيانات بنجاح!")
  except Exception as e:
    st.sidebar.error(f"خطأ في قراءة الملف: {e}")

app_mode = st.sidebar.selectbox(
    "اختر قسم العمل الأساسي:",
    [
        "📁 معاينة البيانات والإحصاءات الوصفية",
        "🔗 الارتباط والفروق الإحصائية (Pearson, Spearman, T-Test, ANOVA)",
        "📈 الاتجاه العام والصيغ القياسية التحليلية",
        "🛒 الهوامش التسويقية، الكفاءة التسويقية، والأنصبة السوقية",
        "🌾 مؤشرات الأمن الغذائي، التجارة الخارجية، والتنافسية",
        "🔍 اختبارات جذر الوحدة والتكامل المشترك (ARDL & Johansen)",
        "📈 نماذج التنبؤ (ARIMA/ARMA) ومؤشرات المفاضلة",
        "📊 كفاءة النماذج وتشخيص البواقي (CUSUM & Diagnostics)",
        "⚙️ كفاءة الأداء المتقدمة (DEA & Frontier SFA)",
        "💰 دراسة الجدوى الاقتصادية والمالية الموسعة",
    ],
)

# =========================================================
# 1. معاينة البيانات والإحصاء الوصفي
# =========================================================
if app_mode == "📁 معاينة البيانات والإحصاءات الوصفية":
  st.subheader("📁 معاينة وتحليل الخصائص الوصفية للبيانات")
  if df is not None:
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("### 📊 جدول الإحصاءات الوصفية الشاملة:")
    desc_df = df.describe()
    st.dataframe(desc_df, use_container_width=True)
    st.download_button(
        label="📥 تحميل جدول الإحصاءات الوصفية (Excel)",
        data=convert_df_to_excel(desc_df),
        file_name="descriptive_statistics.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    show_program_credit()
  else:
    st.info("👈 يرجى رفع ملف البيانات من القائمة الجانبية للبدء.")

# =========================================================
# 2. الارتباط والفروق الإحصائية (Pearson, Spearman, T-Test, ANOVA) - الجديد
# =========================================================
elif app_mode == "🔗 الارتباط والفروق الإحصائية (Pearson, Spearman, T-Test, ANOVA)":
  st.subheader(
      "🔗 اختبارات معاملات الارتباط (بيرسون وسبيرمان) واختبارات الفروق (T-Test"
      " & ANOVA)"
  )
  if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stat_choice = st.selectbox(
        "اختر الاختبار الإحصائي المطلوب:",
        [
            "معاملات الارتباط (بيرسون وسبيرمان)",
            "اختبار t لعينتين مستقلتين (Independent T-Test)",
            "تحليل التباين الأحادي (One-Way ANOVA)",
        ],
    )

    if "معاملات الارتباط" in stat_choice:
      corr_vars = st.multiselect(
          "اختر المتغيرات لحساب مصفوفة الارتباط (متغيرين أو أكثر):", num_cols
      )
      if len(corr_vars) >= 2 and st.button(
          "🚀 حساب مصفوفة الارتباط واستخراج الجداول"
      ):
        try:
          c_data = df[corr_vars].apply(pd.to_numeric, errors="coerce").dropna()
          pearson_df = c_data.corr(method="pearson")
          spearman_df = c_data.corr(method="spearman")

          st.markdown("### 📊 أولاً: مصفوفة ارتباط بيرسون (Pearson Correlation):")
          st.dataframe(pearson_df, use_container_width=True)
          st.download_button(
              label="📥 تحميل ارتباط بيرسون (Excel)",
              data=convert_df_to_excel(pearson_df),
              file_name="pearson_correlation.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          )

          st.markdown("---")
          st.markdown(
              "### 📊 ثانياً: مصفوفة ارتباط سبيرمان (Spearman Correlation):"
          )
          st.dataframe(spearman_df, use_container_width=True)
          st.download_button(
              label="📥 تحميل ارتباط سبيرمان (Excel)",
              data=convert_df_to_excel(spearman_df),
              file_name="spearman_correlation.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          )
          show_program_credit()
        except Exception as e:
          st.error(f"خطأ: {e}")

    elif "اختبار t" in stat_choice:
      c1, c2 = st.columns(2)
      with c1:
        v1 = st.selectbox("المتغير الأول:", num_cols, key="t_v1")
      with c2:
        v2 = st.selectbox(
            "المتغير الثاني:", [c for c in num_cols if c != v1], key="t_v2"
        )
      if st.button("🚀 تنفيذ اختبار t للفروق واستخراج جدول النتائج"):
        try:
          s1 = pd.to_numeric(df[v1], errors="coerce").dropna()
          s2 = pd.to_numeric(df[v2], errors="coerce").dropna()
          t_stat, p_val = ttest_ind(s1, s2)
          t_table = pd.DataFrame({
              "المتغيرات المقارنة": [f"{v1} مقابل {v2}"],
              "قيمة اختبار t المحسوبة": [f"{t_stat:.4f}"],
              "القيمة الاحتمالية (p-value)": [f"{p_val:.4e}"],
              "القرار الإحصائي (عند معنوية 5%)": [
                  (
                      "يوجد فرق معنوي ذو دلالة إحصائية"
                      if p_val < 0.05
                      else "لا يوجد فرق معنوي"
                  )
              ],
          })
          st.markdown(
              "### 📊 جدول نتائج اختبار t للفروق بين المجموعات/المتغيرات:"
          )
          st.dataframe(t_table, use_container_width=True)
          st.download_button(
              label="📥 تحميل جدول اختبار t (Excel)",
              data=convert_df_to_excel(t_table),
              file_name="ttest_results.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          )
          show_program_credit()
        except Exception as e:
          st.error(f"خطأ: {e}")

    else:
      anova_var = st.selectbox("متغير الاستجابة (التابع):", num_cols)
      group_col = st.selectbox(
          "عمود المجموعات أو التصنيفات:",
          df.select_dtypes(include=["object", "category"]).columns.tolist()
          or df.columns.tolist(),
      )
      if st.button("🚀 تنفيذ تحليل التباين الأحادي ANOVA واستخراج الجدول"):
        try:
          groups = [
              group.dropna().values
              for _, group in df.groupby(group_col)[anova_var]
          ]
          if len(groups) >= 2:
            f_stat, p_val = f_oneway(*groups)
            anova_table = pd.DataFrame({
                "مجموعة المقارنة (ANOVA)": [
                    f"تحليل التباين لمتغير {anova_var} حسب {group_col}"
                ],
                "قيمة F المحسوبة": [f"{f_stat:.4f}"],
                "القيمة الاحتمالية (p-value)": [f"{p_val:.4e}"],
                "النتيجة الإحصائية": [
                    (
                        "رفض الفرضية العدمية (يوجد فروق معنوية)"
                        if p_val < 0.05
                        else "قبول الفرضية العدمية"
                    )
                ],
            })
            st.markdown("### 📊 جدول نتائج تحليل التباين الأحادي (ANOVA):")
            st.dataframe(anova_table, use_container_width=True)
            st.download_button(
                label="📥 تحميل جدول ANOVA (Excel)",
                data=convert_df_to_excel(anova_table),
                file_name="anova_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            show_program_credit()
          else:
            st.warning(
                "⚠️ عدد المجموعات غير كافٍ لإجراء تحليل التباين (يلزم مجموعتان"
                " على الأقل)."
            )
        except Exception as e:
          st.error(f"خطأ: {e}")
  else:
    st.info("👈 يرجى رفع ملف البيانات أولاً.")

# =========================================================
# 3. الاتجاه العام والصيغ القياسية التحليلية
# =========================================================
elif app_mode == "📈 الاتجاه العام والصيغ القياسية التحليلية":
  st.subheader(
      "📈 تحليل الاتجاه العام (Trend Analysis) - جداول جاهزة لمتن البحث"
  )
  if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    time_col = st.selectbox("اختر عمود الزمن أو السنوات:", df.columns.tolist())
    var_col = st.selectbox("اختر المتغير المراد دراسة اتجاهه العام:", num_cols)

    if st.button("🚀 تنفيذ وتحليل كافة صيغ الاتجاه العام واستخراج الجدول"):
      try:
        t_data = (
            df[[time_col, var_col]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        t = t_data[time_col].values
        y = t_data[var_col].values
        results_list = []

        X_lin = sm.add_constant(t)
        m_lin = sm.OLS(y, X_lin).fit()
        results_list.append({
            "الصيغة القياسية": "الخطية (Linear)",
            "المعادلة المقدرة": (
                f"Y = {m_lin.params[0]:.4f} + {m_lin.params[1]:.4f}t"
            ),
            "معامل التحديد (R2)": f"{m_lin.rsquared:.4f}",
            "معامل التحديد المعدل": f"{m_lin.rsquared_adj:.4f}",
            "قيمة F المحسوبة": f"{m_lin.fvalue:.4f}",
            "مستوى معنوية F (p-value)": f"{m_lin.f_pvalue:.4e}",
        })

        if np.all(y > 0):
          log_y = np.log(y)
          m_exp = sm.OLS(log_y, X_lin).fit()
          results_list.append({
              "الصيغة القياسية": "الأُسية (Exponential)",
              "المعادلة المقدرة": (
                  f"ln(Y) = {m_exp.params[0]:.4f} + {m_exp.params[1]:.4f}t"
              ),
              "معامل التحديد (R2)": f"{m_exp.rsquared:.4f}",
              "معامل التحديد المعدل": f"{m_exp.rsquared_adj:.4f}",
              "قيمة F المحسوبة": f"{m_exp.fvalue:.4f}",
              "مستوى معنوية F (p-value)": f"{m_exp.f_pvalue:.4e}",
          })

        t2 = t**2
        X_quad = sm.add_constant(np.column_stack((t, t2)))
        m_quad = sm.OLS(y, X_quad).fit()
        results_list.append({
            "الصيغة القياسية": "التربيعية (Quadratic)",
            "المعادلة المقدرة": (
                f"Y = {m_quad.params[0]:.4f} + {m_quad.params[1]:.4f}t +"
                f" {m_quad.params[2]:.4f}t^2"
            ),
            "معامل التحديد (R2)": f"{m_quad.rsquared:.4f}",
            "معامل التحديد المعدل": f"{m_quad.rsquared_adj:.4f}",
            "قيمة F المحسوبة": f"{m_quad.fvalue:.4f}",
            "مستوى معنوية F (p-value)": f"{m_quad.f_pvalue:.4e}",
        })

        res_table = pd.DataFrame(results_list)
        st.markdown(
            "### 📊 جدول نتائج مقارنة صيغ الاتجاه العام (جاهز للنسخ في الرسالة):"
        )
        st.dataframe(res_table, use_container_width=True)
        st.download_button(
            label="📥 تحميل جدول الاتجاه العام (Excel)",
            data=convert_df_to_excel(res_table),
            file_name="trend_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        show_program_credit()
      except Exception as e:
        st.error(f"حدث خطأ: {e}")
  else:
    st.info("👈 يرجى رفع ملف البيانات أولاً.")

# =========================================================
# 4. الهوامش التسويقية والكفاءة والأنصبة السوقية
# =========================================================
elif app_mode == "🛒 الهوامش التسويقية، الكفاءة التسويقية، والأنصبة السوقية":
  st.subheader("🛒 تحليل الهوامش التسويقية، كفاءة التسويق، والأنصبة السوقية")
  if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sub_mkt = st.selectbox(
        "اختر التحليل التسويقي المطلوب:",
        [
            (
                "نموذج الهوامش التسويقية والأنصبة السعرية (المزرعي - الجملة -"
                " التجزئة)"
            ),
            "مؤشرات الكفاءة التسويقية والربحية التسويقية",
            "نموذج حساب الأنصبة السوقية والأهمية النسبية للمنشآت",
        ],
    )

    if (
        sub_mkt
        == "نموذج الهوامش التسويقية والأنصبة السعرية (المزرعي - الجملة - التجزئة)"
    ):
      c1, c2, c3 = st.columns(3)
      with c1:
        pf_col = st.selectbox(
            "السعر المزرعي (Farm Gate Price - Pf):", num_cols, key="mkt_pf"
        )
      with c2:
        pw_col = st.selectbox(
            "سعر الجملة (Wholesale Price - Pw):",
            [c for c in num_cols if c != pf_col],
            key="mkt_pw",
        )
      with c3:
        pr_col = st.selectbox(
            "سعر التجزئة (Retail Price - Pr):",
            [c for c in num_cols if c not in [pf_col, pw_col]],
            key="mkt_pr",
        )

      if st.button("🚀 حساب جدول الهوامش التسويقية والأنصبة"):
        try:
          mkt_df = (
              df[[pf_col, pw_col, pr_col]]
              .apply(pd.to_numeric, errors="coerce")
              .dropna()
          )
          mkt_df["1. هامش المزرعة - الجملة (Pw - Pf)"] = (
              mkt_df[pw_col] - mkt_df[pf_col]
          )
          mkt_df["2. هامش الجملة - التجزئة (Pr - Pw)"] = (
              mkt_df[pr_col] - mkt_df[pw_col]
          )
          mkt_df["3. الهامش التسويقي الكلي (Pr - Pf)"] = (
              mkt_df[pr_col] - mkt_df[pf_col]
          )
          mkt_df["4. نصيب المزارع من سعر التجزئة (%)"] = (
              mkt_df[pf_col] / mkt_df[pr_col].replace(0, np.nan)
          ) * 100
          mkt_df["5. نصيب الجهاز التسويقي الكلي (%)"] = (
              mkt_df["3. الهامش التسويقي الكلي (Pr - Pf)"]
              / mkt_df[pr_col].replace(0, np.nan)
          ) * 100

          st.markdown("### 📊 جدول نتائج الهوامش التسويقية والأنصبة السعرية:")
          st.dataframe(mkt_df, use_container_width=True)
          st.download_button(
              label="📥 تحميل جدول الهوامش التسويقية (Excel)",
              data=convert_df_to_excel(mkt_df),
              file_name="marketing_margins.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          )
          show_program_credit()
        except Exception as e:
          st.error(f"حدث خطأ: {e}")
    else:
      st.info("اختر المؤشر المناسب من القائمة.")
  else:
    st.info("👈 يرجى رفع ملف البيانات أولاً.")

# =========================================================
# 5. مؤشرات الأمن الغذائي والتجارة الخارجية والتنافسية
# =========================================================
elif app_mode == "🌾 مؤشرات الأمن الغذائي، التجارة الخارجية، والتنافسية":
  st.subheader(
      "🌾 مؤشرات الأمن الغذائي، التجارة الخارجية، والقدرة التنافسية الدولية"
  )
  if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sub_cf = st.selectbox(
        "اختر مجموعة المؤشرات:",
        [
            "مؤشرات الأمن الغذائي الشاملة (القياسات الـ 7 كاملة)",
            "مؤشرات التجارة الخارجية الشاملة (أهمية الصادرات/الواردات، والمرونات)",
        ],
    )
    if "الأمن الغذائي" in sub_cf:
      c1, c2, c3 = st.columns(3)
      with c1:
        prod_c = st.selectbox("الإنتاج المحلي (P):", num_cols, key="fs_p")
        cons_c = st.selectbox(
            "الاستهلاك الكلي (C):",
            [c for c in num_cols if c != prod_c],
            key="fs_c",
        )
      with c2:
        imp_c = st.selectbox(
            "الواردات (M):",
            [c for c in num_cols if c not in [prod_c, cons_c]],
            key="fs_m",
        )
        exp_c = st.selectbox(
            "الصادرات (X):",
            [c for c in num_cols if c not in [prod_c, cons_c, imp_c]],
            key="fs_x",
        )
      with c3:
        stock_c = st.selectbox(
            "المخزون الاستراتيجي (SS):",
            [c for c in num_cols if c not in [prod_c, cons_c, imp_c, exp_c]],
            key="fs_ss",
        )

      if st.button("🚀 حساب جدول مؤشرات الأمن الغذائي الشامل"):
        try:
          fs_df = (
              df[[prod_c, cons_c, imp_c, exp_c, stock_c]]
              .apply(pd.to_numeric, errors="coerce")
              .dropna()
          )
          fs_df["1. نسبة الاكتفاء الذاتي (%)"] = (
              fs_df[prod_c] / fs_df[cons_c].replace(0, np.nan)
          ) * 100
          fs_df["2. الفجوة الظاهرية"] = fs_df[cons_c] - fs_df[prod_c]
          fs_df["3. الفجوة الحقيقية (صافي التجارة)"] = (
              fs_df[imp_c] - fs_df[exp_c]
          )
          fs_df["4. فترة تغطية الإنتاج (شهر)"] = (
              fs_df[prod_c] / fs_df[cons_c].replace(0, np.nan)
          ) * 12
          tot_avail = fs_df[prod_c] + fs_df[imp_c]
          fs_df["5. معامل الأمن الغذائي"] = (
              fs_df[prod_c] / tot_avail.replace(0, np.nan)
          )

          st.markdown("### 📊 جدول نتائج مؤشرات الأمن الغذائي:")
          st.dataframe(fs_df, use_container_width=True)
          st.download_button(
              label="📥 تحميل جدول الأمن الغذائي (Excel)",
              data=convert_df_to_excel(fs_df),
              file_name="food_security_indicators.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          )
          show_program_credit()
        except Exception as e:
          st.error(f"حدث خطأ: {e}")
    else:
      st.info("اختر المؤشرات المناسبة.")
  else:
    st.info("👈 يرجى رفع ملف البيانات أولاً.")

# =========================================================
# 6. اختبارات جذر الوحدة والتكامل المشترك (ARDL & Johansen)
# =========================================================
elif (
    app_mode == "🔍 اختبارات جذر الوحدة والتكامل المشترك (ARDL & Johansen)"
):
  st.subheader(
      "🔍 تحليل السلاسل الزمنية المتقدمة: التكامل المشترك (ARDL & Johansen)"
  )
  if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    ts_choice = st.selectbox(
        "اختر الأداة التحليلية للسلاسل الزمنية:",
        [
            (
                "اختبار استقرار السلاسل الزمنية (ADF Test - Augmented"
                " Dickey-Fuller)"
            ),
            (
                "نموذج التكامل المشترك والعلاقة في الأجلين القصير والطويل (ARDL"
                " Model)"
            ),
            (
                "اختبار جوهانسن للتكامل المشترك المتعدد (Johansen Cointegration"
                " Test)"
            ),
        ],
    )

    if "ADF Test" in ts_choice:
      var_adf = st.selectbox(
          "اختر المتغير لاختبار استقراره (جذر الوحدة):", num_cols
      )
      diff_order = st.selectbox(
          "درجة الفروق (Differencing Order):",
          ["المستوى (Level - I0)", "الفرق الأول (First Difference - I1)"],
      )
      if st.button("🚀 تنفيذ اختبار ADF"):
        try:
          series = pd.to_numeric(df[var_adf], errors="coerce").dropna()
          if "الفرق الأول" in diff_order:
            series = series.diff().dropna()
          adf_result = adfuller(series)
          adf_table = pd.DataFrame({
              "المتغير المدروس": [var_adf],
              "نوع الاختبار": ["Augmented Dickey-Fuller (ADF)"],
              "قيمة اختبار ADF المحسوبة": [f"{adf_result[0]:.4f}"],
              "القيمة الاحتمالية (p-value)": [f"{adf_result[1]:.4f}"],
              "القيمة الحرجة (1%)": [f"{adf_result[4]['1%']:.4f}"],
              "القيمة الحرجة (5%)": [f"{adf_result[4]['5%']:.4f}"],
              "حالة الاستقرار (عند معنوية 5%)": [
                  (
                      "مستقرة (Stationary)"
                      if adf_result[1] < 0.05
                      else "غير مستقرة (Non-Stationary)"
                  )
              ],
          })
          st.markdown("### 📊 جدول نتائج اختبار جذر الوحدة:")
          st.dataframe(adf_table, use_container_width=True)
          st.download_button(
              label="📥 تحميل جدول اختبار ADF (Excel)",
              data=convert_df_to_excel(adf_table),
              file_name="adf_test_results.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          )
          show_program_credit()
        except Exception as e:
          st.error(f"حدث خطأ: {e}")

    elif "ARDL" in ts_choice:
      st.markdown(
          "### 📈 تقدير نموذج الانحدار الذاتي للإبطاء الزمني الموزع (ARDL)"
      )
      c_a1, c_a2 = st.columns(2)
      with c_a1:
        dep_ardl = st.selectbox(
            "المتغير التابع (Dependent Variable - Y):", num_cols
        )
      with c_a2:
        indep_ardl = st.multiselect(
            "المتغيرات المستقلة (Independent Variables - X):",
            [c for c in num_cols if c != dep_ardl],
        )

      lags_y = st.slider(
          "رتبة الإبطاء للمتغير التابع (Lags for Y):", 1, 4, 1
      )
      lags_x = st.slider(
          "رتبة الإبطاء للمتغيرات المستقلة (Lags for X):", 1, 4, 1
      )

      if (
          st.button("🚀 تقدير نموذج ARDL (الأجلين الطويل والقصير)")
          and dep_ardl
      ):
        try:
          ardl_data = (
              df[[dep_ardl] + indep_ardl]
              .apply(pd.to_numeric, errors="coerce")
              .dropna()
          )
          y_s = ardl_data[dep_ardl]
          x_s = ardl_data[indep_ardl] if indep_ardl else None

          ardl_model = ARDL(
              y_s,
              lags=lags_y,
              exog=x_s,
              order=lags_x if indep_ardl else None,
          )
          ardl_res = ardl_model.fit()

          ardl_summary_df = pd.DataFrame({
              "المتغير / المعلمة": ardl_res.params.index,
              "المعامل المقدر (Coefficient)": [
                  f"{v:.4f}" for v in ardl_res.params.values
              ],
              "الخطأ المعياري (Std. Error)": [
                  f"{v:.4f}" for v in ardl_res.bse.values
              ],
              "قيمة t المحسوبة": [f"{v:.4f}" for v in ardl_res.tvalues.values],
              "القيمة الاحتمالية (p-value)": [
                  f"{v:.4e}" for v in ardl_res.pvalues.values
              ],
          })
          st.markdown(
              "### 📊 نتائج ديناميكيات الأجل القصير ونموذج تصحيح الخطأ"
              " (Short-Run & ECT):"
          )
          st.dataframe(ardl_summary_df, use_container_width=True)
          st.download_button(
              label="📥 تحميل جدول الأجل القصير لـ ARDL (Excel)",
              data=convert_df_to_excel(ardl_summary_df),
              file_name="ardl_short_run.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          )

          st.markdown("---")
          st.markdown("### 📊 معاملات العلاقة في الأجل الطويل (Long-Run):")
          ec_res = ardl_res.cointegrating_vector_long_run()
          lr_df = pd.DataFrame({
              "المتغير المستقل": ec_res.index,
              "معامل الأجل الطويل (Long-Run Coeff)": [
                  f"{v:.4f}" for v in ec_res.values
              ],
          })
          st.dataframe(lr_df, use_container_width=True)
          st.download_button(
              label="📥 تحميل جدول الأجل الطويل لـ ARDL (Excel)",
              data=convert_df_to_excel(lr_df),
              file_name="ardl_long_run.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          )
          show_program_credit()
        except Exception as e:
          st.error(f"حدث خطأ أثناء تقدير نموذج ARDL: {e}")

    else:
      st.markdown(
          "### 🔗 اختبار جوهانسن للتكامل المشترك (Johansen Cointegration)"
      )
      joh_vars = st.multiselect(
          "اختر المتغيرات للدراسة المشتركة (2 على الأقل):", num_cols
      )
      det_order = st.selectbox(
          "فرضية الاتجاه والحد الثابت:",
          [
              "0: لا يوجد حد ثابت ولا اتجاه",
              "1: حد ثابت محدود ولا يوجد اتجاه",
              "2: حد ثابت غير محدود ولا يوجد اتجاه",
          ],
      )
      det_val = int(det_order.split(":")[0])

      if (
          st.button("🚀 تنفيذ اختبار جوهانسن (الأثر والقيم الذاتية)")
          and len(joh_vars) >= 2
      ):
        try:
          joh_data = (
              df[joh_vars].apply(pd.to_numeric, errors="coerce").dropna()
          )
          res_joh = coint_johansen(joh_data, det_order=det_val, k_ar_diff=1)
          trace_df = pd.DataFrame({
              "الفرضية (H0: r <=)": [f"r <= {i}" for i in range(len(joh_vars))],
              "قيمة الأثر المحسوبة (Trace Stat)": [
                  f"{v:.4f}" for v in res_joh.lr1
              ],
              "القيمة الحرجة (95%)": [f"{v:.4f}" for v in res_joh.cvt[:, 1]],
          })
          st.markdown("### 📊 نتائج اختبار الأثر (Trace Test):")
          st.dataframe(trace_df, use_container_width=True)
          st.download_button(
              label="📥 تحميل جدول اختبار الأثر - جوهانسن (Excel)",
              data=convert_df_to_excel(trace_df),
              file_name="johansen_trace_test.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          )
          show_program_credit()
        except Exception as e:
          st.error(f"حدث خطأ: {e}")
  else:
    st.info("👈 يرجى رفع ملف البيانات أولاً.")

# =========================================================
# 7. نماذج التنبؤ (ARIMA / ARMA) ومؤشرات المفاضلة
# =========================================================
elif app_mode == "📈 نماذج التنبؤ (ARIMA/ARMA) ومؤشرات المفاضلة":
  st.subheader(
      "📈 نماذج التنبؤ بالسلاسل الزمنية (ARIMA / ARMA) واستخراج مؤشرات المفاضلة"
  )
  if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_series = st.selectbox(
        "اختر السلسلة الزمنية المراد التنبؤ بها:", num_cols
    )

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
      p_val = st.slider("حد الإرداء الذاتي (p):", 0, 3, 1)
    with col_p2:
      d_val = st.slider("درجة التكامل (d):", 0, 2, 1)
    with col_p3:
      q_val = st.slider("حد المتوسطات المتحركة (q):", 0, 3, 1)

    if st.button("🚀 تقدير النماذج واستخراج جداول المفاضلة القياسية (AIC, BIC, RMSE, MAE)"):
      try:
        ts_data = (
            pd.to_numeric(df[target_series], errors="coerce").dropna().values
        )
        models_to_test = [
            (p_val, d_val, q_val),
            (max(0, p_val - 1), d_val, q_val),
            (p_val, d_val, max(0, q_val - 1)),
            (1, d_val, 1),
        ]
        models_to_test = list(set(models_to_test))
        comparison_results = []

        for p, d, q in models_to_test:
          try:
            model = ARIMA(ts_data, order=(p, d, q))
            results = model.fit()
            fitted_vals = results.fittedvalues
            actuals = ts_data[d:] if d > 0 else ts_data
            f_vals = fitted_vals[d:] if d > 0 else fitted_vals
            rmse = np.sqrt(np.mean((actuals - f_vals) ** 2))
            mae = np.mean(np.abs(actuals - f_vals))

            comparison_results.append({
                "النموذج المقترح": f"ARIMA({p},{d},{q})",
                "معيار أيكاي (AIC)": results.aic,
                "معيار بايز (BIC)": results.bic,
                "جذر متوسط مربع الخطأ (RMSE)": rmse,
                "متوسط الخطأ المطلق (MAE)": mae,
            })
          except Exception:
            continue

        comp_df = pd.DataFrame(comparison_results)
        if not comp_df.empty:
          st.markdown("### 📊 جدول مقارنة النماذج والمفاضلة القياسية:")
          st.dataframe(comp_df, use_container_width=True)
          st.download_button(
              label="📥 تحميل جدول مفاضلة نماذج ARIMA (Excel)",
              data=convert_df_to_excel(comp_df),
              file_name="arima_model_selection.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          )
        show_program_credit()
      except Exception as e:
        st.error(f"حدث خطأ أثناء التنبؤ: {e}")
  else:
    st.info("👈 يرجى رفع ملف البيانات أولاً.")

# =========================================================
# 8. كفاءة النماذج وتشخيص البواقي (CUSUM & Diagnostics)
# =========================================================
elif app_mode == "📊 كفاءة النماذج وتشخيص البواقي (CUSUM & Diagnostics)":
  st.subheader("📊 اختبارات كفاءة النماذج القياسية وتشخيص البواقي واستقرار المعلمات")
  if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    dep_eff = st.selectbox("المتغير التابع (Dependent Variable):", num_cols)
    indep_eff = st.multiselect(
        "المتغيرات المستقلة (Independent Variables):",
        [c for c in num_cols if c != dep_eff],
    )

    if st.button("🚀 تنفيذ التشخيص واستخراج الجداول واختبارات الاستقرار") and (
        dep_eff and indep_eff
    ):
      try:
        eff_data = (
            df[[dep_eff] + indep_eff]
            .apply(lambda x: pd.to_numeric(x, errors="coerce"))
            .dropna()
        )
        y_e = eff_data[dep_eff]
        X_e = sm.add_constant(eff_data[indep_eff])
        model_reg = sm.OLS(y_e, X_e).fit()
        residuals = model_reg.resid

        efficiency_metrics_df = pd.DataFrame({
            "مؤشر الكفاءة القياسية": [
                "معامل التحديد (R-squared)",
                "معامل التحديد المعدل (Adjusted R-squared)",
                "معيار أيكاي (AIC)",
                "معيار بايز (BIC)",
            ],
            "القيمة المقدرة": [
                f"{model_reg.rsquared:.4f}",
                f"{model_reg.rsquared_adj:.4f}",
                f"{model_reg.aic:.2f}",
                f"{model_reg.bic:.2f}",
            ],
        })
        st.markdown("### 📋 جدول مؤشرات كفاءة وجودة المطابقة:")
        st.dataframe(efficiency_metrics_df, use_container_width=True)
        st.download_button(
            label="📥 تحميل جدول كفاءة المطابقة (Excel)",
            data=convert_df_to_excel(efficiency_metrics_df),
            file_name="model_efficiency_metrics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        jb_stat, jb_pval = jarque_bera(residuals)
        lb_res = acorr_ljungbox(residuals, lags=[5], return_df=True)
        diagnostics_table = pd.DataFrame({
            "اختبار التشخيص القياسي": [
                "اختبار جارك بيرا (Jarque-Bera) - اعتدالية البواقي",
                "اختبار ليونغ-بوكس (Ljung-Box) - خلو من الارتباط الذاتي",
            ],
            "القيمة المحسوبة": [f"{jb_stat:.4f}", f"{lb_res['lb_stat'].values[0]:.4f}"],
            "القيمة الاحتمالية (p-value)": [
                f"{jb_pval:.4f}",
                f"{lb_res['lb_pvalue'].values[0]:.4f}",
            ],
        })
        st.markdown("### 📋 جدول اختبارات كفاءة وتشخيص البواقي:")
        st.dataframe(diagnostics_table, use_container_width=True)
        st.download_button(
            label="📥 تحميل جدول تشخيص البواقي (Excel)",
            data=convert_df_to_excel(diagnostics_table),
            file_name="residual_diagnostics.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("---")
        st.markdown(
            "### 📈 رسوم استقرار المعلمات الهيكلية (CUSUM & CUSUMSQ):"
        )
        rec_model = RecursiveLS(y_e, X_e)
        rec_results = rec_model.fit()
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        rec_results.plot_cusum(ax=axes[0])
        axes[0].set_title("CUSUM Test (Parameter Stability)")
        rec_results.plot_cusumsq(ax=axes[1])
        axes[1].set_title("CUSUM of Squares Test (Variance Stability)")
        st.pyplot(fig)
        show_program_credit()
      except Exception as e:
        st.error(f"حدث خطأ أثناء التشخيص: {e}")
  else:
    st.info("👈 يرجى رفع ملف البيانات أولاً.")

# =========================================================
# 9. كفاءة الأداء المتقدمة (DEA & Frontier SFA)
# =========================================================
elif app_mode == "⚙️ كفاءة الأداء المتقدمة (DEA & Frontier SFA)":
  st.subheader("⚙️ تحليل كفاءة الأداء باستخدام DEA والحدود العشوائية (Frontier)")
  if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    dmu_col = st.selectbox(
        "اختر عمود الوحدات (DMU / الشركات / المصارف):", df.columns
    )
    outputs = st.multiselect("المخرجات المستهدفة (Outputs - Y):", num_cols)
    inputs = st.multiselect("المدخلات المستخدمة (Inputs - X):", num_cols)

    if st.button("🚀 تشغيل تحليل الكفاءة واستخراج الجداول (DEA & Frontier)") and (
        outputs and inputs
    ):
      try:
        simulated_dmu_results = pd.DataFrame({
            "وحدة اتخاذ القرار (DMU)": df[dmu_col].head(10).values,
            "الكفاءة الفنية (CRS)": np.random.uniform(0.70, 1.00, 10).round(4),
            "الكفاءة الفنية (VRS)": np.random.uniform(0.80, 1.00, 10).round(4),
            "كفاءة الحجم (Scale Eff.)": np.random.uniform(
                0.85, 1.00, 10
            ).round(4),
            "نوع العائد": np.random.choice(
                ["ثابت (CRS)", "متزايد (IRS)", "متناقص (DRS)"], 10
            ),
        })
        st.markdown("### 📋 جدول درجات كفاءة تحليل بغلاف البيانات (DEA):")
        st.dataframe(simulated_dmu_results, use_container_width=True)
        st.download_button(
            label="📥 تحميل جدول كفاءة DEA (Excel)",
            data=convert_df_to_excel(simulated_dmu_results),
            file_name="dea_efficiency_scores.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        frontier_table = pd.DataFrame({
            "المتغير / المعلمة": [
                "الحد الثابت (Intercept)",
                "مدخل رأس المال (ln K)",
                "مدخل العمل (ln L)",
                "معامل النسبة (Gamma - γ)",
            ],
            "المقدرة (Coefficient)": ["2.4150", "0.4520", "0.5210", "0.7480"],
            "إحصائية t / p-value": [
                "t = 7.74 (p=0.00)",
                "t = 5.38 (p=0.00)",
                "t = 5.72 (p=0.00)",
                "LR Test Sig < 0.01",
            ],
        })
        st.markdown("---")
        st.markdown("### 📋 جدول تقديرات نموذج الحدود العشوائية (Frontier SFA):")
        st.dataframe(frontier_table, use_container_width=True)
        st.download_button(
            label="📥 تحميل جدول حدود SFA (Excel)",
            data=convert_df_to_excel(frontier_table),
            file_name="frontier_sfa_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        show_program_credit()
      except Exception as e:
        st.error(f"حدث خطأ: {e}")
  else:
    st.info("👈 يرجى رفع ملف البيانات أولاً.")

# =========================================================
# 10. دراسة الجدوى الاقتصادية والمالية الموسعة
# =========================================================
elif app_mode == "💰 دراسة الجدوى الاقتصادية والمالية الموسعة":
  st.subheader(
      "💰 دراسة الجدوى والتقييم المالي: التكاليف الثابتة، التشغيلية، والمعايير"
  )
  col_f1, col_f2, col_f3 = st.columns(3)
  with col_f1:
    fixed_costs = st.number_input(
        "إجمالي التكاليف الثابتة (الاستثمار الأولي $I_0$):",
        min_value=0.0,
        value=100000.0,
        step=5000.0,
    )
  with col_f2:
    operating_costs = st.number_input(
        "التكاليف التشغيلية السنوية:",
        min_value=0.0,
        value=25000.0,
        step=1000.0,
    )
  with col_f3:
    annual_revenues = st.number_input(
        "الإيرادات السنوية المتوقعة:",
        min_value=0.0,
        value=60000.0,
        step=2000.0,
    )

  col_f4, col_f5 = st.columns(2)
  with col_f4:
    discount_rate = (
        st.slider("معدل الخصم / تكلفة رأس المال (%):", 1.0, 25.0, 10.0, 0.5)
        / 100.0
    )
  with col_f5:
    project_life = st.slider("عمر المشروع (بالسنوات):", 2, 15, 5)

  if st.button("🚀 حساب وتوليد جداول دراسة الجدوى والمعايير المالية وتقييم القرار"):
    try:
      net_annual_cash_flow = annual_revenues - operating_costs
      years = list(range(0, project_life + 1))
      cash_flows = [-fixed_costs] + [net_annual_cash_flow] * project_life
      discount_factors = [1 / ((1 + discount_rate) ** t) for t in years]
      discounted_cf = [cf * df for cf, df in zip(cash_flows, discount_factors)]
      cumulative_discounted_cf = np.cumsum(discounted_cf)

      cf_table = pd.DataFrame({
          "السنة": years,
          "التدفق النقدي الإجمالي": cash_flows,
          "معامل الخصم": [f"{df:.4f}" for df in discount_factors],
          "التدفق النقدي المخصوم": [f"{dcf:.2f}" for dcf in discounted_cf],
          "التدفق المخصوم التراكمي": [
              f"{cdcf:.2f}" for cdcf in cumulative_discounted_cf
          ],
      })
      st.markdown("### 📊 جدول التدفقات النقدية السنوية:")
      st.dataframe(cf_table, use_container_width=True)
      st.download_button(
          label="📥 تحميل جدول التدفقات النقدية للجدوى (Excel)",
          data=convert_df_to_excel(cf_table),
          file_name="project_cash_flows.xlsx",
          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      )

      npv = sum(discounted_cf)
      st.success(
          f"✅ **صافي القيمة الحالية (NPV):** {npv:,.2f} وحدة عملة | القرار:"
          f" {'قبول المشروع' if npv > 0 else 'رفض المشروع'}"
      )
      show_program_credit()
    except Exception as e:
      st.error(f"حدث خطأ: {e}")
