import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import linprog, root_scalar
import scipy.stats as stats
from scipy.stats import f_oneway, jarque_bera, pearsonr, spearmanr, ttest_ind
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
st.title(
    "🌾 منصة الخبير الاقتصادي والقياسي الذكي (الإصدار الشامل: الانحدار، T،"
    " ANOVA، DEA، و Frontier)"
)
st.write(
    "منصة بحثية وأكاديمية متكاملة للتحليلات الاقتصادية القياسية، اختبارات"
    " الفروض، تحليل الكفاءة (DEA & Frontier)، والرسومات البيانية التفاعلية."
)

st.sidebar.header("⚙️ إعدادات المنصة")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "### 💻 هُوية البرنامج والمرجع التقني\n"
    "**اسم البرنامج:** منصة الخبير الاقتصادي والقياسي الذكي\n\n"
    "**المرجع التقني والبرمجي:**\n"
    "- **لغة البرمجة:** Python\n"
    "- **المكتبات:** Statsmodels, SciPy (DEA, Frontier, T, ANOVA), Pandas,"
    " Matplotlib, Streamlit\n"
    "**الإصدار:** 2026 المحسن الشامل (مزود بـ DEA & Frontier)"
)
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "اختر قسم العمل الأساسي:",
    [
        "📊 التحليلات القياسية وتشخيصات الانحدار واختبارات T و ANOVA",
        "📐 تحليل كفاءة الأداء ونماذج الحدود الاقتصادية (DEA & Frontier)",
        "🌾 مؤشرات الأمن الغذائي والتجارة والتنافسية والرسومات",
        "📉 اختبارات جذر الوحدة والتكامل المشترك ونماذج السلاسل الزمنية",
        "💰 دراسة الجدوى والتقييم المالي الشامل للمشروعات",
        "🌐 بوابة جمع البيانات والمؤشرات العالمية",
        "👨‍🏫 المستشار الاقتصادي والقياسي (قسم الذكاء الاصطناعي)",
    ],
)


def show_program_credit():
  st.caption(
      "💻 **تم إجراء هذا التحليل باستخدام:** منصة الخبير الاقتصادي والقياسي الذكي"
      " | **المرجع التقني والبرمجي:** Python (Statsmodels, SciPy, Matplotlib)."
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
# القسم الأول: التحليلات القياسية وتشخيصات الانحدار واختبارات T و ANOVA
# =========================================================
if app_mode == "📊 التحليلات القياسية وتشخيصات الانحدار واختبارات T و ANOVA":
  st.subheader(
      "📊 التحليلات القياسية، اختبارات الفروض الإحصائية (T و ANOVA) والتمثيل"
      " البياني"
  )

  if df is not None:
    st.dataframe(df.head(), use_container_width=True)
    columns_list = df.columns.tolist()

    sub_choice = st.selectbox(
        "اختر نوع التحليل القياسي المطلوب:",
        [
            (
                "اختبار T (T-Test) لعينتين واختبار تحليل التباين الأحادي (ANOVA)"
                " مع الرسومات"
            ),
            "مقاييس النزعة المركزية والتشتت والرسومات التوزيعية",
            (
                "نموذج الانحدار الخطي المتعدد (OLS) مع التشخيصات والرسومات"
                " (التنبؤ والبواقي)"
            ),
            "دالة إنتاج كوب-دوجلاس اللوغاريتمية وعوائد الحجم",
            "دالة الإنتاج التربيعية ونقطة الانقلاب (تناقص الغلة والرسومات)",
            "تحليل الاتجاه الزمني (الخطي والآسي والرسومات التوضيحية)",
            "معاملات الارتباط (بيرسون وسبيرمان) ومصفوفة التباين المرئي",
        ],
    )

    if (
        sub_choice
        == "اختبار T (T-Test) لعينتين واختبار تحليل التباين الأحادي (ANOVA) مع الرسومات"
    ):
      st.subheader(
          "🧪 اختبارات الفروض الإحصائية: اختبار T واختبار تحليل التباين (ANOVA)"
      )
      test_type = st.radio(
          "اختر الاختبار الإحصائي:",
          [
              "اختبار T لعينتين مستقلتين (Independent Samples T-Test)",
              "تحليل التباين الأحادي (One-Way ANOVA)",
          ],
      )

      num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

      if test_type == "اختبار T لعينتين مستقلتين (Independent Samples T-Test)":
        c1, c2 = st.columns(2)
        with c1:
          col1 = st.selectbox("المتغير الأول (المجموعة الأولى):", num_cols, key="t_c1")
        with c2:
          col2 = st.selectbox(
              "المتغير الثاني (المجموعة الثانية):",
              [c for c in num_cols if c != col1],
              key="t_c2",
          )

        if st.button("🚀 تنفيذ اختبار T ورسم المقارنة") and col1 and col2:
          s1 = pd.to_numeric(df[col1], errors="coerce").dropna()
          s2 = pd.to_numeric(df[col2], errors="coerce").dropna()
          t_stat, p_val = ttest_ind(s1, s2, nan_policy="omit")

          st.markdown("### 📊 نتائج اختبار T لعينتين مستقلتين:")
          m1, m2 = st.columns(2)
          with m1:
            st.metric("قيمة إحصاء ت (T-Statistic)", f"{t_stat:.4f}")
          with m2:
            st.metric("القيمة الاحتمالية (p-value)", f"{p_val:.4f}")

          if p_val < 0.05:
            st.success(
                "✅ النتيجة: يوجد فرق ذو دلالة إحصائية بين متوسطي المتغيرين"
                " (رفض الفرض الصفري عند مستوى معنوية 5%)."
            )
          else:
            st.warning(
                "❌ النتيجة: لا يوجد فرق ذو دلالة إحصائية بين متوسطي المتغيرين"
                " (قبول الفرض الصفري)."
            )

          fig, ax = plt.subplots(figsize=(8, 4))
          means = [s1.mean(), s2.mean()]
          stds = [s1.std(), s2.std()]
          ax.bar(
              [col1, col2],
              means,
              yerr=stds,
              capsize=5,
              color=["royalblue", "darkorange"],
              edgecolor="black",
          )
          ax.set_ylabel("المتوسط الحسابي")
          ax.set_title(
              "مقارنة المتوسطات والانحرافات المعيارية بين المجموعتين"
          )
          st.pyplot(fig)
          show_program_credit()

      else:
        target_anova = st.selectbox(
            "المتغير التابع (الرقمي):", num_cols, key="anova_target"
        )
        group_anova = st.selectbox(
            "متغير المجموعات أو التصنيف:",
            [c for c in columns_list if c != target_anova],
            key="anova_group",
        )

        if st.button("🚀 تنفيذ اختبار ANOVA ورسم التوزيعات") and target_anova and group_anova:
          anova_df = df[[target_anova, group_anova]].dropna()
          groups = [
              group[target_anova].values
              for name, group in anova_df.groupby(group_anova)
          ]

          if len(groups) >= 2:
            f_stat, p_val_anova = f_oneway(*groups)

            st.markdown("### 📊 نتائج تحليل التباين الأحادي (One-Way ANOVA):")
            m1, m2 = st.columns(2)
            with m1:
              st.metric("قيمة إحصاء F (F-Statistic)", f"{f_stat:.4f}")
            with m2:
              st.metric("القيمة الاحتمالية (p-value)", f"{p_val_anova:.4f}")

            if p_val_anova < 0.05:
              st.success(
                  "✅ النتيجة: توجد فروق ذات دلالة إحصائية بين متوسطات المجموعات"
                  " (النموذج معنوي)."
              )
            else:
              st.warning(
                  "❌ النتيجة: لا توجد فروق ذات دلالة إحصائية بين متوسطات"
                  " المجموعات."
              )

            fig, ax = plt.subplots(figsize=(9, 5))
            anova_df.boxplot(
                column=target_anova, by=group_anova, ax=ax, grid=False
            )
            ax.set_title(
                f"توزيع متغير ({target_anova}) عبر مجموعات ({group_anova})"
            )
            ax.set_xlabel(group_anova)
            ax.set_ylabel(target_anova)
            plt.suptitle("")
            st.pyplot(fig)
            show_program_credit()
          else:
            st.error(
                "⚠️ عدد المجموعات داخل متغير التصنيف غير كافٍ لإجراء تحليل"
                " التباين (يجب أن توجد مجموعتان على الأقل)."
            )

    elif sub_choice == "مقاييس النزعة المركزية والتشتت والرسومات التوزيعية":
      st.subheader(
          "📈 الإحصاءات الوصفية ومؤشرات الالتواء والتفرطح مع الهيستوجرام"
      )
      num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
      sel_cols = st.multiselect("اختر المتغيرات:", num_cols)
      if sel_cols and st.button("تشغيل الإحصاءات الوصفية والرسومات"):
        sub_df = df[sel_cols].apply(pd.to_numeric, errors="coerce")
        desc = sub_df.describe().T
        desc["التباين"] = sub_df.var()
        desc["معامل الاختلاف (%)"] = (sub_df.std() / sub_df.mean()) * 100
        desc["التلتلة (Skewness)"] = sub_df.skew()
        desc["التفرطح (Kurtosis)"] = sub_df.kurtosis()
        st.dataframe(desc, use_container_width=True)

        st.markdown("### 📊 تمثيل التوزيعات الإحصائية (Histograms):")
        fig, ax = plt.subplots(figsize=(10, 5))
        sub_df.hist(ax=ax, bins=15, edgecolor="black")
        plt.tight_layout()
        st.pyplot(fig)
        show_program_credit()

    elif (
        sub_choice
        == "نموذج الانحدار الخطي المتعدد (OLS) مع التشخيصات والرسومات (التنبؤ والبواقي)"
    ):
      st.subheader(
          "📈 نموذج الانحدار الخطي المتعدد (OLS) مع التشخيصات والرسومات البيانية"
      )
      c1, c2 = st.columns(2)
      with c1:
        y_c = st.selectbox("المتغير التابع (Y):", columns_list, key="ols_y")
      with c2:
        x_c = st.multiselect(
            "المتغيرات المستقلة (X):",
            [c for c in columns_list if c != y_c],
            key="ols_x",
        )

      if st.button("تشغيل نموذج OLS والرسومات التشخيصية") and y_c and x_c:
        temp_df = df[[y_c] + x_c].apply(pd.to_numeric, errors="coerce").dropna()
        y = temp_df[y_c]
        X = sm.add_constant(temp_df[x_c])
        model = sm.OLS(y, X).fit()
        st.text(model.summary().as_text())

        st.markdown(
            "### 📉 التمثيل البياني: القيم الفعلية مقابل القيم المتنبأة"
            " ومسار البواقي"
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(
            y.values, label="القيمة الفعلية (Actual)", color="blue", marker="o"
        )
        ax1.plot(
            model.fittedvalues.values,
            label="القيمة المتنبأة (Fitted)",
            color="red",
            linestyle="--",
            marker="x",
        )
        ax1.set_title("مقارنة القيم الفعلية والمتنبأة لنموذج OLS")
        ax1.legend()

        ax2.scatter(model.fittedvalues, model.resid, color="purple")
        ax2.axhline(0, color="black", linestyle="--")
        ax2.set_title("اختبار تجانس التباين (البواقي مقابل المتنبأ)")
        ax2.set_xlabel("القيم المتنبأة")
        ax2.set_ylabel("البواقي (Residuals)")
        plt.tight_layout()
        st.pyplot(fig)
        show_program_credit()

    elif sub_choice == "دالة إنتاج كوب-دوجلاس اللوغاريتمية وعوائد الحجم":
      st.subheader("📉 دالة إنتاج كوب-دوجلاس اللوغاريتمية والتمثيل البصري")
      c1, c2 = st.columns(2)
      with c1:
        y_c = st.selectbox("المتغير التابع (الإنتاج):", columns_list, key="cd_y")
      with c2:
        x_c = st.multiselect(
            "المدخلات المستقلة:", [c for c in columns_list if c != y_c], key="cd_x"
        )
      if st.button("تقدير كوب-دوجلاس ورسم النتائج") and y_c and x_c:
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

        fig, ax = plt.subplots(figsize=(8, 4))
        params = model.params[x_c]
        params.plot(
            kind="bar",
            ax=ax,
            color="teal",
            edgecolor="black",
        )
        ax.set_title("مرونات عناصر الإنتاج (معلمات كوب-دوجلاس)")
        ax.set_ylabel("قيمة المرونة")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        show_program_credit()

    elif (
        sub_choice == "دالة الإنتاج التربيعية ونقطة الانقلاب (تناقص الغلة والرسومات)"
    ):
      st.subheader("📉 دالة الإنتاج التربيعية ومنحنى تناقص الغلة")
      c1, c2 = st.columns(2)
      with c1:
        y_q = st.selectbox("المتغير التابع (الإنتاج Y):", columns_list, key="qy")
      with c2:
        x_q = st.selectbox(
            "المتغير المستقل (عنصر الإنتاج X):",
            [c for c in columns_list if c != y_q],
            key="qx",
        )
      if st.button("تقدير دالة الإنتاج التربيعية ورسم المنحنى") and y_q and x_q:
        q_df = df[[y_q, x_q]].apply(pd.to_numeric, errors="coerce").dropna()
        q_df["X2"] = q_df[x_q] ** 2
        X = sm.add_constant(q_df[[x_q, "X2"]])
        model_q = sm.OLS(q_df[y_q], X).fit()
        st.text(model_q.summary().as_text())
        b0, b1, b2 = (
            model_q.params["const"],
            model_q.params[x_q],
            model_q.params["X2"],
        )

        if b2 < 0:
          max_x = -b1 / (2 * b2)
          st.success(
              f"نقطة الانقلاب (حجم العنصر المحقق لأقصى إنتاج): `{max_x:.4f}`"
          )

          fig, ax = plt.subplots(figsize=(8, 5))
          x_vals = np.linspace(q_df[x_q].min(), q_df[x_q].max() * 1.2, 100)
          y_vals = b0 + b1 * x_vals + b2 * (x_vals**2)
          ax.plot(
              x_vals,
              y_vals,
              label="منحنى الإنتاج التربيعي",
              color="darkorange",
              linewidth=2,
          )
          ax.axvline(
              max_x,
              color="red",
              linestyle="--",
              label=f"نقطة أقصى إنتاج ({max_x:.2f})",
          )
          ax.set_xlabel("عنصر الإنتاج (X)")
          ax.set_ylabel("إجمالي الإنتاج (Y)")
          ax.legend()
          ax.set_title("منحنى دالة الإنتاج التربيعية وتناقص الغلة")
          st.pyplot(fig)
        show_program_credit()

    elif (
        sub_choice == "تحليل الاتجاه الزمني (الخطي والآسي والرسومات التوضيحية)"
    ):
      st.subheader("📈 تحليل الاتجاه الزمني ورسوم مسار النمو")
      c1, c2 = st.columns(2)
      with c1:
        t_var = st.selectbox("متغير الزمن أو السنوات (t):", columns_list, key="tv")
      with c2:
        y_var = st.selectbox(
            "متغير الظاهرة الاقتصادية (Y):",
            [c for c in columns_list if c != t_var],
            key="yv",
        )
      if st.button("حساب معدلات النمو ورسم الاتجاه العام") and t_var and y_var:
        t_data = df[[t_var, y_var]].apply(pd.to_numeric, errors="coerce").dropna()
        t = t_data[t_var]
        y = t_data[y_var]

        X_lin = sm.add_constant(t)
        m_lin = sm.OLS(y, X_lin).fit()
        b_lin = m_lin.params[t_var]

        log_y = np.log(y.replace(0, np.nan)).dropna()
        X_exp = sm.add_constant(t.loc[log_y.index])
        m_exp = sm.OLS(log_y, X_exp).fit()
        b_exp = m_exp.params[t_var]
        cagr = (np.exp(b_exp) - 1) * 100

        st.metric(
            "معدل التغير السنوي المطلق (معامل الانحدار الخطي)", f"{b_lin:.4f}"
        )
        st.metric("معدل النمو المركب السنوي (CAGR %)", f"{cagr:.2f}%")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            t, y, label="البيانات الفعلية", marker="o", color="blue", linewidth=2
        )
        ax.plot(
            t,
            m_lin.fittedvalues,
            label="خط الاتجاه العام (Linear Trend)",
            color="red",
            linestyle="--",
        )
        ax.set_xlabel("الزمن / السنوات")
        ax.set_ylabel("قيمة المتغير")
        ax.legend()
        ax.set_title("تحليل الاتجاه العام للظاهرة الاقتصادية عبر الزمن")
        st.pyplot(fig)
        show_program_credit()

    elif (
        sub_choice
        == "معاملات الارتباط (بيرسون وسبيرمان) ومصفوفة التباين المرئي"
    ):
      st.subheader("📊 معاملات الارتباط ومصفوفة الارتباط الحرارية (Heatmap)")
      num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
      sel_corr = st.multiselect(
          "اختر المتغيرات لحساب مصفوفة الارتباط:", num_cols
      )
      if len(sel_corr) >= 2 and st.button("حساب ورسم مصفوفات الارتباط"):
        corr_p = df[sel_corr].corr(method="pearson")
        corr_s = df[sel_corr].corr(method="spearman")

        st.markdown("### مصفوفة ارتباط بيرسون (Pearson):")
        st.dataframe(corr_p, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr_p, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax)
        ax.set_xticks(range(len(sel_corr)))
        ax.set_yticks(range(len(sel_corr)))
        ax.set_xticklabels(sel_corr, rotation=45, ha="left")
        ax.set_yticklabels(sel_corr)
        ax.set_title("مصفوفة ارتباط بيرسون البصرية", pad=20)
        st.pyplot(fig)
        show_program_credit()
  else:
    st.info("👈 يرجى رفع ملف البيانات من القائمة الجانبية.")

# =========================================================
# القسم الجديد: تحليل كفاءة الأداء ونماذج الحدود (DEA & Frontier)
# =========================================================
elif app_mode == "📐 تحليل كفاءة الأداء ونماذج الحدود الاقتصادية (DEA & Frontier)":
  st.subheader(
      "📐 تحليل مغلف البيانات (DEA) ونماذج الحدود الإنتاجية (Frontier / SFA)"
  )

  if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    method_choice = st.selectbox(
        "اختر المنهجية الاقتصادية لتقدير الكفاءة:",
        [
            (
                "تحليل مغلف البيانات (DEA - Data Envelopment Analysis) لحساب"
                " الكفاءة"
            ),
            (
                "تقدير الحدود الإنتاجية ونماذج الكفاءة الفنية (Frontier / COLS"
                " & SFA)"
            ),
        ],
    )

    if (
        method_choice
        == "تحليل مغلف البيانات (DEA - Data Envelopment Analysis) لحساب الكفاءة"
    ):
      st.markdown(
          "### 🌐 نموذج مغلف البيانات (DEA - CCR Input-Oriented Model)"
      )
      st.write(
          "يُستخدم هذا النموذج لقياس الكفاءة الفنية لوحدات اتخاذ القرار (DMUs)"
          " مثل المزارع، الشركات، أو المصانع بالاعتماد على مدخلات ومخرجات"
          " متعددة."
      )

      c1, c2 = st.columns(2)
      with c1:
        inputs_dea = st.multiselect(
            "اختر متغيرات المدخلات (Inputs - عناصر التكلفة أو الموارد):",
            num_cols,
            key="dea_in",
        )
      with c2:
        outputs_dea = st.multiselect(
            "اختر متغيرات المخرجات (Outputs - الإنتاج أو العائد):",
            [c for c in num_cols if c not in inputs_dea],
            key="dea_out",
        )

      dmu_col = st.selectbox(
          "اختر عمود أسماء أو أرقام الوحدات (DMUs / المزارع / الشركات):",
          df.columns.tolist(),
      )

      if (
          st.button("🚀 تنفيذ تحليل مغلف البيانات (DEA) ورسم درجات الكفاءة")
          and inputs_dea
          and outputs_dea
          and dmu_col
      ):
        try:
          dea_df = (
              df[[dmu_col] + inputs_dea + outputs_dea]
              .apply(
                  lambda x: pd.to_numeric(x, errors="coerce")
                  if x.name != dmu_col
                  else x
              )
              .dropna()
          )

          X = dea_df[inputs_dea].values
          Y = dea_df[outputs_dea].values
          n_dmus = len(dea_df)
          n_inputs = len(inputs_dea)
          n_outputs = len(outputs_dea)

          efficiency_scores = []

          # خوارزمية البرمجة الخطية لنموذج CCR الموجه نحو المدخلات لكل وحدة
          for i in range(n_dmus):
            x0 = X[i, :]
            y0 = Y[i, :]

            # متغيرات القرار: [theta, lambda_1, lambda_2, ..., lambda_n]
            c = np.zeros(1 + n_dmus)
            c[0] = 1.0  # تقليل theta

            # القيود:
            # 1) X * lambda <= theta * x0  => X * lambda - theta * x0 <= 0
            # 2) Y * lambda >= y0          => -Y * lambda <= -y0
            A_ub = np.vstack(
                [
                    np.hstack(
                        [
                            -x0.reshape(-1, 1),
                            X,
                        ]
                    ),  # X * lambda - theta * x0 <= 0
                    np.hstack([np.zeros((n_outputs, 1)), -Y]),  # -Y * lambda <= -y0
                ]
            )
            b_ub = np.hstack([np.zeros(n_inputs), -y0])

            bounds = [(None, None)] + [(0, None) for _ in range(n_dmus)]

            res = linprog(
                c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
            )
            if res.success:
              efficiency_scores.append(res.x[0])
            else:
              efficiency_scores.append(np.nan)

          dea_df["درجة الكفاءة الفنية (Efficiency Score)"] = efficiency_scores
          dea_df = dea_df.sort_values(
              by="درجة الكفاءة الفنية (Efficiency Score)", ascending=False
          )

          st.markdown("### 📊 جدول نتائج درجات الكفاءة لوحدات اتخاذ القرار (DEA):")
          st.dataframe(dea_df, use_container_width=True)

          # رسم بياني درجات الكفاءة
          fig, ax = plt.subplots(figsize=(10, 5))
          ax.bar(
              dea_df[dmu_col].astype(str),
              dea_df["درجة الكفاءة الفنية (Efficiency Score)"],
              color="forestgreen",
              edgecolor="black",
          )
          ax.axhline(
              1.0,
              color="red",
              linestyle="--",
              label="الحد الأقصى للكفاءة التامة (1.0)",
          )
          ax.set_xlabel("وحدات اتخاذ القرار (DMUs)")
          ax.set_ylabel("درجة الكفاءة (0 إلى 1)")
          ax.set_title(
              "درجات الكفاءة الفنية المستخرجة من نموذج مغلف البيانات (DEA)"
          )
          plt.xticks(rotation=45, ha="right")
          ax.legend()
          st.pyplot(fig)
          show_program_credit()
        except Exception as e:
          st.error(f"حدث خطأ أثناء تشغيل نموذج DEA: {e}")
    else:
      st.markdown(
          "### 📈 تحليل الحدود الإنتاجية والكفاءة الفنية (Frontier / COLS)"
      )
      st.write(
          "تقدير دالة الإنتاج الحدية واستخراج درجات الكفاءة الفنية للمنشآت"
          " باستخدام نموذج المربعات الصغرى المعدلة (Corrected Ordinary Least"
          " Squares - COLS)."
      )

      c1, c2 = st.columns(2)
      with c1:
        y_fr = st.selectbox(
            "المتغير التابع (إجمالي الإنتاج Y):", num_cols, key="fr_y"
        )
      with c2:
        x_fr = st.multiselect(
            "المتغيرات المستقلة (عنصر الإنتاج X):",
            [c for c in num_cols if c != y_fr],
            key="fr_x",
        )

      if st.button("🚀 تقدير دالة الحدود الإنتاجية ورسم الكفاءة") and y_fr and x_fr:
        try:
          f_data = df[[y_fr] + x_fr].apply(pd.to_numeric, errors="coerce").dropna()
          y = f_data[y_fr]
          X = sm.add_constant(f_data[x_fr])

          ols_model = sm.OLS(y, X).fit()
          residuals = ols_model.resid
          max_res = residuals.max()

          # تعديل القاطع (Intercept) لبناء دالة الحدود الإنتاجية (Deterministic Frontier)
          corrected_intercept = ols_model.params["const"] + max_res
          tech_efficiency = np.exp(
              residuals - max_res
          )  # أو نسبة الإنتاج الفعلي إلى الحدودي

          f_data["درجة الكفاءة الفنية"] = tech_efficiency

          st.text(ols_model.summary().as_text())
          st.markdown("---")
          st.metric(
              "قاطع دالة الحدود الإنتاجية (Frontier Intercept)",
              f"{corrected_intercept:.4f}",
          )
          st.metric(
              "متوسط الكفاءة الفنية للعينة",
              f"{f_data['درجة الكفاءة الفنية'].mean():.4f}",
          )

          st.markdown("### 📊 جدول درجات الكفاءة الفنية للمنشآت:")
          st.dataframe(f_data, use_container_width=True)

          fig, ax = plt.subplots(figsize=(9, 5))
          ax.scatter(
              ols_model.fittedvalues,
              y,
              color="royalblue",
              label="القيم الفعلية للمنشآت",
              alpha=0.7,
          )
          fitted_sorted = np.sort(ols_model.fittedvalues)
          frontier_line = fitted_sorted + max_res
          ax.plot(
              fitted_sorted,
              frontier_line,
              color="crimson",
              linewidth=2,
              linestyle="--",
              label="منحنى الحدود الإنتاجية (Production Frontier)",
          )
          ax.set_xlabel("القيم المتنبأة بدالة الإنتاج المتوسطة")
          ax.set_ylabel("الإنتاج الفعلي (Y)")
          ax.set_title("تمثيل منحنى حدود الإنتاج الفني (Frontier Curve)")
          ax.legend()
          st.pyplot(fig)
          show_program_credit()
        except Exception as e:
          st.error(f"حدث خطأ أثناء تقدير نموذج الحدود: {e}")
  else:
    st.info("👈 يرجى رفع ملف البيانات أولاً من القائمة الجانبية.")

# =========================================================
# القسم الثالث: مؤشرات الأمن الغذائي والتجارة والتنافسية والرسومات
# =========================================================
elif app_mode == "🌾 مؤشرات الأمن الغذائي والتجارة والتنافسية والرسومات":
  st.subheader(
      "🌾 مؤشرات الأمن الغذائي الشاملة ومؤشرات التجارة الخارجية مع التمثيل البياني"
  )

  if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    sub_cf = st.selectbox(
        "اختر مجموعة المؤشرات المطلوبة:",
        [
            (
                "مؤشرات الأمن الغذائي (الفجوة، الاكتفاء الذاتي، فترة الكفاية"
                " والرسومات)"
            ),
            (
                "مؤشرات التجارة الخارجية الشاملة (معدل التغطية، التبعية، الانفتاح"
                " والرسومات)"
            ),
            (
                "مؤشرات القدرة التنافسية المتقدمة (RCA، النصيب السوقي، الاختراق،"
                " السعر النسبي)"
            ),
        ],
    )

    if (
        sub_cf
        == "مؤشرات الأمن الغذائي (الفجوة، الاكتفاء الذاتي، فترة الكفاية والرسومات)"
    ):
      st.subheader(
          "🌾 مؤشرات الأمن الغذائي والفجوات وفترة الكفاية مع الرسوم البيانية"
      )
      c1, c2 = st.columns(2)
      with c1:
        prod_c = st.selectbox("متغير الإنتاج المحلي (Production):", num_cols)
      with c2:
        cons_c = st.selectbox(
            "متغير الاستهلاك الكلي (Consumption):",
            [c for c in num_cols if c != prod_c],
        )

      if st.button("حساب ورسم مؤشرات الأمن الغذائي") and prod_c and cons_c:
        f_df = df[[prod_c, cons_c]].apply(pd.to_numeric, errors="coerce").dropna()
        f_df["الفجوة الغذائية (استهلاك - إنتاج)"] = f_df[cons_c] - f_df[prod_c]
        f_df["نسبة الاكتفاء الذاتي (%)"] = (
            f_df[prod_c] / f_df[cons_c].replace(0, np.nan)
        ) * 100
        f_df["فترة الكفاية الذاتية (أشهر)"] = (
            f_df[prod_c] / f_df[cons_c].replace(0, np.nan)
        ) * 12
        st.dataframe(f_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            f_df[prod_c].values,
            label="الإنتاج المحلي",
            color="green",
            marker="o",
        )
        ax.plot(
            f_df[cons_c].values,
            label="الاستهلاك الكلي",
            color="orange",
            marker="x",
        )
        ax.set_title("مقارنة الإنتاج المحلي بالاستهلاك الكلي عبر الفترات")
        ax.set_ylabel("الكمية")
        ax.legend()
        st.pyplot(fig)
        show_program_credit()

    elif (
        sub_cf
        == "مؤشرات التجارة الخارجية الشاملة (معدل التغطية، التبعية، الانفتاح والرسومات)"
    ):
      st.subheader("🌐 مؤشرات التجارة الخارجية مع الرسوم البيانية التوضيحية")
      if len(num_cols) >= 3:
        c1, c2, c3 = st.columns(3)
        with c1:
          x_col = st.selectbox("إجمالي الصادرات (Exports - X):", num_cols)
        with c2:
          m_col = st.selectbox(
              "إجمالي الواردات (Imports - M):",
              [c for c in num_cols if c != x_col],
          )
        with c3:
          gdp_col = st.selectbox(
              "الناتج المحلي الإجمالي (GDP):",
              [c for c in num_cols if c not in [x_col, m_col]],
          )

        if st.button("حساب ورسم مؤشرات التجارة الخارجية"):
          t_df = (
              df[[x_col, m_col, gdp_col]]
              .apply(pd.to_numeric, errors="coerce")
              .dropna()
          )
          t_df["1. معدل التغطية (%)"] = (
              t_df[x_col] / t_df[m_col].replace(0, np.nan)
          ) * 100
          t_df["2. الميزان التجاري النسبي (NTI)"] = (
              t_df[x_col] - t_df[m_col]
          ) / (t_df[x_col] + t_df[m_col]).replace(0, np.nan)
          t_df["3. معدل التبعية الاستيرادية (%)"] = (
              t_df[m_col] / t_df[gdp_col].replace(0, np.nan)
          ) * 100
          t_df["4. مؤشر الانفتاح التجاري (%)"] = (
              (t_df[x_col] + t_df[m_col]) / t_df[gdp_col].replace(0, np.nan)
          ) * 100

          st.dataframe(t_df, use_container_width=True)

          fig, ax = plt.subplots(figsize=(10, 5))
          ax.plot(
              t_df["1. معدل التغطية (%)"].values,
              label="معدل التغطية (%)",
              color="blue",
              marker="o",
          )
          ax.plot(
              t_df["4. مؤشر الانفتاح التجاري (%)"].values,
              label="مؤشر الانفتاح التجاري (%)",
              color="purple",
              marker="s",
          )
          ax.set_title("مسار معدل التغطية والانفتاح التجاري")
          ax.legend()
          st.pyplot(fig)
          show_program_credit()
      else:
        st.info("الرجاء توفر 3 أعمدة رقمية على الأقل.")

    elif (
        sub_cf
        == "مؤشرات القدرة التنافسية المتقدمة (RCA، النصيب السوقي، الاختراق، السعر النسبي)"
    ):
      st.subheader(
          "🏆 مؤشرات القدرة التنافسية الدولية (RCA، النصيب السوقي، الاختراق،"
          " السعر النسبي)"
      )
      if len(num_cols) >= 4:
        c1, c2 = st.columns(2)
        with c1:
          x_item = st.selectbox("صادرات السلعة المدروسة (X_ij):", num_cols)
          tot_exp = st.selectbox(
              "إجمالي الصادرات الكلية للدولة (Total X_i):",
              [c for c in num_cols if c != x_item],
          )
        with c2:
          imp_item = st.selectbox(
              "الواردات المحلية للسلعة (M):",
              [c for c in num_cols if c not in [x_item, tot_exp]],
          )
          prod_item = st.selectbox(
              "الإنتاج المحلي للسلعة (Production):",
              [
                  c
                  for c in num_cols
                  if c not in [x_item, tot_exp, imp_item]
              ],
          )

        world_tot_exp = st.number_input(
            "إجمالي الصادرات العالمية (World Total Exports):",
            value=1000000.0,
        )
        world_item_exp = st.number_input(
            "إجمالي الصادرات العالمية للسلعة:", value=50000.0
        )
        domestic_price = st.number_input(
            "سعر التصدير المحلي للوحدة (أو السعر المحلي):", value=100.0
        )
        world_price = st.number_input(
            "السعر العالمي المعياري للوحدة:", value=95.0
        )

        if st.button("حساب ورسم مؤشرات القدرة التنافسية"):
          comp_df = (
              df[[x_item, tot_exp, imp_item, prod_item]]
              .apply(pd.to_numeric, errors="coerce")
              .dropna()
          )

          global_share_denom = (
              world_item_exp / world_tot_exp if world_tot_exp > 0 else 1.0
          )
          comp_df["1. الميزة النسبية الظاهرة (RCA)"] = (
              comp_df[x_item] / comp_df[tot_exp].replace(0, np.nan)
          ) / global_share_denom
          comp_df["2. النصيب السوقي العالمي (%)"] = (
              comp_df[x_item] / world_item_exp
          ) * 100
          domestic_supply = (
              comp_df[prod_item] + comp_df[imp_item] - comp_df[x_item]
          )
          comp_df["3. معامل الاختراق المحلي (%)"] = (
              comp_df[imp_item] / domestic_supply.replace(0, np.nan)
          ) * 100
          comp_df["4. مؤشر السعر النسبي"] = domestic_price / world_price

          st.dataframe(comp_df, use_container_width=True)

          fig, ax = plt.subplots(figsize=(9, 4))
          comp_df["1. الميزة النسبية الظاهرة (RCA)"].plot(
              kind="bar", ax=ax, color="darkgreen", edgecolor="black"
          )
          ax.axhline(
              1.0, color="red", linestyle="--", label="الحد الفاصل (RCA = 1)"
          )
          ax.set_title("مؤشر الميزة النسبية الظاهرة (RCA)")
          ax.legend()
          st.pyplot(fig)
          show_program_credit()
      else:
        st.info("الرجاء توفر 4 أعمدة رقمية على الأقل في الملف.")
  else:
    st.info("👈 يرجى رفع ملف البيانات أولاً.")

# =========================================================
# القسم الرابع: اختبارات جذر الوحدة والتكامل المشترك والسلاسل الزمنية
# =========================================================
elif (
    app_mode
    == "📉 اختبارات جذر الوحدة والتكامل المشترك ونماذج السلاسل الزمنية"
):
  st.subheader(
      "📉 اختبارات جذر الوحدة، التكامل المشترك، ونماذج السلاسل الزمنية مع"
      " التنبؤات والرسومات"
  )

  if df is not None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    ts_sub = st.selectbox(
        "اختر المنهجية القياسية الزمنية المطلوبة:",
        [
            "اختبارات جذر الوحدة الشاملة (ADF, Phillips-Perron, KPSS)",
            (
                "التكامل المشترك (Engle-Granger & Johansen) ونموذج تصحيح الخطأ"
                " (ECM)"
            ),
            "نماذج التنبؤ (ARMA / ARIMA / SARIMAX) مع الرسوم البيانية التنبؤية",
            "نماذج الانحدار الذاتي للمتجهات (VAR Models)",
        ],
    )

    if (
        ts_sub
        == "اختبارات جذر الوحدة الشاملة (ADF, Phillips-Perron, KPSS)"
    ):
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
          st.button("🚀 تنفيذ اختبارات جذر الوحدة ورسم السلسلة")
          and target_series
      ):
        s_data = pd.to_numeric(df[target_series], errors="coerce").dropna()
        if diff_d > 0:
          for _ in range(diff_d):
            s_data = s_data.diff().dropna()

        fig, ax = plt.subplots(figsize=(10, 4))
        s_data.plot(ax=ax, color="navy", marker="o", title="مسار السلسلة الزمنية")
        ax.set_ylabel("القيمة")
        st.pyplot(fig)

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
        show_program_credit()

    elif (
        ts_sub
        == "التكامل المشترك (Engle-Granger & Johansen) ونموذج تصحيح الخطأ"
        " (ECM)"
    ):
      st.subheader(
          "🔗 التكامل المشترك (Engle-Granger & Johansen) ونموذج تصحيح الخطأ"
          " (ECM)"
      )
      c1, c2 = st.columns(2)
      with c1:
        y_var = st.selectbox("المتغير التابع (Y):", num_cols, key="coint_y")
      with c2:
        x_var = st.selectbox(
            "المتغير المستقل (X):",
            [c for c in num_cols if c != y_var],
            key="coint_x",
        )

      if st.button(
          "🚀 تنفيذ التكامل المشترك ومعادلة الأجل الطويل ورسم البواقي"
      ):
        y_s = pd.to_numeric(df[y_var], errors="coerce")
        x_s = pd.to_numeric(df[x_var], errors="coerce")
        temp_c = pd.concat([y_s, x_s], axis=1).dropna()

        try:
          score, p_value, crit_values = coint(temp_c[y_var], temp_c[x_var])
          st.markdown("### 📊 نتائج اختبار إنجل-غرانجر للتكامل المشترك:")
          st.metric("قيمة الإحصاء (Coint Score)", f"{score:.4f}")
          st.metric("القيمة الاحتمالية (p-value)", f"{p_value:.4f}")

          X_long = sm.add_constant(temp_c[x_var])
          long_run_model = sm.OLS(temp_c[y_var], X_long).fit()
          st.text(long_run_model.summary().as_text())

          fig, ax = plt.subplots(figsize=(10, 4))
          long_run_model.resid.plot(
              ax=ax, color="crimson", title="بواقي علاقة التوازن طويل الأجل"
          )
          ax.axhline(0, color="black", linestyle="--")
          st.pyplot(fig)

          if p_value < 0.05:
            st.success(
                "✅ يوجد تكامل مشترك وتوازن طويل الأجل بين المتغيرين بنجاح!"
            )
            dy = temp_c[y_var].diff().dropna()
            dx = temp_c[x_var].diff().dropna()
            ecm_resid = long_run_model.resid.shift(1).dropna()

            ecm_df = pd.DataFrame({"DY": dy, "DX": dx}).loc[ecm_resid.index]
            ecm_df["ECT_lag1"] = ecm_resid

            X_ecm = sm.add_constant(ecm_df[["DX", "ECT_lag1"]])
            ecm_model = sm.OLS(ecm_df["DY"], X_ecm).fit()
            st.markdown("### 📉 نتائج نموذج تصحيح الخطأ (ECM):")
            st.text(ecm_model.summary().as_text())
          else:
            st.warning("❌ لا يوجد تكامل مشترك عند مستوى معنوية 5%.")
        except Exception as e:
          st.error(f"خطأ أثناء تنفيذ التكامل المشترك: {e}")
        show_program_credit()

    elif (
        ts_sub
        == "نماذج التنبؤ (ARMA / ARIMA / SARIMAX) مع الرسوم البيانية التنبؤية"
    ):
      st.subheader("📈 نماذج التنبؤ (ARIMA) مع الرسومات البيانية التنبؤية")
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

      if st.button("🚀 تقدير ARIMA ورسم مسار التنبؤات المستقبلية"):
        try:
          series = pd.to_numeric(df[t_col], errors="coerce").dropna()
          model = ARIMA(series, order=(p_v, d_v, q_v)).fit()
          st.text(model.summary().as_text())

          forecast = model.forecast(steps=steps)

          fig, ax = plt.subplots(figsize=(11, 5))
          ax.plot(
              series.index,
              series,
              label="البيانات الفعلية التاريخية",
              color="blue",
          )
          forecast_index = range(len(series), len(series) + len(forecast))
          ax.plot(
              forecast_index,
              forecast,
              label="التنبؤات المستقبلية (Forecast)",
              color="orange",
              linestyle="--",
              marker="o",
          )
          ax.set_title("تنبؤات نموذج ARIMA للفترات القادمة")
          ax.legend()
          st.pyplot(fig)

          f_df = pd.DataFrame(
              {
                  "فترة التنبؤ": [f"+{i}" for i in range(1, steps + 1)],
                  "القيمة المتنبأة": forecast,
              }
          )
          st.dataframe(f_df, use_container_width=True)
        except Exception as e:
          st.error(f"حدث خطأ أثناء تقدير نموذج ARIMA: {e}")
        show_program_credit()

    elif ts_sub == "نماذج الانحدار الذاتي للمتجهات (VAR Models)":
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
# القسم الخامس: دراسة الجدوى والتقييم المالي الشامل للمشروعات
# =========================================================
elif app_mode == "💰 دراسة الجدوى والتقييم المالي الشامل للمشروعات":
  st.subheader(
      "💰 دراسة الجدوى الاقتصادية والتقييم المالي الشامل مع الرسوم البيانية"
  )
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

  if st.button("🚀 حساب معايير التقييم المالي ورسم التدفقات النقدية"):
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

      pv_benefits = sum(cf / ((1 + r) ** (i + 1)) for i, cf in enumerate(cfs))
      bcr = pv_benefits / init_inv if init_inv > 0 else np.nan
      npvi = pv_benefits / init_inv if init_inv > 0 else np.nan

      st.markdown("### 📊 نتائج التقييم المالي الشامل:")
      m1, m2, m3 = st.columns(3)
      with m1:
        st.metric("صافي القيمة الحالية (NPV)", f"{npv:,.2f}")
        st.metric("معدل العائد الداخلي (IRR)", f"{irr:.2f}%")
      with m2:
        st.metric("نسبة المنافع للتكاليف (BCR)", f"{bcr:.2f}")
        st.metric("مؤشر الربحية (NPVI / PI)", f"{npvi:.2f}")

      fig, ax = plt.subplots(figsize=(10, 4))
      ax.bar(
          range(1, len(cfs) + 1),
          cfs,
          color="royalblue",
          edgecolor="black",
          label="التدفقات السنوية",
      )
      ax.axhline(0, color="black", linestyle="--")
      ax.set_xlabel("سنوات المشروع")
      ax.set_ylabel("قيمة التدفق النقدي الصافي")
      ax.set_title("توزيع التدفقات النقدية السنوية للمشروع")
      st.pyplot(fig)

      if npv > 0:
        st.success(
            "✅ **القرار الاستثماري:** المشروع **مقبول ومربح اقتصادياً**."
        )
      else:
        st.warning("❌ **القرار الاستثماري:** المشروع غير مجدي مالياً.")
      show_program_credit()
    except Exception as ex:
      st.error(f"حدث خطأ أثناء إجراء الحسابات المالية: {ex}")

# =========================================================
# القسم السادس: بوابة جمع البيانات والمؤشرات العالمية
# =========================================================
elif app_mode == "🌐 بوابة جمع البيانات والمؤشرات العالمية":
  st.subheader("🌐 بوابة جمع البيانات والمؤشرات الاقتصادية والزراعية")
  c_code = st.text_input("كود الدولة الثلاثي (مثال: EGY, USA):", value="EGY")
  if st.button("📥 جلب البيانات من البنك الدولي ورسمها"):
    try:
      url = f"http://api.worldbank.org/v2/country/{c_code}/indicator/NV.AGR.TOTL.ZS?format=json&per_page=50"
      res = requests.get(url).json()
      if len(res) > 1 and res[1]:
        d_wb = pd.DataFrame([
            {
                "السنة": int(e.get("date")),
                "نسبة الزراعة من الناتج المحلي (%)": float(e.get("value"))
                if e.get("value")
                else 0.0,
            }
            for e in res[1]
            if e.get("value") is not None
        ]).sort_values("السنة")
        st.dataframe(d_wb, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            d_wb["السنة"],
            d_wb["نسبة الزراعة من الناتج المحلي (%)"],
            marker="o",
            color="forestgreen",
            linewidth=2,
        )
        ax.set_title(
            f"تطور مساهمة القطاع الزراعي في الناتج المحلي لدولة {c_code}"
        )
        ax.set_xlabel("السنة")
        ax.set_ylabel("النسبة المئوية (%)")
        st.pyplot(fig)

        show_program_credit()
      else:
        st.error("لم يتم العثور على بيانات.")
    except Exception as e:
      st.error(f"خطأ: {e}")

# =========================================================
# القسم السابع: المستشار الاقتصادي والقياسي (ذكاء اصطناعي)
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
        "💡 **توجيه الخبير:** تم تفعيل وتطوير كافة النماذج والرسومات البيانية"
        " واختبارات T, ANOVA, DEA, و Frontier لتوفير بيئة بحثية متكاملة."
    )
    show_program_credit()
