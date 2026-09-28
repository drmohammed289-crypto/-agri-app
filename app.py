# =========================================================
# القسم الثالث: المستشار الاقتصادي والبحث المباشر (بدون مفتاح API)
# =========================================================
elif (
    app_mode
    == "👨‍🏫 المستشار الاقتصادي والقياسي (قسم الذكاء الاصطناعي المتخصص)"
):
  st.subheader(
      "🏛️ غرفة الخبير الأكاديمي والبحث المباشر (اقتصاد زراعي وقياسي)"
  )
  st.markdown(
      "هذا القسم يقوم بالبحث المباشر والفوري في شبكة الإنترنت لجلب أحدث"
      " الدراسات، البيانات، والأدبيات العلمية **بدون أي قيود أو مفاتيح API**."
  )

  from duckduckgo_search import DDGS

  # تهيئة سجل المحادثة
  if "expert_messages" not in st.session_state:
    st.session_state.expert_messages = [{
        "role": "assistant",
        "content": (
            "أهلاً بك يا فندم. أنا جاهز للبحث المباشر عن أي موضوع اقتصادي أو"
            " زراعي وجلب المصادر والمراجع ذات الصلة فوراً."
        ),
    }]

  for message in st.session_state.expert_messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  prompt_text = st.chat_input("اكتب موضوع البحث أو استفسارك هنا...")

  if prompt_text:
    st.session_state.expert_messages.append(
        {"role": "user", "content": prompt_text}
    )
    with st.chat_message("user"):
      st.markdown(prompt_text)

    with st.chat_message("assistant"):
      with st.spinner("جاري البحث المباشر في الويب وجلب المصادر والمراجع..."):
        response_content = ""
        try:
          # تنفيذ البحث المباشر بدون مفتاح
          with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    f"agricultural economics research {prompt_text}",
                    max_results=6,
                )
            )

            if results:
              response_content = (
                  "### 📚 النتائج والمراجع البحثية المسترجعة مباشرة:\n\n"
              )
              for idx, r in enumerate(results, 1):
                title = r.get("title", "بدون عنوان")
                body = r.get("body", "")
                href = r.get("href", "#")
                response_content += (
                    f"**{idx}. [{title}]({href})**\n- {body}\n\n"
                )
            else:
              response_content = (
                  "⚠️ لم يتم العثور على نتائج مطابقة مباشرة. جرب تعديل كلمات"
                  " البحث."
              )
        except Exception as e:
          response_content = f"⚠️ حدث خطأ أثناء تنفيذ البحث المباشر: {e}"

        st.markdown(response_content)
        st.session_state.expert_messages.append(
            {"role": "assistant", "content": response_content}
        )
