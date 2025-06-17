import streamlit as st
import openai
from PyPDF2 import PdfReader

# Hardcoded API key (לשם דוגמה בלבד – לא מומלץ בפרודקשן)
openai.api_key = "sk-proj-..."

SYSTEM_PROMPT = """
אתה Assistant לגיוס ובחינת מועמדים בינלאומי, בעל מומחיות ב-DevOps/SRE/Cloud.
המטרה שלך:
  1. לפרק את קורות החיים לאלמנטים: ניסיון תעסוקתי, טכנולוגיות, הישגים כמותיים, פרויקטים מרכזיים.
  2. לזהות “Red Flags” – חזרות מופרזות, חפיפות תאריכים, buzzwords ללא עומק, ניסוחים שיווקיים מופרזים.
  3. לחלץ הישגים ממשיים: פישוט תהליכים, קיצוץ זמנים, חיסכון עלויות עם מספרים (אחוזים, דקות, אירועים).
  4. לשאול שאלות אימות עומק שמחזירות קוד/קונפיגורציה/דוגמאות קונקרטיות (YAML, JSON, CLI snippets).
  5. לדרג רמת האמינות של המועמד (1–5) על סמך עומק התשובות והדיוק.
  6. להמליץ על צעדי בדיקה נוספים: מבחנים טכניים, בדיקת repos ציבוריים, פיילוט קצר או היקף תקציבי מתומחר.
"""

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def analyze_cv(cv_text):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"להלן קורות החיים של המועמד:\n\n{cv_text}\n\nאנא בצע את השלבים שהוגדרו במערכת."}
        ]
    )
    return response.choices[0].message.content

def main():
    st.set_page_config(page_title="ניתוח קורות חיים", layout="wide")
    st.title("📝 ניתוח קורות חיים אוטומטי")
    st.write("העלה קובץ PDF או TXT עם קורות החיים ותקבל תמצית עם תובנות ומשוב.")

    uploaded_file = st.file_uploader("בחר קובץ PDF או TXT של קורות החיים", type=["pdf", "txt"])
    if not uploaded_file:
        return

    if uploaded_file.type == "application/pdf":
        cv_text = extract_text_from_pdf(uploaded_file)
    else:
        cv_text = uploaded_file.read().decode("utf-8")

    st.subheader("תוכן קורות החיים שהועלו")
    st.text_area("", cv_text, height=200)

    if st.button("ניתוח קורות חיים"):
        with st.spinner("מבצע ניתוח..."):
            analysis = analyze_cv(cv_text)
        st.subheader("תוצאות הניתוח")
        st.markdown(analysis)

if __name__ == "__main__":
    main()
