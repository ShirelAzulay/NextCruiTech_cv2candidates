# nct – NextCruitech Candidates App

An interactive system for analyzing candidates, built with **Streamlit** and designed to run within a virtual environment (venv).  
The UI supports resume parsing, prompt-based interactions, and AI-powered modules.

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/ShirelAzulay/NextCruiTech_cv2candidates.git
cd NextCruiTech_cv2candidates
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
```

### 3. Activate the environment
```bash
source venv/bin/activate
```

> On Windows:
```cmd
venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

> If `streamlit` is missing from `requirements.txt`, install it manually:
```bash
pip install streamlit
```

---

## ▶️ Running the App
```bash
streamlit run ui/app.py
```

---

## 🧪 Environment Check (optional)

To verify your runtime setup:
```bash
which python
which streamlit
streamlit --version
```

---

## 🗂 Project Structure

```
.
├── ui/
│   ├── app.py              ← Main Streamlit app
│   ├── app_back.py         ← Backup version
│   ├── app_insights.py     ← Insights and analytics logic
│   ├── logo.png            ← App logo
│   └── prompt.txt          ← Prompt file for LLM
├── requirements.txt        ← Python dependencies
├── run.py                  ← Optional runner script
└── README.md               ← Project documentation
```

---

## ⚠️ Notes

- Requires Python 3.8+
- Future features may include OCR, resume parsing, and AI integrations (e.g., OpenAI/Gemini APIs)
- Make sure you’re running inside the virtual environment before starting the app.

---

## 📬 Contact

For questions or collaboration:  
[ShirelAzulay on GitHub](https://github.com/ShirelAzulay)
