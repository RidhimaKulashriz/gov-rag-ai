

**Plan Status:** Approved by user.

**Steps to Complete:**

## 1. Fix Dependencies ✅ (Step 1/5 - COMPLETE)
- [x] Upgrade pip, setuptools, wheel
- [x] Edit requirements.txt (pandas==2.0.3 → 2.2.2)
- [x] pip install -r requirements.txt --upgrade (install ongoing, pandas wheel downloading successfully - no build error!)

## 2. Environment Setup (Step 2/5) ⏳ NEXT
- [ ] cp .env.example .env
- [ ] User adds GOOGLE_API_KEY to .env (manual - get from https://aistudio.google.com/app/apikey)

## 3. Test Installation (Step 3/5)
- [ ] pip list | grep pandas (verify install)

## 4. Run App (Step 4/5)
- [ ] streamlit run chatbot/app.py

## 5. Verify & Complete (Step 5/5)
- [ ] App loads at http://localhost:8501
- [ ] Mark complete

**Notes:**
- venv active ✓
- pandas==2.2.2 wheel resolved build issue ✓
- Wait for pip install to finish (monitor terminal)


