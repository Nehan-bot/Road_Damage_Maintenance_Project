RoadSight AI Streamlit redesign
==============================

Package contents:
- app/streamlit_app.py: redesigned app source
- assets/roadsight_logo.png: app logo
- assets/road_safety_hero.jpg: simple illustrated landing-page hero

Install:
1. Back up your current app/streamlit_app.py.
2. Copy this package's app/streamlit_app.py into your project's app/ folder.
3. Copy the assets/ folder into your project root (next to app/, data/, src/, etc.).
4. Keep your existing model files and database/db_manager.py unchanged.
5. Activate your existing .venv and run:
   python -m streamlit run app/streamlit_app.py

The app retains your current model paths and database API calls. Test inference and database save/history behavior locally before committing.
