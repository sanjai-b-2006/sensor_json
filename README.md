# AI Water Quality Analyzer 💧

A modern web application powered by Machine Learning to predict water safety.

## 🚀 How to Host (Detailed Guide)

To avoid long loading times (the "slow" problem with Stlite), I highly recommend **Option A**.

### Option A: Streamlit Community Cloud (FASTEST - Recommended)
This is the standard way to host Streamlit apps. It's free and very fast because it runs on actual servers.

1.  **Push to GitHub**: Upload only `app.py` and `requirements.txt` to a GitHub repository.
2.  **Sign in to Streamlit**: Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3.  **Deploy**: 
    - Click "New app".
    - Select your repository, branch, and `app.py` as the main file path.
    - Click **Deploy!**
4.  **Result**: Your site will be live at a custom URL (e.g., `yourapp.streamlit.app`) and will load instantly.

### Option B: GitHub Pages (Serverless)
Use this if you want the site to be hosted on `your-username.github.io`. The first load is slow because it downloads Python into the browser.

1.  **Upload Files**: Upload `index.html`, `app.py`, and `style.css` to your GitHub repo.
2.  **Enable Pages**:
    - Go to **Settings** > **Pages**.
    - Set the branch to `main`.
    - Click **Save**.
3.  **Wait**: It takes about 1-2 minutes for GitHub to build the page.

---

## 🧠 Model & Data Integration
The app is currently configured to fetch:
- **JSON Data**: [sensor_data.json](https://raw.githubusercontent.com/sanjai-b-2006/sensor_json/main/sensor_data.json)
- **ML Model**: [edge_water_quality_model.pkl](https://raw.githubusercontent.com/sanjai-b-2006/sensor_json/main/edge_water_quality_model.pkl)

To update the source, change the `JSON_URL` and `MODEL_URL` constants at the top of `app.py`.

## 📦 Requirements
If running locally or on Streamlit Cloud, you need:
```text
streamlit
pandas
plotly
scikit-learn
joblib
```
