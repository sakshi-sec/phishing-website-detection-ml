"""
ML-Based Phishing Website Detection System - Streamlit web application
Author : Sakshi Wagh (Roll No. 67)
Run    : streamlit run app.py      (run train_model.py once before this)
"""
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Phishing Website Detection System", page_icon="🛡️", layout="wide")

# ---------- load artefacts ----------------------------------------------------
@st.cache_resource
def load_artifacts():
    bundle = joblib.load("phishing_model.joblib")
    with open("model_results.json") as f:
        results = json.load(f)
    data = pd.read_csv("phisingData.csv").drop_duplicates().reset_index(drop=True)
    return bundle["model"], bundle["features"], results, data

model, FEATURES, R, data = load_artifacts()

# ---------- feature descriptions (UCI Phishing Websites dataset) --------------
HELP = {
 "having_IP_Address": "Does the URL use an IP address instead of a domain name?",
 "URL_Length": "Length of the URL (short < 54, medium 54-75, long > 75 characters).",
 "Shortining_Service": "Is a URL-shortening service (e.g. bit.ly) used?",
 "having_At_Symbol": "Does the URL contain the '@' symbol?",
 "double_slash_redirecting": "Is '//' used for redirection after the protocol part?",
 "Prefix_Suffix": "Does the domain name contain a '-' (hyphen)?",
 "having_Sub_Domain": "Number of sub-domains (dots) in the domain name.",
 "SSLfinal_State": "HTTPS with a trusted certificate issuer and sufficient age.",
 "Domain_registeration_length": "Registration period of the domain (longer is safer).",
 "Favicon": "Is the favicon loaded from an external domain?",
 "port": "Are non-standard ports open / used?",
 "HTTPS_token": "Does the domain part contain a fake 'https' token?",
 "Request_URL": "Percentage of page objects (images, scripts) loaded from other domains.",
 "URL_of_Anchor": "Percentage of anchor (<a>) tags pointing to other domains.",
 "Links_in_tags": "Percentage of links in <Meta>, <Script> and <Link> tags to other domains.",
 "SFH": "Server Form Handler: blank / about:blank / different domain.",
 "Submitting_to_email": "Does the form submit data directly to an e-mail address?",
 "Abnormal_URL": "Is the host name missing from the page's URL record (WHOIS)?",
 "Redirect": "Number of times the website is redirected.",
 "on_mouseover": "Does onMouseOver change the status bar text?",
 "RightClick": "Is the right-click menu disabled?",
 "popUpWidnow": "Does a pop-up window ask for text input?",
 "Iframe": "Is an invisible iframe used?",
 "age_of_domain": "Age of the domain (older than 6 months is safer).",
 "DNSRecord": "Is a DNS record available for the domain?",
 "web_traffic": "Traffic rank of the website.",
 "Page_Rank": "PageRank value of the website.",
 "Google_Index": "Is the website indexed by Google?",
 "Links_pointing_to_page": "Number of external links pointing to the page.",
 "Statistical_report": "Is the host listed in phishing statistics reports?",
}
LABEL = {1: "Legitimate (1)", 0: "Suspicious (0)", -1: "Phishing (-1)"}

# ---------- sample records taken from the held-out TEST set -------------------
from sklearn.model_selection import train_test_split

def _test_rows():
    X = data.drop(columns="Result")
    y = (data["Result"] == -1).astype(int)
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    return X_te, y_te

def sample(label_value):
    X_te, y_te = _test_rows()
    want = 1 if label_value == -1 else 0
    rows = X_te[y_te == want]
    probs = model.predict_proba(rows)[:, 1]
    target = 0.90 if want == 1 else 0.10            # a typical (not extreme) confidence
    ok = (probs >= 0.5) if want == 1 else (probs < 0.5)
    idx = (abs(probs - target) + (~ok) * 10).argmin()
    return rows.iloc[idx].to_dict()

SAMPLES = {"Phishing": sample(-1), "Legitimate": sample(1)}

# ---------- header -------------------------------------------------------------
st.header("🛡️ ML-Based Phishing Website Detection System")
st.caption("Enter the 30 website characteristics below. The trained Random Forest model "
           "classifies the website as phishing or legitimate and shows its confidence.")

qp = st.query_params
if "sample" in qp and "loaded" not in st.session_state:
    s = qp["sample"].capitalize()
    if s in SAMPLES:
        for k, v in SAMPLES[s].items():
            st.session_state[k] = int(v)
    st.session_state["loaded"] = True

with st.sidebar:
    st.header("Quick start")
    st.write("Load a real record from the dataset, or enter values manually.")
    if st.button("Load sample phishing website", use_container_width=True):
        for k, v in SAMPLES["Phishing"].items(): st.session_state[k] = int(v)
    if st.button("Load sample legitimate website", use_container_width=True):
        for k, v in SAMPLES["Legitimate"].items(): st.session_state[k] = int(v)
    st.divider()
    st.subheader("About the model")
    st.write(f"**Algorithm:** {R['best']}")
    st.write(f"**Test accuracy:** {R['res'][R['best']]['acc']*100:.2f}%")
    st.write(f"**5-fold CV accuracy:** {R['cv_mean']*100:.2f}%")
    st.write(f"**Training records:** {R['n_train']:,}")

tab1, tab2, tab3 = st.tabs(["🔍 Detect Website", "📊 Data Analysis", "🏆 Model Comparison"])

# ---------- tab 1 : detection --------------------------------------------------
with tab1:
    st.subheader("Website features")
    cols = st.columns(3)
    values = {}
    for i, f in enumerate(FEATURES):
        opts = sorted(data[f].unique().tolist(), reverse=True)
        if f not in st.session_state:
            st.session_state[f] = 1 if 1 in opts else opts[0]
        with cols[i % 3]:
            values[f] = st.selectbox(f, opts, format_func=lambda v: LABEL[v], key=f, help=HELP[f])
    run = st.button("Check Website", type="primary", use_container_width=True) or qp.get("run") == "1"

    if run:
        X = pd.DataFrame([values])[FEATURES]
        p_phish = float(model.predict_proba(X)[0][1])
        st.divider()
        st.subheader("Prediction result")
        if p_phish >= 0.5:
            st.error("⚠️ PHISHING WEBSITE DETECTED — do not enter any personal or banking details.")
        else:
            st.success("✅ LEGITIMATE WEBSITE — no strong phishing indicators were found.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Phishing probability", f"{p_phish*100:.1f}%")
        m2.metric("Legitimate probability", f"{(1-p_phish)*100:.1f}%")
        risk = "High" if p_phish >= 0.8 else "Medium" if p_phish >= 0.5 else "Low"
        m3.metric("Risk level", risk)
        imp = pd.Series(R["imp"])
        flagged = [(f, imp[f]) for f in FEATURES if values[f] == -1]
        flagged.sort(key=lambda t: -t[1])
        st.markdown("**Top phishing indicators found in the entered values** (ordered by model importance)")
        if flagged:
            st.table(pd.DataFrame([{"Feature": f, "Importance": f"{w:.3f}", "Meaning": HELP[f]}
                                   for f, w in flagged[:5]]).set_index("Feature"))
        else:
            st.write("None of the entered features is marked as a phishing indicator.")

# ---------- tab 2 : data analysis ----------------------------------------------
with tab2:
    st.subheader("Dataset and feature analysis")
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        bars = ax.bar(["Legitimate", "Phishing"], [R["n_legit"], R["n_phish"]], color=["#2E7D32", "#C62828"])
        ax.bar_label(bars); ax.set_ylabel("Number of websites"); ax.set_title("Class distribution (after removing duplicates)")
        st.pyplot(fig)
    with b:
        imp = pd.Series(R["imp"]).sort_values().tail(10)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.barh(imp.index, imp.values, color="#1F4E79"); ax.set_xlabel("Importance"); ax.set_title("Top 10 features (Random Forest)")
        plt.tight_layout(); st.pyplot(fig)
    cm = R["res"][R["best"]]["cm"]
    c, d = st.columns(2)
    with c:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i][j] > 300 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Legitimate", "Phishing"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Legitimate", "Phishing"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix - {R['best']}")
        st.pyplot(fig)
    with d:
        st.write("**Dataset summary**")
        st.table(pd.DataFrame({"Value": [f"{R['n_total']:,}", "30", f"{R['n_train']:,}", f"{R['n_test']:,}"]},
                              index=["Records (after de-duplication)", "Input features", "Training records", "Testing records"]))

# ---------- tab 3 : model comparison -------------------------------------------
with tab3:
    st.subheader("Comparison of machine learning models (test set)")
    df = pd.DataFrame(R["res"]).T[["acc", "prec", "rec", "f1", "auc"]]
    df.columns = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    st.dataframe((df * 100).round(2).style.format("{:.2f}"), use_container_width=True)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    order = df.sort_values("Accuracy")
    ax.barh(order.index, order["Accuracy"] * 100, color=["#1F4E79" if n == R["best"] else "#9DB7D1" for n in order.index])
    ax.set_xlim(50, 100); ax.set_xlabel("Accuracy (%)"); ax.set_title("Model accuracy comparison")
    for i, v in enumerate(order["Accuracy"] * 100): ax.text(v + 0.3, i, f"{v:.2f}", va="center")
    plt.tight_layout(); st.pyplot(fig)
