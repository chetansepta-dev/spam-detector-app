import streamlit as st
import joblib

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="centered")

# 2. Load Model (Using caching so it doesn't reload on every click)
@st.cache_resource
def load_model():
    return joblib.load('spam_model_v1.pkl')

model = load_model()
threshold = 0.50

# 3. Header Section
st.title("🛡️ Email Spam Detector")
st.markdown("Paste an email or text message below to analyze its contents using Machine Learning.")

# 4. Input Area
email = st.text_area("Message Content:", height=150, placeholder="Type or paste your email here...")

# 5. Action Button (Using 'primary' type makes it stand out)
if st.button("Analyze Message 🔍", type="primary", use_container_width=True):
    if email:
        # Loading animation
        with st.spinner("Analyzing patterns..."):
            probabilities = model.predict_proba([email])[0]
            ham_probability = probabilities[0]
            spam_probability = probabilities[1]
            
            st.markdown("---")
            
            # 6. Main Result Alert
            if spam_probability >= threshold:
                st.error("**Threat Detected!** This message is classified as SPAM.")
            else:
                st.success(" **Looks Safe!** This message is classified as HAM.")
            
            # 7. Probability Dashboard using Columns and Metrics
            st.markdown("### Confidence Scores")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="Spam Probability", value=f"{spam_probability * 100:.1f}%")
                st.progress(float(spam_probability)) # Adds a visual bar
                
            with col2:
                st.metric(label="Ham Probability", value=f"{ham_probability * 100:.1f}%")
                st.progress(float(ham_probability)) # Adds a visual bar

    else:
        st.warning("⚠️ Please enter a message to analyze.")

# 8. Footer Info box
with st.expander("ℹ️ How does this work?"):
    st.write("This tool uses a trained Natural Language Processing (NLP) model to evaluate text. If the spam probability exceeds 50%, it is flagged as a threat.")