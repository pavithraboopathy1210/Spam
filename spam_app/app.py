import streamlit as st
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📩 Spam Message Detector")
st.write("Enter a message to check if it is Spam or Ham")

user_input = st.text_area("Type your message here")

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter a message")
    else:
        data = cv.transform([user_input])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ HAM (Normal Message)")
 #streamlit run app.py
