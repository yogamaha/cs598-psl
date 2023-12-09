import streamlit as st

def main():
    st.title("Welcome to My Streamlit App")
    st.write("This is a simple example of a Streamlit app.")

    name = st.text_input("Enter your name:")
    st.write("Hello,", name)

    number = st.slider("Select a number:", 1, 10)
    st.write("You selected:", number)

    if st.button("Click me"):
        st.write("Button clicked!")

if __name__ == "__main__":
    main()
