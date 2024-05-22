import streamlit as st

def homepage():
    st.title("Welcome to My Multi-Page Streamlit App")
    st.write("This is the homepage of the app.")
    st.write("Navigate through the pages using the sidebar.")

def about_page():
    st.title("About")
    st.write("This is the about page of the app.")
    st.write("Here you can learn more about the app and its purpose.")

def contact_page():
    st.title("Contact")
    st.write("This is the contact page of the app.")
    st.write("Feel free to reach out to us if you have any questions or feedback.")

# Create a dictionary to map page names to their corresponding functions
pages = {
    "Home": homepage,
    "About": about_page,
    "Contact": contact_page
}

# Sidebar navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", list(pages.keys()))

# Execute the selected page function
pages[selected_page]()
