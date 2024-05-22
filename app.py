# # import streamlit as st
# # import base64

# # def main():
# #     st.title("PDF Display Example")

# #     # Allow user to upload a PDF file
# #     uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

# #     if uploaded_file is not None:
# #         st.write("Here is the PDF:")
# #         pdf_contents = uploaded_file.read()
# #         encoded_pdf = base64.b64encode(pdf_contents).decode("utf-8")
# #         st.markdown(f'<embed src="data:application/pdf;base64,{encoded_pdf}" width="800" height="600"></embed>', unsafe_allow_html=True)

# # if __name__ == "__main__":
# #     main()


# # import streamlit as st
# # import base64

# # def main():
# #     st.title("PDF Display Example")

# #     # Allow user to upload a PDF file in the middle of the webpage
# #     st.write("Upload PDF:")
# #     uploaded_file = st.file_uploader("", type="pdf")

# #     # Display PDF in the sidebar
# #     st.sidebar.header("Uploaded PDF")
# #     if uploaded_file is not None:
# #         pdf_contents = uploaded_file.read()
# #         encoded_pdf = base64.b64encode(pdf_contents).decode("utf-8")
# #         st.sidebar.markdown(f'<embed src="data:application/pdf;base64,{encoded_pdf}" width="100%" height="600"></embed>', unsafe_allow_html=True)

# # if __name__ == "__main__":
# #     main()

# import streamlit as st
# import base64

# def main():
#     st.title("PDF Display Example")

#     # Allow user to upload multiple PDF files
#     st.write("Upload PDF:")
#     uploaded_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

#     # Display PDFs in the sidebar
#     st.sidebar.header("Uploaded PDFs")
#     for uploaded_file in uploaded_files:
#         if uploaded_file is not None:
#             pdf_contents = uploaded_file.read()
#             encoded_pdf = base64.b64encode(pdf_contents).decode("utf-8")
#             st.sidebar.markdown(f'<embed src="data:application/pdf;base64,{encoded_pdf}" width="100%" height="600"></embed>', unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()
