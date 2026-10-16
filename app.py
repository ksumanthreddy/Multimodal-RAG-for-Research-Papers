import streamlit as st
import db
import rag

st.set_page_config(page_title="Ask your PDFs", page_icon="📄")
st.title("📄 Ask your PDFs")

# connect to the database
try:
    pdf_list = db.list_pdfs()
except Exception as error:
    st.error(f"Cannot connect to PostgreSQL: {error}")
    st.stop()

# sidebar: show available PDFs and choose where to search
with st.sidebar:
    st.header("Documents")

    if pdf_list:
        for name, pages in pdf_list:
            st.write(f"• {name} ({pages} pages)")

        choice = st.selectbox(
            "Search in",
            ["All PDFs"] + [name for name, _pages in pdf_list]
        )
    else:
        st.warning("No indexed PDFs found.")
        choice = "All PDFs"

question = st.text_input("Your question")

if question:
    if not pdf_list:
        st.warning("No indexed PDFs found. Run ingest.py first.")
    else:
        pdf_filter = None if choice == "All PDFs" else choice

        with st.spinner("Searching and asking Gemini ..."):
            try:
                result = rag.answer_question(question, pdf_filter)

                st.subheader("Answer")
                st.markdown(result["answer"])

                if result["sources"]:
                    st.subheader("Sources")
                    for line in result["sources"]:
                        st.write("• " + line)

                if result["images"]:
                    st.subheader("Original images from the cited pages")
                    for image in result["images"]:
                        st.image(
                            image["path"],
                            caption=f"[{image['label']}] {image['pdf_name']} - page {image['page_no']}"
                        )

                if result["chunks"]:
                    with st.expander("Top chunks from the search"):
                        for chunk in result["chunks"]:
                            st.markdown(
                                f"**{chunk['pdf_name']} - page {chunk['page_no']}**"
                            )
                            st.text(chunk["content"][:600])

            except Exception as error:
                st.error(f"Error while answering your question: {error}")
