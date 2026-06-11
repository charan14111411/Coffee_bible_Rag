from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(pdf_path: str):
    """
    Load PDF and return LangChain Documents.
    """

    loader = PyMuPDFLoader(pdf_path)

    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    return documents