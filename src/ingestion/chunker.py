from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []
    
    for doc in documents:
        page_content = doc.page_content
        page_chunks = splitter.split_documents([doc])
        
        for chunk in page_chunks:
            chunk_content = chunk.page_content
            start_idx = page_content.find(chunk_content)
            
            if start_idx != -1:
                end_idx = start_idx + len(chunk_content)
                start_line = page_content[:start_idx].count("\n") + 1
                end_line = page_content[:end_idx].count("\n") + 1
            else:
                # Fallback if character indices cannot be matched
                start_line = 1
                end_line = page_content.count("\n") + 1
                
            chunk.metadata["start_line"] = start_line
            chunk.metadata["end_line"] = end_line
            chunks.append(chunk)

    print(f"Created {len(chunks)} chunks with line bound metadata")

    return chunks