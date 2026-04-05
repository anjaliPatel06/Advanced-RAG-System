import os
import re
import pickle
import hashlib
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "rag-index"

if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)


# --- IMPROVED: Extract real page URL from nav links in Crawl4AI docs ---
def extract_page_url(text):
    """Try to extract the canonical URL this page belongs to from its nav breadcrumb."""
    # Pattern: the last "Search" nav link often contains the current page URL
    match = re.search(r'\[ Search \]\((https://[^\)]+)\)', text)
    if match:
        return match.group(1)
    # Fallback: first https URL found
    match = re.search(r'https://docs\.crawl4ai\.com/[^\s\)\"\']+', text)
    if match:
        return match.group(0)
    return None


def clean_text(text):
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        # Remove navigation links like "* [Home](https://...)"
        if line.startswith("* [") and "](" in line:
            continue

        # Remove bare URLs
        if line.startswith("http"):
            continue

        # Remove stray markdown symbols
        if line in ["*", "-", "×", "---", "***", "* * *"]:
            continue

        # Remove very short lines (noise)
        if len(line) < 5:
            continue

        # Remove lines that are only special characters / punctuation
        if re.match(r'^[\W_]+$', line):
            continue

        # Remove repeated header nav lines
        if line.count("#") > 3 and len(line) < 60:
            continue

        # IMPROVED: Remove table-of-contents anchor lines like "  * [Section](#section)"
        if re.match(r'\s*\*\s*\[.+\]\(#.+\)', line):
            continue

        cleaned.append(line)

    # Remove duplicate consecutive lines
    unique = list(dict.fromkeys(cleaned))
    return "\n".join(unique)


# Load documents
loader = DirectoryLoader(
    "data/scraped",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documents = loader.load()
print("Total documents loaded:", len(documents))

# IMPROVED: Attach real page URL as metadata before cleaning
for doc in documents:
    page_url = extract_page_url(doc.page_content)
    doc.page_content = clean_text(doc.page_content)
    raw_source = doc.metadata.get("source", "unknown")
    doc.metadata["source"] = raw_source
    doc.metadata["page_url"] = page_url or raw_source
    doc.metadata["type"] = "document"

for doc in documents[:2]:
    print(doc.page_content[:200])
    print("URL:", doc.metadata.get("page_url"))
    print("-----")

headers = [
    ("#", "title"),
    ("##", "section"),
    ("###", "subsection"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

semantic_chunks = []

for doc in documents:
    splits = markdown_splitter.split_text(doc.page_content)
    for split in splits:
        split.metadata["source"] = doc.metadata.get("source", "unknown")
        split.metadata["page_url"] = doc.metadata.get("page_url", "unknown")
        split.metadata["type"] = "document"
    semantic_chunks.extend(splits)

small_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=120
)

if len(semantic_chunks) == 0:
    print("Fallback triggered")
    for doc in documents:
        semantic_chunks.extend(small_splitter.split_documents([doc]))

# IMPROVED: Deduplication by content hash to avoid duplicate chunks across files
seen_hashes = set()
final_chunks = []

for chunk in semantic_chunks:
    sub_chunks = small_splitter.split_text(chunk.page_content)

    for sub in sub_chunks:
        sub = sub.strip()
        if len(sub) < 60:
            continue

        # Hash-based deduplication
        content_hash = hashlib.md5(sub.encode()).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        final_chunks.append(
            Document(
                page_content=sub,
                metadata=chunk.metadata
            )
        )

print("Final unique chunks:", len(final_chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

stats = index.describe_index_stats()

if stats.get("total_vector_count", 0) == 0:
    # IMPROVED: Upload in batches to avoid API timeouts
    batch_size = 100
    for i in range(0, len(final_chunks), batch_size):
        batch = final_chunks[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"Uploaded batch {i // batch_size + 1} / {len(final_chunks) // batch_size + 1}")
    print("Data inserted into Pinecone")
else:
    print("Data already exists, skipping insert")

with open("chunks.pkl", "wb") as f:
    pickle.dump(final_chunks, f)

print("chunks.pkl saved.")