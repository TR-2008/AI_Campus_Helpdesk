import io
import re
import streamlit as st
from pypdf import PdfReader
from docx import Document
from common import gemini_text, extract_json, safe_error

st.set_page_config(page_title="AI Campus Helpdesk", page_icon="🎓", layout="wide")
st.title("🎓 AI Campus Helpdesk")
st.caption("Ask questions about campus policies using the institution's uploaded documents.")

def read_upload(upload):
    name = upload.name.lower()
    raw = upload.getvalue()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for i, page in enumerate(reader.pages):
            pages.append(f"[Source: {upload.name}, page {i+1}]\n{page.extract_text() or ''}")
        return "\n\n".join(pages)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(raw))
        return f"[Source: {upload.name}]\n" + "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if name.endswith(".txt"):
        return f"[Source: {upload.name}]\n" + raw.decode("utf-8", errors="replace")
    raise ValueError("Unsupported file type.")

def chunks(text, size=1100, overlap=150):
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end])
        if end == len(text): break
        start = max(end - overlap, start + 1)
    return out

with st.sidebar:
    st.header("Institutional knowledge base")
    uploads = st.file_uploader("Upload official documents", type=["pdf","docx","txt"], accept_multiple_files=True)
    st.caption("Upload current, approved policies, fee schedules, admission guides, and timetables.")
    if st.button("Process documents", type="primary", disabled=not uploads):
        try:
            all_chunks = []
            for file in uploads:
                content = read_upload(file)
                if len(content.strip()) < 30:
                    st.warning(f"Little or no extractable text in {file.name}. Scanned PDFs need OCR before upload.")
                    continue
                all_chunks.extend(chunks(content))
            st.session_state.kb = all_chunks
            st.session_state.kb_names = [f.name for f in uploads]
            st.success(f"Processed {len(all_chunks)} text sections.")
        except Exception as e:
            st.error(f"Document processing failed: {safe_error(e)}")

if "kb" not in st.session_state:
    st.session_state.kb = []

if st.session_state.kb:
    st.success(f"Knowledge base ready: {len(st.session_state.kb)} sections from {len(st.session_state.get('kb_names', []))} files.")
else:
    st.warning("Upload and process institutional documents in the sidebar before asking questions.")

question = st.text_input("Your question", placeholder="What is the last date to submit the admission fee?")
if st.button("Ask helpdesk", type="primary", disabled=not question.strip() or not st.session_state.kb):
    # Lightweight retrieval: rank chunks by query-term overlap; no external vector database required.
    terms = {t.lower() for t in re.findall(r"\w+", question) if len(t) > 2}
    ranked = []
    for idx, chunk in enumerate(st.session_state.kb):
        words = set(t.lower() for t in re.findall(r"\w+", chunk))
        score = len(terms & words)
        if score:
            ranked.append((score, idx, chunk))
    ranked.sort(reverse=True)
    selected = ranked[:6]
    if not selected:
        st.warning("I could not find a relevant passage in the uploaded documents. Try different wording or upload the relevant policy.")
    else:
        context = "\n\n---\n\n".join(f"[Passage {n+1}]\n{item[2]}" for n, item in enumerate(selected))
        prompt = f"""
Answer the student's question using ONLY the supplied excerpts.
If the excerpts do not contain the answer, say clearly that the uploaded documents do not specify it.
Do not infer dates, fees, eligibility, or policy details. Keep answer concise.
Return valid JSON only:
{{"answer":"...", "citations":[{{"source":"exact source filename/page text if present", "quote":"short supporting quote"}}], "confidence":"high|medium|low"}}
QUESTION: {question}
EXCERPTS:
{context[:12000]}
"""
        try:
            with st.spinner("Searching institutional documents and preparing an answer..."):
                result = extract_json(gemini_text(prompt,
                    "You are a cautious university helpdesk assistant. Never invent policy. Cite only supplied evidence."))
            st.subheader("Answer")
            st.write(result.get("answer", "No answer returned."))
            st.caption(f"Evidence confidence: {result.get('confidence', 'not provided')}")
            cites = result.get("citations", [])
            if cites:
                st.subheader("Sources")
                for c in cites:
                    st.markdown(f"- **{c.get('source','Uploaded document')}**")
                    if c.get("quote"): st.caption(f"“{c['quote']}”")
            else:
                st.warning("No source citation was returned; verify this answer against the original documents.")
        except Exception as e:
            st.error(f"Could not answer the question. {safe_error(e)}")

st.caption("Important: This is an aid, not an official university notice. Confirm deadlines and payments with the relevant office.")
