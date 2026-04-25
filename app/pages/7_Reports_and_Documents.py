import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import streamlit as st

from components import file_utils as fu
from components.sidebar import render_sidebar_stats
from components.styling import inject_css, section_title

inject_css()
render_sidebar_stats()

section_title("Reports & Documents")
root = fu.get_project_root()

st.markdown("### README")
readme = root / "README.md"
if readme.is_file():
    st.download_button("Download README.md", data=readme.read_bytes(), file_name="README.md", mime="text/markdown")
    with st.expander("Preview README"):
        st.markdown(
            readme.read_text(encoding="utf-8", errors="replace")[:8000]
            + ("\n\n…" if readme.stat().st_size > 8000 else "")
        )
else:
    st.warning("README.md not found.")

st.markdown("### Manuscript / write-up (Markdown)")
report_dir = fu.report_docs_dir()
md_files = sorted(report_dir.glob("*.md")) if report_dir.is_dir() else []
for p in md_files:
    with st.expander(f"📄 {p.name}"):
        st.download_button(f"Download {p.name}", data=p.read_bytes(), file_name=p.name, key=f"dl_{p.name}")
        st.markdown(p.read_text(encoding="utf-8", errors="replace")[:6000] + ("…" if p.stat().st_size > 6000 else ""))

st.markdown("### PDFs (project root, manuscript/, Lit_Review/)")
all_pdfs = []
for folder in [root, fu.report_docs_dir(), fu.lit_review_dir()]:
    if folder.is_dir():
        all_pdfs.extend(fu.list_pdfs(folder))
seen = set()
for p in sorted(all_pdfs, key=lambda x: str(x)):
    key = str(p.resolve())
    if key in seen:
        continue
    seen.add(key)
    col1, col2 = st.columns([3, 1])
    col1.markdown(f"**{p.name}**  \n`{p.parent.name}/`")
    col2.download_button("Download", data=p.read_bytes(), file_name=p.name, key=f"pdf_{hash(key)}")

st.markdown("### Ref / other")
ref = fu.ref_dir()
if ref.is_dir():
    for p in sorted(ref.iterdir()):
        if p.is_file():
            st.download_button(f"Download {p.name}", data=p.read_bytes(), file_name=p.name, key=f"ref_{p.name}")
