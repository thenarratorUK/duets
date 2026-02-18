# app.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import streamlit as st

SAMPLES_DIR = Path(__file__).parent / "Samples"
AUDIO_EXTS = (".mp3",)  # keep simple; add more if needed

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _split_pair_from_and_filename(filename: str) -> Optional[Tuple[str, str]]:
    # Parse: "<female> and <male>.mp3" using the last " and "
    token = " and "
    if token not in filename:
        return None
    stem = filename[:-4] if filename.lower().endswith(".mp3") else filename
    left, right = stem.rsplit(token, 1)
    female = _clean_spaces(left)
    male = _clean_spaces(right)
    if not female or not male:
        return None
    return female, male

def _build_with_filename(pov: str, additional: str) -> str:
    return f"{pov} with {additional}.mp3"

def _pair_base_filename(pov: str, add: str, pairs: Set[Tuple[str, str]]) -> Optional[str]:
    for f, m in pairs:
        if (pov == f and add == m) or (pov == m and add == f):
            return f"{f} and {m}.mp3"
    return None

def main() -> None:
    st.set_page_config(page_title="Duet Samples", layout="centered")
    st.title("Duet Samples")

    if not SAMPLES_DIR.exists():
        st.error(f'Folder not found: {SAMPLES_DIR}')
        return

    # List audio files in ./Samples that contain " and " (for pair discovery)
    files = [p for p in SAMPLES_DIR.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    by_name: Dict[str, Path] = {p.name: p for p in files}

    pairs: Set[Tuple[str, str]] = set()
    for p in files:
        parsed = _split_pair_from_and_filename(p.name)
        if parsed:
            pairs.add(parsed)

    if not pairs:
        st.warning('No files found in ./Samples containing " and " in the filename.')
        return

    females = sorted({f for f, _ in pairs}, key=str.lower)
    males = sorted({m for _, m in pairs}, key=str.lower)

    female_labels = [f"(F) {n}" for n in females]
    male_labels = [f"(M) {n}" for n in males]
    all_labels = sorted(female_labels + male_labels, key=str.lower)

    label_to_value = {lbl: lbl[4:] for lbl in all_labels}  # strips "(X) "

    if "pov_label" not in st.session_state:
        st.session_state.pov_label = all_labels[0]
    if "add_label" not in st.session_state:
        st.session_state.add_label = all_labels[0]

    col1, col2, col3 = st.columns([1, 1, 0.45], vertical_alignment="bottom")
    with col1:
        st.selectbox("POV Narrator", options=all_labels, key="pov_label")
    with col2:
        st.selectbox("Additional Narrator", options=all_labels, key="add_label")
    with col3:
        if st.button("Swap"):
            st.session_state.pov_label, st.session_state.add_label = (
                st.session_state.add_label,
                st.session_state.pov_label,
            )
            st.rerun()

    pov = label_to_value[st.session_state.pov_label]
    add = label_to_value[st.session_state.add_label]

    pov_filename = _build_with_filename(pov, add)
    pov_path = by_name.get(pov_filename)

    if pov_path is None:
        st.error("This combination is not valid.")
        st.caption(f'Expected file: "{pov_filename}"')
        return

    st.audio(str(pov_path), format="audio/mpeg")

    d1, d2 = st.columns([1, 1])

    with d1:
        with open(pov_path, "rb") as f:
            st.download_button(
                "Download this audio",
                data=f,
                file_name=pov_filename,
                mime="audio/mpeg",
            )

    with d2:
        base_filename = _pair_base_filename(pov, add, pairs)
        if not base_filename or base_filename not in by_name:
            st.download_button("Download both PoVs", data=b"", file_name="", disabled=True)
            st.caption('No matching "Female and Male" file was found for this pairing.')
        else:
            base_path = by_name[base_filename]
            with open(base_path, "rb") as f:
                st.download_button(
                    "Download both PoVs",
                    data=f,
                    file_name=base_filename,
                    mime="audio/mpeg",
                )

if __name__ == "__main__":
    main()
