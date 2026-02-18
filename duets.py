# app.py
from __future__ import annotations

import re
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import streamlit as st


SAMPLES_DIR = Path(__file__).parent / "Samples"
AUDIO_EXTS = (".mp3",)


def render_brand_header(logo_width_px: int = 200):
    left, middle, right = st.columns([1, 1, 1], vertical_alignment="center")

    with left:
        logo_path = Path(__file__).with_name("logo.png")
        logo_alt_path = Path(__file__).with_name("logo_alt.png")

        if logo_path.exists():
            light_uri = _img_to_data_uri(logo_path)
            dark_uri = _img_to_data_uri(logo_alt_path) if logo_alt_path.exists() else light_uri

            html = textwrap.dedent("""
                <style>
                  .dw-logo-light {{ display: inline-block; }}
                  .dw-logo-dark  {{ display: none; }}

                  @media (prefers-color-scheme: dark) {{
                    .dw-logo-light {{ display: none; }}
                    .dw-logo-dark  {{ display: inline-block; }}
                  }}
                </style>

                <img class="dw-logo-light" src="{light_uri}" width="{w}" />
                <img class="dw-logo-dark"  src="{dark_uri}"  width="{w}" />
            """).format(light_uri=light_uri, dark_uri=dark_uri, w=int(logo_width_px))

            st.markdown(html, unsafe_allow_html=True)

    with right:
        st.markdown('Created by David Winter  \n("The Narrator")  \nhttps://www.thenarrator.co.uk')

    st.markdown("---")

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _split_pair_from_and_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse: "<female> and <male>.mp3" (splits on the LAST " and ").
    Returns (female_name, male_name) or None if not parseable.
    """
    token = " and "
    if token not in filename:
        return None

    if filename.lower().endswith(".mp3"):
        stem = filename[:-4]
    else:
        stem = filename

    left, right = stem.rsplit(token, 1)
    female = _clean_spaces(left)
    male = _clean_spaces(right)

    if not female or not male:
        return None

    return female, male


def _build_with_filename(pov: str, additional: str) -> str:
    return f"{pov} with {additional}.mp3"


def _pair_base_filename(pov: str, add: str, pairs: Set[Tuple[str, str]]) -> Optional[str]:
    """
    Returns canonical base filename: "<female> and <male>.mp3" for the selected narrators,
    regardless of dropdown order, if the pair exists.
    """
    for f, m in pairs:
        if (pov == f and add == m) or (pov == m and add == f):
            return f"{f} and {m}.mp3"
    return None


def _valid_additional_labels_for_pov(
    pov_value: str,
    pairs: Set[Tuple[str, str]],
    female_labels: List[str],
    male_labels: List[str],
    label_to_value: Dict[str, str],
) -> List[str]:
    """
    If POV is female, valid additional narrators are the paired males.
    If POV is male, valid additional narrators are the paired females.
    Returns display labels, alphabetised.
    """
    females = {label_to_value[lbl] for lbl in female_labels}
    males = {label_to_value[lbl] for lbl in male_labels}

    valid_adds: Set[str] = set()

    if pov_value in females:
        for f, m in pairs:
            if f == pov_value:
                valid_adds.add(m)
        add_labels = [lbl for lbl in male_labels if label_to_value[lbl] in valid_adds]
    elif pov_value in males:
        for f, m in pairs:
            if m == pov_value:
                valid_adds.add(f)
        add_labels = [lbl for lbl in female_labels if label_to_value[lbl] in valid_adds]
    else:
        add_labels = []

    return sorted(add_labels, key=lambda s: s.lower())


def main() -> None:
    st.set_page_config(page_title="Duet Samples", layout="centered")
    render_brand_header(logo_width_px=200)
    st.title("Duet Samples")

    if not SAMPLES_DIR.exists():
        st.error(f"Folder not found: {SAMPLES_DIR}")
        return

    # Scan ./Samples for MP3s
    files = [
        p for p in SAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]
    by_name: Dict[str, Path] = {p.name: p for p in files}

    # Discover narrator pairs from files containing " and "
    pairs: Set[Tuple[str, str]] = set()
    for p in files:
        parsed = _split_pair_from_and_filename(p.name)
        if parsed:
            pairs.add(parsed)

    if not pairs:
        st.warning('No files found in ./Samples containing " and " in the filename.')
        return

    females = sorted({f for f, _ in pairs}, key=lambda s: s.lower())
    males = sorted({m for _, m in pairs}, key=lambda s: s.lower())

    # Display labels: prefix with (F)/(M) for dropdown only
    female_labels = [f"(F) {n}" for n in females]
    male_labels = [f"(M) {n}" for n in males]
    all_labels = sorted(female_labels + male_labels, key=lambda s: s.lower())

    # Map display label -> raw narrator name
    label_to_value: Dict[str, str] = {lbl: lbl[4:] for lbl in all_labels}  # strips "(X) "

    # --- Safe session state pattern (widget keys separate from authoritative keys) ---
    if "pov_label" not in st.session_state:
        st.session_state.pov_label = all_labels[0]
    if "add_label" not in st.session_state:
        # Try to pick a valid counterpart if possible
        pov_value = label_to_value[st.session_state.pov_label]
        valid_adds = _valid_additional_labels_for_pov(
            pov_value, pairs, female_labels, male_labels, label_to_value
        )
        st.session_state.add_label = valid_adds[0] if valid_adds else all_labels[0]

    if "pov_widget" not in st.session_state:
        st.session_state.pov_widget = st.session_state.pov_label
    if "add_widget" not in st.session_state:
        st.session_state.add_widget = st.session_state.add_label

    def _sync_from_widgets() -> None:
        st.session_state.pov_label = st.session_state.pov_widget
        st.session_state.add_label = st.session_state.add_widget

    def _do_swap() -> None:
        st.session_state.pov_label, st.session_state.add_label = (
            st.session_state.add_label,
            st.session_state.pov_label,
        )
        st.session_state.pov_widget = st.session_state.pov_label
        st.session_state.add_widget = st.session_state.add_label

    # Layout
    col1, col2, col3 = st.columns([1, 1, 0.45], vertical_alignment="bottom")

    with col1:
        st.selectbox(
            "POV Narrator",
            options=all_labels,
            key="pov_widget",
            on_change=_sync_from_widgets,
        )

    pov_value = label_to_value[st.session_state.pov_label]

    # Filter Additional Narrator choices to only valid pairings for current POV (when possible)
    add_options = _valid_additional_labels_for_pov(
        pov_value, pairs, female_labels, male_labels, label_to_value
    )
    if not add_options:
        add_options = all_labels

    # Ensure widget value remains in options
    if st.session_state.add_widget not in add_options:
        st.session_state.add_widget = add_options[0]
        _sync_from_widgets()

    with col2:
        st.selectbox(
            "Additional Narrator",
            options=add_options,
            key="add_widget",
            on_change=_sync_from_widgets,
        )

    with col3:
        st.button("Swap", on_click=_do_swap)

    pov = label_to_value[st.session_state.pov_label]
    add = label_to_value[st.session_state.add_label]

    # Determine which file to play: "<POV> with <Additional>.mp3"
    pov_filename = _build_with_filename(pov, add)
    pov_path = by_name.get(pov_filename)

    if pov_path is None:
        st.error("This combination is not valid.")
        st.caption(f'Expected file: "{pov_filename}"')
        return

    # Media player
    st.audio(str(pov_path), format="audio/mpeg")

    # Download buttons
    d1, d2 = st.columns([1, 1])

    with d1:
        with open(pov_path, "rb") as f:
            st.download_button(
                label="Download this audio",
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
                    label="Download both PoVs",
                    data=f,
                    file_name=base_filename,
                    mime="audio/mpeg",
                )


if __name__ == "__main__":
    main()
