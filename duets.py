# app.py
# Streamlit app: reads /Samples from a GitHub repo, discovers narrator pairs from filenames containing " and ",
# then lets you select POV/Additional narrators and play/download the matching audio.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import requests
import streamlit as st


SAMPLES_DIR = "Samples"
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


@dataclass(frozen=True)
class GitHubFile:
    name: str
    download_url: str


def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strip_ext(filename: str) -> str:
    for ext in AUDIO_EXTS:
        if filename.lower().endswith(ext):
            return filename[: -len(ext)]
    return filename


def _split_pair_from_and_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse: "<female> and <male>.mp3" (only using the last occurrence of " and ").
    Returns (female_name, male_name) or None if not parseable.
    """
    stem = _strip_ext(filename)
    # Require the token " and " to reduce accidental splits.
    token = " and "
    if token not in stem:
        return None
    left, right = stem.rsplit(token, 1)
    female = _clean_spaces(left)
    male = _clean_spaces(right)
    if not female or not male:
        return None
    return female, male


def _build_with_filename(pov: str, additional: str) -> str:
    # File naming rule for playback/download-this-audio button.
    # Example: "Mina Fairlow with David Winter (British).mp3"
    return f"{pov} with {additional}.mp3"


@st.cache_data(show_spinner=False, ttl=300)
def list_github_samples(
    owner: str,
    repo: str,
    branch: str,
    token: str,
) -> List[GitHubFile]:
    """
    Lists files in /Samples via GitHub Contents API.
    Requires repo to be public or a valid token for private repos.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{SAMPLES_DIR}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.get(url, headers=headers, params={"ref": branch}, timeout=30)
    if resp.status_code == 404:
        raise RuntimeError(
            f"Could not find repo/folder: {owner}/{repo} (branch {branch}) /{SAMPLES_DIR}"
        )
    if resp.status_code == 401 or resp.status_code == 403:
        raise RuntimeError(
            f"GitHub API access failed (HTTP {resp.status_code}). "
            f"Repo may be private, token may be missing/invalid, or rate-limited."
        )
    resp.raise_for_status()

    items = resp.json()
    files: List[GitHubFile] = []
    for it in items:
        if it.get("type") != "file":
            continue
        name = it.get("name", "")
        if not name.lower().endswith(AUDIO_EXTS):
            continue
        download_url = it.get("download_url") or ""
        if not download_url:
            continue
        files.append(GitHubFile(name=name, download_url=download_url))
    return files


def download_bytes(url: str, token: str) -> bytes:
    headers = {}
    if token:
        # download_url is raw.githubusercontent.com; Authorization may not be needed for public,
        # but works for many private cases when GitHub serves the file.
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.content


def main() -> None:
    st.set_page_config(page_title="Duet Samples", layout="centered")

    st.title("Duet Samples")

    with st.sidebar:
        st.header("GitHub Source")
        owner = st.text_input("Owner", value="")
        repo = st.text_input("Repo", value="")
        branch = st.text_input("Branch", value="main")
        token = st.text_input("GitHub Token (optional)", value="", type="password")
        st.caption(f"Folder scanned: /{SAMPLES_DIR}")

    if not owner or not repo or not branch:
        st.info("Enter Owner, Repo, and Branch in the sidebar.")
        return

    try:
        gh_files = list_github_samples(owner=owner, repo=repo, branch=branch, token=token)
    except Exception as e:
        st.error(str(e))
        return

    # Index by filename for quick lookup
    by_name: Dict[str, GitHubFile] = {f.name: f for f in gh_files}

    # Discover narrator pairs from files containing " and "
    pairs: Set[Tuple[str, str]] = set()
    for f in gh_files:
        parsed = _split_pair_from_and_filename(f.name)
        if parsed:
            pairs.add(parsed)

    if not pairs:
        st.warning('No files found in /Samples containing " and " in the filename.')
        return

    females = sorted({f for f, _ in pairs}, key=lambda s: s.lower())
    males = sorted({m for _, m in pairs}, key=lambda s: s.lower())

    # Narrator label mapping for dropdown display
    # (F)/(M) prefix for display only; underlying value remains the clean narrator name.
    label_to_value: Dict[str, str] = {}
    female_labels: List[str] = []
    for name in females:
        label = f"(F) {name}"
        label_to_value[label] = name
        female_labels.append(label)

    male_labels: List[str] = []
    for name in males:
        label = f"(M) {name}"
        label_to_value[label] = name
        male_labels.append(label)

    all_labels = sorted(female_labels + male_labels, key=lambda s: s.lower())

    # Session state for selections
    if "pov_label" not in st.session_state:
        st.session_state.pov_label = all_labels[0]
    if "add_label" not in st.session_state:
        # Prefer a valid counterpart if possible
        pov_value = label_to_value[st.session_state.pov_label]
        suggested_adds = _suggest_additional_labels(pov_value, pairs, female_labels, male_labels, label_to_value)
        st.session_state.add_label = suggested_adds[0] if suggested_adds else all_labels[0]

    col1, col2, col3 = st.columns([1, 1, 0.45], vertical_alignment="bottom")

    with col1:
        pov_label = st.selectbox(
            "POV Narrator",
            options=all_labels,
            key="pov_label",
        )

    pov_value = label_to_value[pov_label]

    # Filter Additional Narrator options to those that are valid counterparts for the current POV (if any exist).
    add_options = _suggest_additional_labels(pov_value, pairs, female_labels, male_labels, label_to_value)
    if not add_options:
        add_options = all_labels

    # Ensure current selection remains valid
    if st.session_state.add_label not in add_options:
        st.session_state.add_label = add_options[0]

    with col2:
        add_label = st.selectbox(
            "Additional Narrator",
            options=add_options,
            key="add_label",
        )

    add_value = label_to_value[add_label]

    with col3:
        if st.button("Swap"):
            st.session_state.pov_label, st.session_state.add_label = (
                st.session_state.add_label,
                st.session_state.pov_label,
            )
            st.rerun()

    # Compute the expected POV audio filename
    pov_filename = _build_with_filename(pov_value, add_value)

    # Determine pair base file for "download both PoVs": always Female and Male.mp3
    pair_base_filename = _pair_base_filename(pov_value, add_value, pairs)

    # Validate
    pov_file = by_name.get(pov_filename)
    valid_combo = pov_file is not None

    if not valid_combo:
        st.error("This combination is not valid.")
        st.caption(f'Expected file: "{pov_filename}"')
        return

    # Audio player and downloads
    try:
        audio_bytes = download_bytes(pov_file.download_url, token=token)
    except Exception as e:
        st.error(f"Failed to download audio: {e}")
        return

    st.audio(audio_bytes, format="audio/mpeg")

    dcol1, dcol2 = st.columns([1, 1])

    with dcol1:
        st.download_button(
            label="Download this audio",
            data=audio_bytes,
            file_name=pov_filename,
            mime="audio/mpeg",
        )

    with dcol2:
        if not pair_base_filename:
            st.download_button(
                label="Download both PoVs",
                data=b"",
                file_name="",
                disabled=True,
                mime="application/octet-stream",
            )
            st.caption('No matching "Female and Male" file was found for this pairing.')
        else:
            base_file = by_name.get(pair_base_filename)
            if not base_file:
                st.download_button(
                    label="Download both PoVs",
                    data=b"",
                    file_name="",
                    disabled=True,
                    mime="application/octet-stream",
                )
                st.caption(f'No matching base file found: "{pair_base_filename}"')
            else:
                try:
                    base_bytes = download_bytes(base_file.download_url, token=token)
                except Exception as e:
                    st.error(f"Failed to download pair audio: {e}")
                    return

                st.download_button(
                    label="Download both PoVs",
                    data=base_bytes,
                    file_name=pair_base_filename,
                    mime="audio/mpeg",
                )


def _suggest_additional_labels(
    pov_value: str,
    pairs: Set[Tuple[str, str]],
    female_labels: List[str],
    male_labels: List[str],
    label_to_value: Dict[str, str],
) -> List[str]:
    """
    If POV narrator is female, valid additional narrators are the paired males.
    If POV narrator is male, valid additional narrators are the paired females.
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


def _pair_base_filename(pov_value: str, add_value: str, pairs: Set[Tuple[str, str]]) -> Optional[str]:
    """
    Returns the canonical base filename: "<female> and <male>.mp3" for the selected narrators,
    regardless of dropdown order, if the pair exists.
    """
    # Check both orders against the known (female, male) pairs
    for f, m in pairs:
        if (pov_value == f and add_value == m) or (pov_value == m and add_value == f):
            return f"{f} and {m}.mp3"
    return None


if __name__ == "__main__":
    main()