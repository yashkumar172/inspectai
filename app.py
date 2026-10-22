"""InspectAI Streamlit app.

Run from the repository root with the project's virtual environment active:

    streamlit run app.py

The pages live in `inspectai.ui`; `.streamlit/config.toml` sets the 10 MB upload limit and turns
off usage statistics. Machine-specific locations come from the INSPECTAI_* environment variables.
"""

import logging

import streamlit as st

from inspectai.ui import about, dashboard_page, home, inspect_page

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="InspectAI", page_icon=":material/search:", layout="wide")

navigation = st.navigation(
    [
        st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
        st.Page(
            inspect_page.render,
            title="Inspect an image",
            icon=":material/image_search:",
            url_path="inspect",
        ),
        st.Page(
            dashboard_page.render,
            title="Evaluation dashboard",
            icon=":material/analytics:",
            url_path="dashboard",
        ),
        st.Page(
            about.render, title="About and limitations", icon=":material/info:", url_path="about"
        ),
    ]
)
navigation.run()
