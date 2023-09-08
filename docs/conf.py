# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QuantumReservoirPy'
copyright = '2023 SINTEF Digital'
author = 'SINTEF Digital'
language = "en"



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_design"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "_static/qreservoirpy_logo_original.jpg"
html_favicon = "_static/qreservoirpy_logo_original.jpg"
html_title = "QuantumReservoirPy"
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    "css/styles.css"
]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/OpenQuantumComputing/quantumreservoirpy",
            "icon": "fa-brands fa-github"
        }
    ],
    "logo": {
        "text": "QuantumReservoirPy"
    },
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "article_header_start": [],
    "article_footer_items": [],
    "footer_start": ["copyright"],
    "footer_end": [],
    "show_toc_level": 3
}

html_show_sourcelink = False
