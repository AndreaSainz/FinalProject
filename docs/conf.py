nbsphinx_kernel_name = 'python3'

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'ct_reconstruction'
copyright = '2024, Andrea Sainz Bear'
author = 'Andrea Sainz Bear'
release = '0.1.0-beta'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'nbsphinx',
    'myst_parser',
    'sphinx.ext.autodoc',    # Autodoc extension for extracting docstrings
	'sphinx.ext.mathjax',
	'sphinx_rtd_theme',
    'sphinx_gallery.load_style',  # load CSS for gallery (needs SG >= 0.6)
]
nbsphinx_exporter = "classic"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'navigation_depth': 2, 
    'collapse_navigation': True,  
    'titles_only': False,  
}



master_doc = 'index'

highlight_language = 'python3'

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# To write correctly the markdowns
myst_enable_extensions = ["dollarmath", "amsmath"] 

# For links in the markdowns
myst_url_schemes = ["http", "https", "mailto"]

# Disable section numbering
secnumber_suffix = ''  # No suffix means no section numbers
