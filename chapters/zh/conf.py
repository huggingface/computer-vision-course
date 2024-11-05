# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = "Compute Vision Course"
copyright = "Hugging Face Computer Vision community"
author = "Johannes Kolbe"

# The full version, including alpha/beta/rc tags
release = "v1.2"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "sphinx_gallery.load_style",  # load CSS for gallery (needs SG >= 0.6)
]

# 启用 myst-parser 的 dollarmath 扩展
myst_enable_extensions = [
    "dollarmath",
]

# 可选：配置 dollarmath 扩展的选项
# 允许在块级数学公式中使用标签
myst_dmath_allow_labels = True
# 允许在块级数学公式中使用编号
myst_dmath_allow_number = True


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".mdx": "markdown",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "sticky_navigation": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "style.css",  # 指定文件路径，如果在 _static 文件夹中，这里只需要写文件名
]
master_doc = "index"

highlight_language = "python3"

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
# pygments_style = 'sphinx'
