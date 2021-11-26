# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
"""
conf.py
"""
import os
import sys
sys.path.append(os.path.abspath('../../PaddleRec/'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))
import sphinx_rtd_theme
import sphinx_markdown_tables

# -- Project information -----------------------------------------------------

project = 'PaddleRec'
copyright = '2020, PaddlePaddle'
author = 'PaddlePaddle'

# The full version, including alpha/beta/rc tags
release = '2.2.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.mathjax',
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon', "markdown2rst",
    'sphinx.ext.githubpages', 'sphinx_markdown_tables'
]

# Support Inline mathjax
m2r_disable_inline_math = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
source_suffix = ['.rst', '.md']
#exclude_patterns = ['pgl.graph_kernel', 'pgl.layers.conv']
lanaguage = "zh_cn"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# for latex bug
#latex_engine = 'xelatex'
latex_engine = 'pdflatex'
#latex_engine = "xelatex"

latex_elements={# The paper size ('letterpaper' or 'a4paper').
    'papersize':'a4paper',# The font size ('10pt', '11pt' or '12pt').
    'pointsize':'12pt','classoptions':',oneside','babel':'',
    'inputenc':'',
    'utf8extra':'',
    'preamble': r"""
\usepackage{xeCJK}
\usepackage{indentfirst}
\setlength{\parindent}{2em}
\setCJKmainfont{WenQuanYi Micro Hei}
\setCJKmonofont[Scale=0.9]{WenQuanYi Micro Hei Mono}
\setCJKfamilyfont{song}{WenQuanYi Micro Hei}
\setCJKfamilyfont{sf}{WenQuanYi Micro Hei}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip = 0pt plus 1pt
"""}

extensions = ['recommonmark', 'sphinx_markdown_tables']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}
