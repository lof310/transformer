import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'transformer'
copyright = '2026, Leinier Orama Fernández'
author = 'Leinier Orama Fernández'
release = '0.1.0'

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages'
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist"
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_typehints = 'description'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'furo'
#html_logo = '_static/logo.png'
html_static_path = ['_static']

todo_include_todos = True
