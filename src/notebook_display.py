"""Helpers for viewing a finished report inside a Jupyter notebook."""

from pathlib import Path
from typing import Union

MATHJAX_CONFIG = r"""
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['\\[', '\\]']]
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""


def display_report(report_path: Union[str, Path]) -> None:
    """Render a saved HTML report inline in the notebook."""
    from IPython.display import HTML, display

    html_content = Path(report_path).read_text(encoding="utf-8")
    display(HTML(MATHJAX_CONFIG + html_content))
