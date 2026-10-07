"""Markdown report templates with {Placeholder_Name} values and toggleable sections.

Template syntax
---------------
* ``{Placeholder_Name}``  is replaced with generated content (see report_components.py).
* ``<!-- BEGIN:Section_Name -->`` ... ``<!-- END:Section_Name -->`` wraps an optional
  section. Sections listed in ``exclude_sections`` are removed. Sections cannot be nested.
* ``<!-- description: text -->`` (optional) is shown when listing templates.
* ``\\[ ... \\]`` blocks are passed through untouched so MathJax can render formulas.
* Everything else is standard Markdown (tables supported); raw HTML is also allowed.
"""

import html
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Union

import markdown

logger = logging.getLogger(__name__)

SECTION_RE = re.compile(r"<!--\s*BEGIN:(\w+)\s*-->(.*?)<!--\s*END:\1\s*-->", re.DOTALL)
DESCRIPTION_RE = re.compile(r"<!--\s*description:\s*(.*?)\s*-->", re.DOTALL)
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*)\}")
MATH_RE = re.compile(r"\\\[.*?\\\]", re.DOTALL)

STYLE_FILE = "report.css"


@dataclass
class Placeholder:
    """Value for a template placeholder.

    kind: "text" (HTML-escaped), "markdown" (inserted before conversion) or
    "html" (inserted verbatim after conversion, for tables).
    """

    value: str
    kind: str = "text"


def list_templates(templates_dir: Union[str, Path]) -> Dict[str, str]:
    """Return {template name: description} for every .md file in templates_dir."""
    found = {}
    for path in sorted(Path(templates_dir).glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        match = DESCRIPTION_RE.search(path.read_text(encoding="utf-8"))
        found[path.stem] = match.group(1) if match else ""
    return found


def resolve_template(template: Union[str, Path], templates_dir: Union[str, Path]) -> Path:
    """Find a template by name (in templates_dir) or by path to a .md file."""
    candidate = Path(template)
    if candidate.suffix.lower() == ".md" and candidate.is_file():
        return candidate.resolve()
    named = Path(templates_dir) / f"{template}.md"
    if named.is_file():
        return named
    available = sorted(list_templates(templates_dir))
    raise FileNotFoundError(f"Template '{template}' not found. Available templates: {available}")


def list_sections(template_path: Union[str, Path]) -> List[str]:
    """Names of the optional sections defined in a template."""
    return [m.group(1) for m in SECTION_RE.finditer(Path(template_path).read_text(encoding="utf-8"))]


def find_placeholders(text: str) -> List[str]:
    """Placeholder-like names used in template text (math blocks ignored)."""
    return sorted(set(PLACEHOLDER_RE.findall(MATH_RE.sub("", text))))


def render_template(
    template_path: Union[str, Path],
    placeholders: Dict[str, Placeholder],
    exclude_sections: Iterable[str] = (),
) -> str:
    """Render a template to an HTML fragment."""
    template_path = Path(template_path)
    source = template_path.read_text(encoding="utf-8")
    text = DESCRIPTION_RE.sub("", source, count=1)
    text = _apply_sections(text, set(exclude_sections), template_path.name)

    math_blocks: List[str] = []

    def stash_math(match):
        math_blocks.append(match.group(0))
        return f"@@MATH_{len(math_blocks) - 1}@@"

    text = MATH_RE.sub(stash_math, text)

    html_blocks: List[str] = []

    def substitute(match):
        name = match.group(1)
        placeholder = placeholders.get(name)
        if placeholder is None:
            if "_" in name:
                logger.warning("Template %s uses unknown placeholder {%s}", template_path.name, name)
            return match.group(0)
        if placeholder.kind == "html":
            html_blocks.append(placeholder.value)
            return f"@@HTML_{len(html_blocks) - 1}@@"
        if placeholder.kind == "markdown":
            return placeholder.value
        return html.escape(str(placeholder.value))

    text = PLACEHOLDER_RE.sub(substitute, text)

    body = markdown.markdown(text, extensions=["tables", "sane_lists"])
    body = body.replace("<table>", '<table class="styled-table">')

    for i, block in enumerate(html_blocks):
        token = f"@@HTML_{i}@@"
        body = body.replace(f"<p>{token}</p>", block).replace(token, block)
    for i, block in enumerate(math_blocks):
        body = body.replace(f"@@MATH_{i}@@", block)

    unused = sorted(set(placeholders) - set(find_placeholders(source)))
    logger.debug("Placeholders not used by %s: %s", template_path.name, unused)
    return body


def _apply_sections(text: str, excluded: set, template_name: str) -> str:
    defined = {m.group(1) for m in SECTION_RE.finditer(text)}
    unknown = excluded - defined
    if unknown:
        logger.warning(
            "exclude_sections not found in template %s: %s (available: %s)",
            template_name, sorted(unknown), sorted(defined),
        )

    def replace(match):
        if match.group(1) in excluded:
            logger.debug("Excluding section %s", match.group(1))
            return ""
        return match.group(2)

    return SECTION_RE.sub(replace, text)


def load_style(template_path: Union[str, Path], templates_dir: Union[str, Path]) -> str:
    """CSS for a template: <template>.css next to it, else templates/report.css."""
    for css in (Path(template_path).with_suffix(".css"), Path(templates_dir) / STYLE_FILE):
        if css.is_file():
            return css.read_text(encoding="utf-8")
    logger.warning("No stylesheet found (%s); report will be unstyled", Path(templates_dir) / STYLE_FILE)
    return ""
