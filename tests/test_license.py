"""LICENSE must stay the verbatim MIT text.

GitHub's license detector (and JOSS reviewers, who require an OSI-approved
license) match LICENSE against the canonical MIT text. Anything appended to it
makes GitHub report "Other". That happened once: a scope note carving the
embedded data out of the MIT grant was appended here. Explanations of what MIT
does and doesn't cover belong in DATA_LICENSE.md.
"""

MIT_BODY = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def _words(text: str) -> list[str]:
    return text.split()


def test_license_is_plain_mit(project_root):
    text = (project_root / "LICENSE").read_text()
    head, sep, body = text.partition("Permission is hereby granted")
    assert sep, "LICENSE is missing the MIT permission notice"

    # Header is the title plus one copyright line. The detector strips a
    # single-line copyright; a wrapped holder name leaves stray words behind.
    header = [line for line in head.splitlines() if line.strip()]
    assert header[0] == "MIT License"
    assert len(header) == 2 and header[1].startswith("Copyright (c) "), (
        f"expected 'MIT License' + one 'Copyright (c) ...' line, got {header}"
    )

    assert _words(sep + body) == _words(MIT_BODY), (
        "LICENSE differs from the canonical MIT text, so GitHub will report "
        "it as 'Other'. Put scope notes in DATA_LICENSE.md instead."
    )
