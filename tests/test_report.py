import json
import re

import pytest

from framing import report


def test_render_embeds_payload():
    html = report.render(
        {"a": 1, "b": [1.5, float("nan")]}, "<script>const D = /*__PAYLOAD__*/null;</script>"
    )
    blob = re.search(r"const D = (.*);</script>", html).group(1)
    assert json.loads(blob) == {"a": 1, "b": [1.5, None]}


def test_render_escapes_closing_script():
    html = report.render({"name": "</script><b>"}, "<script>/*__PAYLOAD__*/null</script>")
    assert html.count("</script>") == 1


def test_render_requires_placeholder():
    with pytest.raises(ValueError):
        report.render({}, "<html></html>")


def test_packaged_template_has_placeholder():
    html = report.render({"x": 1})
    assert "/*__PAYLOAD__*/" not in html and '{"x":1}' in html
