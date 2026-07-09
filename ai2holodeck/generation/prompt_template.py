"""Minimal ``PromptTemplate`` shim, replacing the legacy ``langchain``
dependency.

Holodeck historically used ``langchain==0.0.171`` solely for
``PromptTemplate`` (and the ``OpenAI`` class as a type hint). The actual LLM
client is the modern ``openai`` SDK wired up in ``holodeck.py``; langchain was
never invoked at runtime.

``langchain==0.0.171`` pins ``numpy<2``, which conflicts with the rest of the
stack (``bpy>=5.1.2`` requires ``numpy>=2.2``). This shim reproduces the
narrow slice of ``PromptTemplate`` behavior Holodeck relies on — substituting
only the declared ``input_variables`` and leaving every other ``{...}`` (e.g.
the JSON examples embedded in ``prompts.py``) untouched — so langchain can be
dropped entirely.
"""

from typing import Mapping


class PromptTemplate:
    """Format a template by substituting declared input variables only.

    Mirrors ``langchain.PromptTemplate`` usage in this repo:
    ``PromptTemplate(input_variables=[...], template=...).format(**kwargs)``.

    Unlike ``str.format``, this does *not* interpret ``{``/``}`` that are not
    one of the declared ``input_variables`` — so embedded JSON examples in the
    prompt strings (which contain literal, unmatched braces) pass through
    verbatim.
    """

    def __init__(self, input_variables, template, template_format="f-string"):
        self.input_variables = list(input_variables)
        self.template = template
        self.template_format = template_format

    def format(self, **kwargs: Mapping[str, object]) -> str:
        result = self.template
        for var in self.input_variables:
            result = result.replace("{" + var + "}", str(kwargs[var]))
        return result

    def format_prompt(self, **kwargs: Mapping[str, object]) -> str:
        return self.format(**kwargs)
