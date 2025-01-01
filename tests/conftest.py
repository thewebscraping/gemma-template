import pytest  # noqa

from gemma_template import FieldPosition, Template

INPUT_TEMPLATE = """{{ system_prompt }}
{% if instruction %}\n{{ instruction }}\n{% endif %}
{% if prompt_structure %}{{ prompt_structure }}\n{% else %}{{ prompt }}\n{% endif %}
# Text:
{{ input }}
{% if topic_value %}\nTopics: {{ topic_value }}\n{% endif %}{% if keyword_value %}Keywords: {{ keyword_value }}\n{% endif %}
"""

OUTPUT_TEMPLATE = """{% if structure_fields %}{% for field in structure_fields %}## **{{ field.label.custom or field.label.default }}:**\n{% if field.key == 'title' %}### {% endif%}{{ field.value }}\n\n{% endfor %}{% else %}{{ output }}{% endif %}"""

INSTRUCTION_TEMPLATE = """# Role:
You are a highly skilled professional content writer, linguistic analyst, and multilingual expert specializing in structured writing and advanced text processing.

# Task:
Your primary objectives are:
1. Simplification: Rewrite the input text or document to ensure it is accessible and easy to understand for a general audience while preserving the original meaning and essential details.
2. Lexical and Grammatical Analysis: Analyze and refine vocabulary and grammar using unigrams (single words), bigrams (two words), and trigrams (three words) to enhance readability and depth.
3. Structure and Organization: Ensure your response adheres strictly to the prescribed structure format.
4. Language Consistency: Respond in the same language as the input text unless explicitly directed otherwise.

# Additional Guidelines:
1. Provide a rewritten, enhanced version of the input text, ensuring professionalism, clarity, and improved structure.
2. Focus on multilingual proficiency, using complex vocabulary, grammar to improve your responses.
3. Preserve the context and cultural nuances of the original text when rewriting.

# Text Analysis:
Example 1: Unigrams (single words){% for word in unigrams %}\n{{ word }} => {{ language }}{% endfor %}
Text Analysis 3: These are common {{ language }} words, indicating the text is in {{ language }}.

Example 2: Bigrams (two words){% for word in bigrams %}\n{{ word }} => {{ language }}{% endfor %}
Text Analysis 2: Frequent bigrams in {{ language }} confirm the language context.

Example 3: Trigrams (three words){% for word in trigrams %}\n{{ word }} => {{ language }}{% endfor %}
Text Analysis 3: Trigrams further validate the linguistic analysis and the necessity to respond in {{ language }}.

# Conclusion of Text Analysis:
The linguistic analysis confirms the text is predominantly in {{ language }}. Consequently, the response should be structured and written in {{ language }} to align with the original text and context.
"""  # noqa: E501

PROMPT_TEMPLATE = """{% if prompt %}\n\n# Input Text:\n{{ prompt }}\n\n{% endif %}{% if structure_fields %}# Response Structure Format:
You must follow the response structure:

{% for field in structure_fields %}{{ field.label }}\n{% endfor %}
By adhering to this format, the response will maintain linguistic integrity while enhancing professionalism, structure and alignment with user expectations.\n
{% endif %}"""  # noqa: E501


@pytest.fixture
def data():
    return [
        {
            "id": "JnZJolR76_u2",
            "title": "Gemma open models",
            "description": "Gemma: Introducing new state-of-the-art open models",
            "document": "Gemma open models are built from the same research and technology as Gemini models. Gemma 2 comes in 2B, 9B and 27B and Gemma 1 comes in 2B and 7B sizes.",
            "categories": ["Topic 1", "Topic 2"],
            "tags": ["Tag 1", "Tag 2"],
            "output": "Sample output",
            "main_points": ["Main point 1", "Main point 2"],
        },
        {
            "id": "JnZJolR76_u2",
            "title": "Gemma open models",
            "description": "Gemma: Introducing new state-of-the-art open models",
            "document": "Gemma open models are built from the same research and technology as Gemini models. Gemma 2 comes in 2B, 9B and 27B and Gemma 1 comes in 2B and 7B sizes.",
            "categories": ["Topic 1", "Topic 2"],
            "tags": ["Tag 1", "Tag 2"],
            "output": "Sample output",
            "main_points": ["Main point 1", "Main point 2"],
        },
    ]


@pytest.fixture
def config():
    return dict(
        max_hidden_ratio=.5,
        max_hidden_words=.1,
        min_chars_length=2,
        max_chars_length=8,
    )


@pytest.fixture
def template_instance():
    return Template(
        input_template=[INPUT_TEMPLATE],
        output_template=[OUTPUT_TEMPLATE],
        instruction_template=[INSTRUCTION_TEMPLATE],
        prompt_template=[PROMPT_TEMPLATE],
        position=FieldPosition(
            title=["Custom Title"],
            description=["Custom Description"],
            document=["Custom Article"],
            main_points=["Custom Main Points"],
            categories=["Custom Categories"],
            tags=["Custom Tags"],
        ),
        system_prompts=["You are a multilingual professional writer."],
    )
