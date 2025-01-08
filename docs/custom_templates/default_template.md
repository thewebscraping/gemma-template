# Default Templates
Gemma Template uses Jinja2 template.

See also: [`models.Attr`](../models.md#attributes_5)

* * *

# Gemma Fine Tuning Template
```text
<start_of_turn>user
{{ input }}<end_of_turn>
<start_of_turn>model
{{ output }}<end_of_turn>

```

* * *

# Gemma Prompt Template
```text
<start_of_turn>user
{{ input }}<end_of_turn>
<start_of_turn>model

```

* * *

# Input Template
```text
{{ system_prompt }}
{% if instruction %}\n{{ instruction }}\n{% endif %}
{% if prompt_structure %}{{ prompt_structure }}\n{% else %}{{ prompt }}\n{% endif %}
# Text:
{{ input }}
{% if topic_value %}\nTopics: {{ topic_value }}\n{% endif %}{% if keyword_value %}Keywords: {{ keyword_value }}\n{% endif %}
```

* * *

# Output Template
```text
{% if structure_fields %}{% for field in structure_fields %}## **{{ field.label.custom or field.label.default }}:**\n{% if field.key == 'title' %}### {% endif%}{{ field.value }}\n\n{% endfor %}{% else %}{{ output }}{% endif %}
```

* * *

# Instruction Template
```text
# Role:
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
```

* * *

# Prompt Template
```text
{% if prompt %}\n\n# Input Text:\n{{ prompt }}\n\n{% endif %}{% if structure_fields %}# Response Structure Format:
You must follow the response structure:

{% for field in structure_fields %}{{ field.label }}\n{% endfor %}
By adhering to this format, the response will maintain linguistic integrity while enhancing professionalism, structure and alignment with user expectations.\n
{% endif %}
```
