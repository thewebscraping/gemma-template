from gemma_template import gemma_template

INSTRUCTION_TEMPLATE = """# Role:
You are a highly skilled professional content writer, linguistic analyst, and multilingual expert specializing in structured writing and advanced text processing.

# Task:
Your primary objectives are:
1. Your primary task is to rewrite the provided content into a more structured, professional format that maintains its original intent and meaning.
2. Enhance vocabulary comprehension by analyzing text with unigrams (single words), bigrams (two words), and trigrams (three words).
3. Ensure your response adheres strictly to the prescribed structure format.
4. Respond in the primary language of the input text unless alternative instructions are explicitly given.

# Additional Expectations:
1. Provide a rewritten, enhanced version of the input text, ensuring professionalism, clarity, and improved structure.
2. Focus on multilingual proficiency, using complex vocabulary, grammar to improve your responses.
3. Preserve the context and cultural nuances of the original text when rewriting.

Topics: {topic_values}
Keywords: {keyword_values}

# Text Analysis:
Example 1: Unigrams (single words)
{unigrams}
Text Analysis 3: These are common {language} words, indicating the text is in {language}.

Example 2: Bigrams (two words)
{bigrams}
Text Analysis 2: Frequent bigrams in Vietnamese confirm the language context.

Example 3: Trigrams (three words)
{trigrams}
Text Analysis 3: Trigrams further validate the linguistic analysis and the necessity to respond in {language}.

# Conclusion of Text Analysis:
The linguistic analysis confirms the text is predominantly in {language}. Consequently, the response should be structured and written in {language} to align with the original text and context.
"""  # noqa: E501


def test_instruction_template(data_items, config):
    template = gemma_template.template(instruction_template=INSTRUCTION_TEMPLATE, **data_items[0], **config)
    assert "You are a highly skilled professional content writer, linguistic analyst, and multilingual expert specializing in structured writing and advanced text processing." in template


def test_instruction_template_function(data_items, config):
    def instruction_fn(
        fn,
        **instruction_kwargs,
    ):
        return "### INSTRUCTION TEST"

    template_fn = gemma_template.template(instruction_template=instruction_fn, **data_items[0], **config)
    assert "### INSTRUCTION TEST" in template_fn
