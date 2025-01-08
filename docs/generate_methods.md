# Generate Methods

```pycon
>>> from gemma_template import Template, gemma_template
>>> isinstance(gemma_template, Template)
True
```

See also: [Method Arguments](../models/#method-arguments)

## Generate User Prompt
Create user prompt for Gemma Fine tuning.

!!! Parameters
    * **max_hidden_words (Union[int, float]):** default `0`.
        * Replace words in the document with '_____'.
        * `int`: exact number of words to be masked.
        * `float`: percentage of number of words to be masked.
    * **min_chars_length (int):** default `2`.
        * Minimum character of a word, used to create unigrams, bigrams, and trigrams. Default is 2.
    * **max_chars_length (int):** default `0`.
        * Maximum character of a word, used to create unigrams, bigrams and trigrams. Default is 0.
    * **bullet_style (enum):** default `asterisk`
        * Create Markdown style bulleted list `asterisk`, `dash`, `number`.

```pycon
>>> prompt = gemma_template.generate_user_prompt(
...     document='This is a Test!',
...     description='Test description',
...     categories=['Example 1', 'Example 2'],
...     tags=['Tag 1', 'Tag 2'], output='Test'
... )
'''
You are a multilingual professional writer.

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
Example 1: Unigrams (single words)
This => English
Text Analysis 3: These are common English words, indicating the text is in English.

Example 2: Bigrams (two words)
is a => English
a Test => English
Text Analysis 2: Frequent bigrams in English confirm the language context.

Example 3: Trigrams (three words)
is a Test => English
Text Analysis 3: Trigrams further validate the linguistic analysis and the necessity to respond in English.

# Conclusion of Text Analysis:
The linguistic analysis confirms the text is predominantly in English. Consequently, the response should be structured and written in English to align with the original text and context.

# Input Text:
Rewrite the input text or document to make it more persuasive and compelling. Focus on strengthening arguments, appealing to emotions, and using rhetorical techniques to convince the reader.

# Response Structure Format:
You must follow the response structure:

**Meta Description:** Begin the description with a thought-provoking question to spark curiosity and encourage clicks.
**Edit Article:** Rewrite the input text or document to make it simpler and easier to understand for a general audience. Use clear and concise language while preserving the original meaning and key details.
**Topics:** Rewrite categories to match keywords used in the article.
**Keywords:** Rewrite tags to include relevant keywords in about 5 keywords.

By adhering to this format, the response will maintain linguistic integrity while enhancing professionalism, structure and alignment with user expectations.

# Text:
This is a Test!
'''
```

### Masked Hidden Text

```pycon
>>> prompt = gemma_template.generate_user_prompt(
...     document='Gemma open models are built from the same research and technology as Gemini models. Gemma 2 comes in 2B, 9B and 27B and Gemma 1 comes in 2B and 7B sizes.',
...     max_hidden_words=0.1,
...     min_chars_length=2,
... )
'''
You are a multilingual professional writer.
...

# Text:
Gemma open models are built from _____ same _____ and technology _____ Gemini models. Gemma 2 comes in 2B, 9B and 27B and Gemma 1 comes in 2B and 7B sizes.
'''
```

***

## Generate Model Prompt
Create model prompt for Gemma Fine tuning.

```pycon
>>> prompt = gemma_template.generate_model_prompt(
...     document='This is a Test!',
...     description='Test description',
...     categories=['Example 1', 'Example 2'],
...     tags=['Tag 1', 'Tag 2'], output='Test'
... )

'''## **Introduction:**
Test description

## **Article:**
Test

## **Categories:**
* Example 1
* Example 2

## **Tags:**
* Tag 1
* Tag 2
'''
```

***

## Generate Prompt for Question
Quickly create question prompts using the Gemma model.

```pycon
>>> prompt = gemma_template.generate_prompt(document='This is a Test!')
'''
<start_of_turn>user
You are a multilingual professional writer.

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
Example 1: Unigrams (single words)
This => English
Text Analysis 3: These are common English words, indicating the text is in English.

Example 2: Bigrams (two words)
is a => English
a Test => English
Text Analysis 2: Frequent bigrams in English confirm the language context.

Example 3: Trigrams (three words)
is a Test => English
Text Analysis 3: Trigrams further validate the linguistic analysis and the necessity to respond in English.

# Conclusion of Text Analysis:
The linguistic analysis confirms the text is predominantly in English. Consequently, the response should be structured and written in English to align with the original text and context.

# Input Text:
Reimagine the input text or document with a more engaging and creative tone. Add  metaphors, analogies, or storytelling elements to make it more captivating for readers.

# Response Structure Format:
You must follow the response structure:

**Article:** Reimagine the input text or document with a more engaging and creative tone. Add  metaphors, analogies, or storytelling elements to make it more captivating for readers.

By adhering to this format, the response will maintain linguistic integrity while enhancing professionalism, structure and alignment with user expectations.

# Text:
This is a Test!
<end_of_turn>
<start_of_turn>model
'''
```
