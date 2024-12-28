
Gemma Template
==============

This library was developed for the Kaggle challenge:
[**Google - Unlocking Global Communication with Gemma**](https://www.kaggle.com/competitions/gemma-language-tuning), sponsored by Google.

### Credit Requirement

**Important:** If you are a participant in the competition and wish to use this source code in your submission,
you must clearly credit the original author before the competition's end date, **January 14, 2025**.

Please include the following information in your submission:

```text
Author: Tu Pham
Kaggle Username: [bigfishdev](https://www.kaggle.com/bigfishdev)
GitHub: [https://github.com/thewebscraping/gemma-template/](https://github.com/thewebscraping/gemma-template)
```

# Overview

**Gemma Template** is a lightweight and efficient Python library for generating templates to fine-tune models and craft prompts.
Designed for flexibility, it seamlessly supports Gemma, LLaMa and other language frameworks, offering fast, user-friendly customization.
With multilingual capabilities and advanced configuration options, ensures precise, professional, and dynamic template creation.

# Multilingual Content Writing Assistant

This writing assistant is a multilingual professional writer specializing in crafting structured, engaging, and SEO-optimized content.
It enhances text readability, aligns with linguistic nuances, and preserves original context across various languages.

---

## Key Features:
#### 1. **Creative and Engaging Rewrites**
- Transforms input text into captivating and reader-friendly content.
- Utilizes vivid imagery and descriptive language to enhance engagement.

#### 2. **Advanced Text Analysis**
- Processes text with unigrams, bigrams, and trigrams to understand linguistic patterns.
- Ensures language-specific nuances and cultural integrity are preserved.

#### 3. **SEO-Optimized Responses**
- Incorporates keywords naturally to improve search engine visibility.
- Aligns rewritten content with SEO best practices for discoverability.

#### 4. **Professional and Multilingual Expertise**
- Fully support for creating template with local language.
- Supports multiple languages with advanced vocabulary and grammar enhancement.
- Adapts tone and style to maintain professionalism and clarity.
- Support hidden mask input text.
- Optional: learn vocabulary enhancement with unigrams, bigrams and trigrams instruction template.
- Full documentation, easy configuration prompts with examples.

#### 5. **Customize Advanced Response Structure and Dataset Format**
- Fully support for advanced structure response format customization.
- Support output multiple formats such as Alpaca, GPT, STF text.
- Can be used with other models such as LLama.
- Dynamic prompts are enhanced using Round-Robin loop.

**Installation**
----------------

To install the library, you can choose between two methods:

#### **1\. Install via PyPI:**

```shell
pip install gemma-template
```

#### **2\. Install via GitHub Repository:**

```shell
pip install git+https://github.com/thewebscraping/gemma-template.git
```

**Quick Start**
----------------
Start using Gemma Template with just a few lines of code:

```python
from gemma_template.models import *

prompt_instance = Template(
         structure_field=StructureField(
         title=["Custom Title"],
         description=["Custom Description"],
         document=["Custom Article"],
         main_points=["Custom Main Points"],
         categories=["Custom Categories"],
         tags=["Custom Tags"],
    ),
)   # Create fully customized structured reminders.

response = prompt_instance.template(
    template=GEMMA_TEMPLATE,
    user_template=USER_TEMPLATE,
    instruction_template=INSTRUCTION_TEMPLATE,
    structure_template=STRUCTURE_TEMPLATE,
    title="Gemma open models",
    description="Gemma: Introducing new state-of-the-art open models.",
    document="Gemma open models are built from the same research and technology as Gemini models. Gemma 2 comes in 2B, 9B and 27B and Gemma 1 comes in 2B and 7B sizes.",
    main_points=["Main point 1", "Main point 2"],
    categories=["Artificial Intelligence", "Gemma"],
    tags=["AI", "LLM", "Google"],
    output="A new family of open language models demonstrating strong performance across academic benchmarks for language understanding, reasoning, and safety.",
    max_hidden_words=.1,  # set 0 if you don't want to hide words.
    min_chars_length=2,  # Minimum character of a word, used to create unigrams, bigrams, and trigrams. Default is 2.
    max_chars_length=0,  # Maximum character of a word, used to create unigrams, bigrams and trigrams.. Default is 0.
 )  # remove kwargs if not used.
print(response)
```

### Output:

```text
<start_of_turn>user

You are a multilingual professional writer.

Rewrite the text with a more engaging and creative tone. Use vivid imagery, descriptive language, and a conversational style to captivate the reader.

# Role:
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

Topics: Artificial Intelligence, Gemma
Keywords: AI, LLM, Google

# Text Analysis:
Example 1: Unigrams (single words)
and => English
built => English
from => English
the => English
research => English
Text Analysis 3: These are common English words, indicating the text is in English.

Example 2: Bigrams (two words)
technology as => English
Text Analysis 2: Frequent bigrams in Vietnamese confirm the language context.

Example 3: Trigrams (three words)
technology as Gemini => English
Text Analysis 3: Trigrams further validate the linguistic analysis and the necessity to respond in English.

# Conclusion of Text Analysis:
The linguistic analysis confirms the text is predominantly in English. Consequently, the response should be structured and written in English to align with the original text and context.

# Response Structure Format:
You must follow the response structure:
**Custom Title (Title):** Rewrite the title to make it concise, memorable, and optimized for SEO.
**Custom Description (Description):** Write description of the article in one or two sentences while focusing on reader benefits and engage curiosity.
**Custom Article (Article):** Rewrite this content to be SEO-friendly. Include relevant tags, optimize the title and subheadings, and ensure the text flows naturally for search engines and readers.
**Custom Main Points (Main Points):** Simplify the original key points to make them clearer and more reader-friendly.
**Custom Categories (Categories):** Assign appropriate categories to the article based text or target audience.
**Custom Tags (Tags):** Create tags to include relevant keywords. Ensure the tags align with popular search queries.

By adhering to this format, the response will maintain linguistic integrity while enhancing professionalism, structure and alignment with user expectations.

# Text:
Gemma open models are built from _____ same research _____ technology as Gemini models. Gemma 2 comes in 2B, 9B _____ 27B and Gemma 1 comes in 2B and 7B sizes.

<end_of_turn>
<start_of_turn>model

## **Custom Title**:
### Gemma open models

## **Custom Description**:
Gemma: Introducing new state-of-the-art open models.

## **Custom Article**:
A new family of open language models demonstrating strong performance across academic benchmarks for language understanding, reasoning, and safety.

## **Custom Main Points**:
- Main point 1
- Main point 2

## **Custom Categories**:
- Artificial Intelligence
- Gemma

## **Custom Tags**:
- AI
- LLM
- Google<end_of_turn>

```

## Load Dataset
Returns: Dataset: A Hugging Face Dataset or DatasetDict object containing the processed prompts.

**Load Dataset from local file path**
```python
prompt_instance = Template()
data_dict = [
    {
        "id": "JnZJolR76_u2",
        "title": "Sample title",
        "description": "Sample description",
        "document": "Sample document",
        "categories": ["Topic 1", "Topic 2"],
        "tags": ["Tag 1", "Tag 2"],
        "output": "Sample output",
        "main_points": ["Main point 1", "Main point 2"],
    }
]
dataset = prompt_instance.load_dataset(data_dict, output_format='text')   # enum: text, gpt, alpaca
print(dataset['text'][0])
```

**Load Dataset from HuggingFace**
```python
dataset = gemma_template.load_dataset(
    "your_huggingface_dataset",
    output_format='gpt',   # enum: text, gpt, alpaca
    instruction_template=INSTRUCTION_TEMPLATE,  # Template for instruction the user prompt.
    structure_template=STRUCTURE_TEMPLATE,   # Template for structuring the user prompt.
    max_hidden_ratio=10,  # Percentage of documents that need to be word masked. Min: 0, Max: 1. Default: 0.
    # Replace 10% of words in the input document with '_____'.
    # Use int to extract the correct number of words. The `max_hidden_ratio` parameter must be greater than 0.
    max_hidden_words=.1,
    min_chars_length=2,   # Minimum character of a word, used to create unigrams, bigrams, and trigrams. Default is 2.
    max_chars_length=8    # Maximum character of a word, used to create unigrams, bigrams and trigrams. Default is 0.
)
```
