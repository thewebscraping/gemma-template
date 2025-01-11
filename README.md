# Gemma Template

This library was developed for the Kaggle challenge:
[**Google - Unlocking Global Communication with Gemma**](https://www.kaggle.com/competitions/gemma-language-tuning), sponsored by Google.

## Credit Requirement

**Important:** If you are a participant in the competition and wish to use this source code in your submission,
you must clearly credit the original author before the competition's end date, **January 14, 2025**.

Please include the following information in your submission:

```text
Author: Tu Pham
Kaggle Username: [bigfishdev](https://www.kaggle.com/bigfishdev)
GitHub: [https://github.com/thewebscraping/gemma-template/](https://github.com/thewebscraping/gemma-template)
LinkedIn: [https://www.linkedin.com/in/thetwofarm](https://www.linkedin.com/in/thetwofarm)
```

# Overview

Gemma Template is a lightweight and efficient Python library for generating templates to fine-tune models and craft prompts.
Designed for flexibility, it seamlessly supports Gemma, LLaMA, and other language frameworks, offering fast, user-friendly customization.
With multilingual capabilities and advanced configuration options, it ensures precise, professional, and dynamic template creation.

### Learning Process and Acknowledgements
As a newbie, I created Gemma Template based on what I read and learned from the following sources:

- Google Gemma Cookbook: [Advanced Prompting Techniques](https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/Advanced_Prompting_Techniques.ipynb)
- Google Gemma Cookbook: [Finetune with LLaMA Factory](https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/Finetune_with_LLaMA_Factory.ipynb)
- Google Gemma Cookbook: [Fine tuning Gemma for Function Calling](https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/Finetuning_Gemma_for_Function_Calling.ipynb)
- Alpaca: [Alpaca Lora Documentation](https://github.com/tloen/alpaca-lora)
- Unsloth: [Finetune Llama 3.2, Mistral, Phi-3.5, Qwen 2.5 & Gemma 2-5x faster with 80% less memory!](https://github.com/unslothai/unsloth)


Gemma Template supports exporting dataset files in three formats: `Text`, `Alpaca`, and `OpenAI`.

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
- Full support for creating templates in local languages.
- Supports multiple languages with advanced prompting techniques.
- Vocabulary and grammar enhancement with unigrams, bigrams, and trigrams instruction template.
- Supports hidden mask input text. Adapts tone and style to maintain professionalism and clarity.
- Full documentation with easy configuration prompts and examples.

#### 5. **Customize Advanced Response Structure and Dataset Format**
- Supports advanced response structure format customization.
- Compatible with other models such as LLaMa.
- Enhances dynamic prompts using Round-Robin loops.
- Outputs multiple formats such as Text, Alpaca and OpenAI.

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

## Load Dataset
Returns: A Hugging Face Dataset or DatasetDict object containing the processed prompts.

**Load Dataset from data dict**
```python
from gemma_template import gemma_template

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
dataset = gemma_template.load_dataset(data_dict, output_format='text')   # enum: `text`, `alpaca` and `openai`.
print(dataset['text'][0])
```

**Load Dataset from HuggingFace**
```python
from gemma_template import gemma_template

dataset = gemma_template.load_dataset(
    "YOUR_JSON_FILE_PATH_OR_HUGGINGFACE_DATASET",
    # enum: `text`, `alpaca` and `openai`.
    output_format='text',
    # Percentage of documents that need to be word masked.
    # Min: 0, Max: 1. Default: 0.
    max_hidden_ratio=.1,
    # Replace 10% of words in the input document with '_____'.
    # Use int to extract the correct number of words. The `max_hidden_ratio` parameter must be greater than 0.
    max_hidden_words=.1,
    # Minimum character of a word, used to create unigrams, bigrams, and trigrams. Default is 2.
    min_chars_length=2,
    # Maximum character of a word, used to create unigrams, bigrams and trigrams. Default is 0.
    max_chars_length=8,
)
```

## Fully Customized Template

```python
from gemma_template import Template, FieldPosition, INPUT_TEMPLATE, OUTPUT_TEMPLATE, INSTRUCTION_TEMPLATE, PROMPT_TEMPLATE

template_instance = Template(
    instruction_template=[INSTRUCTION_TEMPLATE],  # Optional: dynamic Round-Robin loops
    prompt_template=[PROMPT_TEMPLATE],  # Optional: dynamic Round-Robin loops
    input_template=[INPUT_TEMPLATE],  # Optional: dynamic Round-Robin loops
    output_template=[OUTPUT_TEMPLATE],  # Optional: dynamic Round-Robin loops
    position=FieldPosition(
            title=["Custom Title"],
            description=["Custom Description"],
            document=["Custom Article"],
            main_points=["Custom Main Points"],
            categories=["Custom Categories"],
            tags=["Custom Tags"],
    ),  # Optional: dynamic Round-Robin loops
)

response = template_instance.apply_template(
    title="Gemma open models",
    description="Gemma: Introducing new state-of-the-art open models.",
    main_points=["Main point 1", "Main point 2"],
    categories=["Artificial Intelligence", "Gemma"],
    tags=["AI", "LLM", "Google"],
    document="Gemma open models are built from the same research and technology as Gemini models. Gemma 2 comes in 2B, 9B and 27B and Gemma 1 comes in 2B and 7B sizes.",
    output="A new family of open language models demonstrating strong performance across academic benchmarks for language understanding, reasoning, and safety.",
    max_hidden_words=.1,  # set 0 if you don't want to hide words.
    min_chars_length=2,  # Minimum character of a word, used to create unigrams, bigrams, and trigrams. Default is 2.
    max_chars_length=0,  # Maximum character of a word, used to create unigrams, bigrams and trigrams. Default is 0.
)  # remove kwargs if not used.

print(response)
```

### Output:

```text
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
and => English
built => English
from => English
the => English
research => English
Text Analysis 3: These are common English words, indicating the text is in English.

Example 2: Bigrams (two words)
technology as => English
Text Analysis 2: Frequent bigrams in English confirm the language context.

Example 3: Trigrams (three words)
technology as Gemini => English
Text Analysis 3: Trigrams further validate the linguistic analysis and the necessity to respond in English.

# Conclusion of Text Analysis:
The linguistic analysis confirms the text is predominantly in English. Consequently, the response should be structured and written in English to align with the original text and context.

# Input Text:
Rewrite the input text or document to highlight its unique value proposition while ensuring it ranks well for targeted keywords.

# Response Structure Format:
You must follow the response structure:

**Custom Title (Title):** Rewrite the title to maximize clarity, appeal, and relevance to the content.
**Custom Description (Description):** Create a description focusing on how the article addresses a common problem or challenge readers face.
**Custom Article (Article):** Rewrite the input text or document with an authoritative tone, incorporating credible sources, data, and references to boost trustworthiness and SEO ranking.
**Custom Main Points (Main Points):** Ensure all key points flow logically from one to the next.
**Custom Categories (Categories):** Use categories that align with similar articles on the topic and improve SEO and discoverability.
**Custom Tags (Tags):** Rewrite tags to make them more specific and targeted.

By adhering to this format, the response will maintain linguistic integrity while enhancing professionalism, structure and alignment with user expectations.

# Text:
Gemma open models are built _____ the same _____ and technology as Gemini models. Gemma 2 comes in 2B, 9B _____ 27B and Gemma 1 comes in 2B and 7B sizes.<end_of_turn>
<start_of_turn>model
## **Custom Title:**
### Gemma open models

## **Custom Description:**
Gemma: Introducing new state-of-the-art open models.

## **Custom Article:**
A new family of open language models demonstrating strong performance across academic benchmarks for language understanding, reasoning, and safety.

## **Custom Main Points:**
* Main point 1
* Main point 2

## **Custom Categories:**
* Artificial Intelligence
* Gemma

## **Custom Tags:**
* AI
* LLM
* Google<end_of_turn>

```

**Read the documentation:** [https://thewebscraping.github.io/gemma-template/](https://thewebscraping.github.io/gemma-template/)
