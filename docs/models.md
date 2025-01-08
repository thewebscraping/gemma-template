# Models
This class is inspired by advanced natural language processing needs and optimized for use with open models like Gemma.

## Template

The `Template` class extends the `BaseTemplate` class to provide specialized functionality for generating structured prompts.
It combines system, user, and structural prompt templates to create flexible, multi-purpose content generation workflows.

### **Attributes**

* `template` (`list[TemplateTypes]`): Base template for constructing the final prompt.
* `input_template` (`list[TemplateTypes]`): Template for user inputs.
* `output_template` (`list[TemplateTypes]`): Template for model outputs.
* `instruction_template` (`list[TemplateTypes]`): Template for instructions, if applicable.
* `prompt_template` (`list[TemplateTypes]`): Template for structured prompts, if applicable.
* `system_prompts` (`list[str]`): Defines the role or behavior of the model.
* `user_prompts` (`list[str]`): Specifies user queries or requests.
* `title` (`list[str]`): Collection of SEO-optimized titles.
* `description` (`list[str]`): Compelling introductions or meta descriptions.
* `document` (`list[str]`): Enhances main content.
* `main_points` (`list[str]`): Summarizes or emphasizes key points.
* `categories` (`list[str]`): Identifies article themes.
* `tags` (`list[str]`): Enhances tags for SEO purposes.
* `position` (`Optional[FieldPosition]`): Manages structured fields within the prompt.

### **Methods**

* `load_dataset`: Processes and loads a dataset, generating prompts based on the provided templates. This function supports various input formats such as file paths, dictionaries, or Hugging Face Dataset objects. It uses templates to create structured prompts and supports concurrent processing for efficiency.
* `to_text`: Generate Text format.
* `to_alpaca`: Generate Alpaca format.
* `to_openai`: Generate OpenAI format.
* `apply_template`: Generates a complete prompt by integrating system, user, and structural elements.
* `generate_prompt`: Generates a prompt to predict.
* `generate_user_prompt`: Generates a user-specific prompt by combining multiple user-defined inputs. This method collects and formats user prompts, optionally including structural elements, to create a coherent and complete prompt.
* `generate_model_prompt`: Generates a model-specific prompt by formatting output and document content according to structural rules. This method organizes output content and any additional fields into a structured format for model processing.
* `get_template_attr`: Generates an kwargs for the Attr instance.


### **Method Arguments**

* `document` (`Optional[list[str]]`): Model input, typically the same as the article content.
* `output` (`Optional[str]`): Model response output, typically the same as the document field.
* `title` (`Optional[str]`): Model response output, typically the same as the title field.
* `description` (`Optional[str]`): Model response output, typically the same as the description field.
* `main_points` (`Optional[list[str]]`): Model response output, typically the same as the main_points field.
* `categories` (`Optional[list[str]]`): Model response output, typically the same as the categories field.
* `tags` (`Optional[list[str]]`): Model response output, typically the same as the tags field.
* `excluded_fields` (`Optional[Sequence[str]]`): Fields excluded to response. Default is empty sequence.
* `bullet_style` (`Optional[Literal['dash', 'number', 'asterisk']]`): The style of the bullet points in the prompt. Default is `'asterisk'`. Possible values:
    * `'dash'`: Bullets styled with dashes (`-`).
    * `'number'`: Bullets styled with numbers (`1.`, `2.`, etc.).
    * `'asterisk'`: Bullets styled with asterisks (`*`).
* `max_hidden_words` (`Union[int, float]`):
    * Replace words in the document with _____.
    * `int`: exact number of words to be masked.
    * `float`: percentage of number of words to be masked.
* `min_chars_length` (`int`): Minimum character of a word, used to create unigrams, bigrams, and trigrams.
* `max_chars_length` (`int`): Maximum character of a word, used to create unigrams, bigrams and trigrams.



## FieldPosition
Inherits from BaseTemplate and adds specific fields for structured data like title, description and tags.

### **Attributes**

* `title` (`list[str]`): List of title suggestions.
* `description` (`list[str]`): List of description suggestions.
* `document` (`list[str]`): List of document suggestions.
* `main_points` (`list[str]`): List of main point suggestions.
* `categories` (`list[str]`): List of category suggestions.
* `tags` (`list[str]`): List of tag suggestions.

### **Methods**

* `items`: Returns a list of tuples containing field information.


## FieldLabel
Represents a label associated with a structured field.

### **Attributes**

* `key` (`str`): The identifier of the label.
* `value` (`str`): The value associated with the label.
* `default` (`str`): The default label name.
* `custom` (`str`): A custom label name, overriding the default.

## Field
Represents a structured field with a key, value, and associated label.

### **Attributes**

* `key` (`str`): The field identifier.
* `value` (`str`): The field value.
* `label` (`FieldLabel`): An instance of `FieldLabel` providing metadata.
-
## Analysis
Holds detailed analysis-related metadata.

### **Attributes:**

* `language` (`Optional[str]`): The language of the content.
* `language_code` (`Optional[str]`): The code representing the language.
* `unigrams`, `bigrams`, `trigrams` (`Optional[Sequence[str]]`): Lists of unigrams, bigrams, and trigrams.
* `topic_value` (`Optional[str]`): The topic's computed value.
* `keyword_value` (`Optional[str]`): The keyword's computed value.

## Attr
Defines attributes for prompts and their metadata.

### **Attributes**

* `system_prompt` (`Optional[str]`): Defines the system prompt.
* `prompt` (`Optional[str]`): The core prompt content.
* `prompt_structure` (`Optional[str]`): A template or structure for the prompt.
* `instruction` (`Optional[str]`): Instructions for generating the prompt.
* `structure_fields` (`List[Field]`): A list of `Field` objects representing structured fields.
* `input` (`Optional[str]`): The input provided to the system.
* `output` (`Optional[str]`): The system's output.
* `analysis` (`Optional[Analysis]`): An `Analysis` object for content metadata.
* `is_masked` (`Optional[bool]`): Whether masking is applied to the prompt.
