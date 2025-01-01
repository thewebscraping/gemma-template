from __future__ import annotations

import asyncio
import json
from math import ceil
from pathlib import Path
from string import punctuation
from typing import (Callable, ClassVar, List, Literal, Optional, Sequence,
                    Union, get_origin)

import nest_asyncio
from datasets import Dataset, DatasetDict, load_dataset
from jinja2 import Environment
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from pydantic import alias_generators, model_validator
from tqdm import tqdm

from .constants import (GEMMA_PROMPT_TEMPLATE, GEMMA_TEMPLATE, INPUT_TEMPLATE,
                        INSTRUCTION_TEMPLATE, OUTPUT_TEMPLATE, PROMPT_TEMPLATE,
                        VIETNAMESE_INPUT_TEMPLATE,
                        VIETNAMESE_INSTRUCTION_TEMPLATE,
                        VIETNAMESE_OUTPUT_TEMPLATE, VIETNAMESE_PROMPT_TEMPLATE)
from .exceptions import DatasetError, MaxHiddenRatioError
from .utils import get_common_words, get_language, mask_hidden

nest_asyncio.apply()

JinjaTemplate = Environment()
TemplateTypes = Union["Template", str, Callable]
BULLET_STYLE_MAPPING = {
    None: " ",
    "dash": "-",
    "asterisk": "*",
    "blockquote": ">"
}


class Base(BaseModel):
    model_config = ConfigDict(alias_generator=alias_generators.to_snake, extra="allow")


class FieldLabel(Base):
    key: str = ""
    value: str = ""
    default: str = ""
    custom: str = ""

    @property
    def name(self):
        return self.custom or self.default

    def __str__(self):
        if self.custom:
            return "**{} ({}):** {}".format(self.custom, self.default, self.value)
        return "**{}:** {}".format(self.name, self.value)

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.key)


class Field(Base):
    key: str = ""
    value: str = ""
    label: FieldLabel = PydanticField(default_factory=FieldLabel)

    def __repr__(self):
        return "<%s: %s>" % self.__class__.__name__, self.key


class Analysis(Base):
    language: Optional[str] = ""
    language_code: Optional[str] = ""
    unigrams: Optional[Sequence[str]] = []
    bigrams: Optional[Sequence[str]] = []
    trigrams: Optional[Sequence[str]] = []
    topic_value: Optional[str] = ""
    keyword_value: Optional[str] = ""


class Attr(Base):
    system_prompt: Optional[str] = ""
    prompt: Optional[str] = ""
    prompt_structure: Optional[str] = ""
    instruction: Optional[str] = ""
    structure_fields: List[Field] = PydanticField(default=[])
    input: Optional[str] = ""
    output: Optional[str] = ""
    analysis: Optional[Analysis] = PydanticField(default_factory=Analysis)
    is_masked: Optional[bool] = None


class BaseTemplate(Base):
    """
    A foundational class that encapsulates common functionalities for structured fields.

    Attributes:
        end_sep: Default separator for concatenating values.

    Methods:
        _process_before: Preprocesses input data.
        _position_value: Retrieves a value from a list based on position.
    """

    _position_data: ClassVar[dict[str, int]] = {}

    end_sep: str = "and"

    @model_validator(mode="before")
    @classmethod
    def _process_before(cls, data: dict) -> dict:
        kwargs = {}
        for field, field_info in cls.model_fields.items():
            origin_type = get_origin(field_info.annotation)
            if isinstance(origin_type, (list, type, set)):
                cls._position_data[field] = 0

            if field in data:
                kwargs[field] = data[field]

        return kwargs

    def _position_value(
        self, field: str, *, position_data: dict = None, values: list[str] = ()
    ) -> str:
        output_str = ""
        field = str(field).lower().strip()
        values = values or getattr(self, field, None)
        position_data = position_data or self._position_data
        if values and field in position_data:
            try:
                output_str = values[position_data[field]]
            except IndexError:
                position_data[field] = 0
                output_str = values[0]

            position_data[field] += 1
        return output_str


class FieldPosition(BaseTemplate):
    """
    Inherits from BaseTemplate and adds specific fields for structured data like title, description and tags.

    Attributes:
        title: List of title suggestions.
        description: List of description suggestions.
        document: List of document suggestions.
        main_points: List of main point suggestions.
        categories: List of category suggestions.
        tags: List of tag suggestions.


    Methods:
        items: Returns a list of tuples containing field information.

    """  # noqa: 501

    _default_position_data: ClassVar[dict[str, int]] = {
        "title": 0,
        "description": 0,
        "document": 0,
        "main_points": 0,
        "categories": 0,
        "tags": 0,
    }
    _default_attrs: ClassVar[dict] = {
        "title": ["Title"],
        "description": [
            "Description",
            "Introduction",
            "Meta Description",
        ],
        "document": ["Article", "Edit Article"],
        "main_points": [
            "Main Points",
            "Key Points",
            "Highlights",
        ],
        "categories": ["Categories", "Topics"],
        "tags": ["Tags", "Keywords"],
    }

    title: Optional[list[str]] = []
    description: Optional[list[str]] = []
    document: Optional[list[str]] = []
    main_points: Optional[list[str]] = []
    categories: Optional[list[str]] = []
    tags: Optional[list[str]] = []

    def labels(self) -> list[FieldLabel]:
        labels = []
        for key, values in self._default_attrs.items():
            default = self._position_value(
                key,
                position_data=self._default_position_data,
                values=self._default_attrs.get(key, []) or [],
            ).title()
            labels.append(
                FieldLabel(
                    key=key,
                    default=default,
                    custom=self._position_value(key).title(),
                )
            )
        return labels


class Template(BaseTemplate):
    """
    Extends the BaseTemplate class to provide specialized functionality for generating structured prompts.
    This class combines system, user, and structural prompt templates to create flexible, multi-purpose
    content generation workflows.

    Attributes:
        template (list[TemplateTypes]): Base template for constructing the final prompt.
        input_template (list[TemplateTypes]): Input Template for user prompt.
        output_template (list[TemplateTypes]): Output Template for model prompt.
        instruction_template (list[TemplateTypes]): Instruction template for instruction prompt, if applicable.
        prompt_template (list[TemplateTypes]): Structure template for structure prompt, if applicable.
        system_prompts (list[str]): A list of system prompts, defining the role or behavior of the model.
        user_prompts (list[str]): A list of user prompts, specifying user queries or requests.
        title (list[str]): A collection of title prompts designed for SEO optimization and clear messaging.
        description (list[str]): Prompts for crafting compelling introductions or meta descriptions.
        document (list[str]): Prompts aimed at refining and enhancing the language of main content.
        main_points (list[str]): Prompts for summarizing or emphasizing main points.
        categories (list[str]): Prompts for identifying or refining article categories or themes.
        tags (list[str]): Templates for selecting or enhancing tags and tags for SEO.
        position (Optional[FieldPosition]): An instance of `FieldPosition` to manage structured fields in the prompt.

    Example Usage:
        >>> from gemma_template import Template, FieldPosition, INPUT_TEMPLATE, OUTPUT_TEMPLATE, INSTRUCTION_TEMPLATE, PROMPT_TEMPLATE
        >>> template_instance = Template(
        ...         instruction_template=[INSTRUCTION_TEMPLATE],  # Optional
        ...         prompt_template=[PROMPT_TEMPLATE],  # Optional
        ...         input_template=[INPUT_TEMPLATE],  # Optional
        ...         output_template=[OUTPUT_TEMPLATE],  # Optional
        ...         position=FieldPosition(
        ...             title=["Custom Title"],
        ...             description=["Custom Description"],
        ...             document=["Custom Article"],
        ...             main_points=["Custom Main Points"],
        ...             categories=["Custom Categories"],
        ...             tags=["Custom Tags"],
        ...    ),
        ... )   # Create fully customized structured reminders.
        >>> response = template_instance.apply_template(
        ...    title="Gemma open models",
        ...    description="Gemma: Introducing new state-of-the-art open models.",
        ...    main_points=["Main point 1", "Main point 2"],
        ...    categories=["Artificial Intelligence", "Gemma"],
        ...    tags=["AI", "LLM", "Google"],
        ...    document="Gemma open models are built from the same research and technology as Gemini models. Gemma 2 comes in 2B, 9B and 27B and Gemma 1 comes in 2B and 7B sizes.",
        ...    output="A new family of open language models demonstrating strong performance across academic benchmarks for language understanding, reasoning, and safety.",
        ...    max_hidden_words=.1,  # set 0 if you don't want to hide words.
        ...    min_chars_length=2,  # Minimum character of a word, used to create unigrams, bigrams, and trigrams. Default is 2.
        ...    max_chars_length=0,  # Maximum character of a word, used to create unigrams, bigrams and trigrams. Default is 0.
        ... )  # remove kwargs if not used.
        >>> print(response)
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

    """  # noqa: E501

    template: list[str] = [GEMMA_TEMPLATE]
    input_template: list[str] = [INPUT_TEMPLATE]
    output_template: list[str] = [OUTPUT_TEMPLATE]
    instruction_template: list[str] = [INSTRUCTION_TEMPLATE]
    prompt_template: list[str] = [PROMPT_TEMPLATE]
    system_prompts: list[str] = PydanticField(
        default=["You are a multilingual professional writer."]
    )
    prompts: list[str] = PydanticField(
        default=[
            "Rewrite the input text or document to be SEO-friendly. Include relevant keywords, optimize the title and subheadings, and ensure the text flows naturally for search engines and readers.",
            "Rewrite the input text or document with an authoritative tone, incorporating credible sources, data, and references to boost trustworthiness and SEO ranking.",
            "Rewrite the input text or document for a professional audience, focusing on technical details, industry-specific terminology, and actionable insights.",
            "Rewrite the input text or document to make it simpler and easier to understand for a general audience. Use clear and concise language while preserving the original meaning and key details.",
            "Reimagine the input text or document with a more engaging and creative tone. Add  metaphors, analogies, or storytelling elements to make it more captivating for readers.",
            "Rewrite the input text or document to make it more persuasive and compelling. Focus on strengthening arguments, appealing to emotions, and using rhetorical techniques to convince the reader.",
            "Rewrite the input text or document to suit a specific cultural or regional audience. Adjust idioms, references, and examples to resonate better with the target readers while keeping the core message intact.",
            "Rewrite the input text or document to highlight its unique value proposition while ensuring it ranks well for targeted keywords.",
        ]
    )
    title: list[str] = PydanticField(
        default=[
            "Rewrite the title to reflect the main keyword and topic.",
            "Rewrite the title to make it concise, memorable, and optimized for SEO.",
            "Create a title that is concise, clear, attention-grabbing, and SEO-optimized.",
            "Develop a title that is catchy, SEO-friendly, and accurately represents the subject matter.",
            "Revise the title to ensure it is keyword-rich, engaging, and easy to understand.",
            "Craft a title that clearly conveys the topic and is optimized for search engines.",
            "Rewrite the title to maximize clarity, appeal, and relevance to the content.",
            "Focus on a surprising or unique angle in the title. Include numbers or statistics in the title for specificity.",
            "Create a title that complements the title but adds more detail. Make the title conversational to draw readers in.",
            "Incorporate trending keywords or phrases into the title. Ensure the title is relevant and closely tied to the content.",
            "Rewrite the title to make it concise, clear, and SEO-optimized.",
            "Add power words to the title to evoke curiosity or emotion.",
            "Focus on the benefits in the title to attract attention.",
            "Use action verbs to create an engaging and dynamic title.",
            "Rewrite the title to reflect the main keyword and topic.",
        ]
    )
    description: list[str] = PydanticField(
        default=[
            "Rewrite the description with a bold claim or statistic to grab attention.",
            "Write description of the article in one or two sentences while focusing on reader benefits and engage curiosity.",
            "Begin the description with an engaging anecdote or story for SEO optimization.",
            "Rewrite the description to highlight a surprising fact or unique insight that intrigues the reader.",
            "Craft a description in one or two sentences that emphasizes the value readers will gain from the article.",
            "Begin the description with a thought-provoking question to spark curiosity and encourage clicks.",
            "Write a description that starts with an actionable tip or advice to immediately engage the audience.",
            "Create a description focusing on how the article addresses a common problem or challenge readers face.",
            "Rewrite the description with language that appeals to emotions, inspiring readers to explore further.",
        ]
    )
    document: list[str] = PydanticField(
        default=[
            "Rewrite the input text or document with an authoritative tone, incorporating credible sources, data, and references to boost trustworthiness and SEO ranking.",
            "Rewrite the input text or document for a professional audience, focusing on technical details, industry-specific terminology, and actionable insights.",
            "Rewrite the input text or document to make it simpler and easier to understand for a general audience. Use clear and concise language while preserving the original meaning and key details.",
            "Reimagine the input text or document with a more engaging and creative tone. Add  metaphors, analogies, or storytelling elements to make it more captivating for readers.",
            "Rewrite the input text or document to make it more persuasive and compelling. Focus on strengthening arguments, appealing to emotions, and using rhetorical techniques to convince the reader.",
            "Rewrite the input text or document to suit a specific cultural or regional audience. Adjust idioms, references, and examples to resonate better with the target readers while keeping the core message intact.",
            "Rewrite the input text or document to highlight its unique value proposition while ensuring it ranks well for targeted keywords.",
            "Rewrite the input text or document to be SEO-friendly. Include relevant keywords, optimize the title and subheadings, and ensure the text flows naturally for search engines and readers.",
        ]
    )
    main_points: list[str] = PydanticField(
        default=[
            "Summarize the main ideas into concise, actionable key points for added context to make them more engaging.",
            "Simplify the original key points to make them clearer and more reader-friendly.",
            "Ensure all key points flow logically from one to the next.",
            "Summarize the key takeaways from this text in main points, ensuring clarity and conciseness.",
            "Generate a summary document that distills the central themes and supporting key points from this text.",
            "Rewrite key points to be more concise and actionable.",
            "Group related key points for better organization.",
            "Add examples or brief explanations to each key point.",
            "Simplify complex ideas into easily digestible key points.",
            "Rewrite key points as questions to make them more engaging.",
            "Ensure all key points flow logically from one to the next.",
            "Turn abstract concepts into concrete actions in the key points.",
        ]
    )
    categories: list[str] = PydanticField(
        default=[
            "Assign appropriate categories to the article based text or target audience.",
            "Rewrite categories to align with industry standards or popular topics.",
            "Use categories that align with similar articles on the topic and improve SEO and discoverability.",
            "Assign categories that reflect the main themes of the article.",
            "Rewrite categories to align with industry standards or popular topics.",
            "Focus on broad yet specific categories for better organization.",
            "Ensure the categories reflect the target audience’s interests.",
            "Rewrite categories to match keywords used in the article.",
            "Choose categories that improve SEO and discoverability.",
            "Use categories that align with similar articles on the topic.",
            "Avoid overly broad or vague categories by being specific.",
            "Rewrite categories to highlight the article’s primary focus areas.",
        ]
    )
    tags: list[str] = PydanticField(
        default=[
            "Rewrite tags to include relevant keywords in about 5 keywords.",
            "Add trending terms or phrases to the tags for increased visibility from 3 to 5 keywords.",
            "Use 5 keywords or tags that reflect the article’s subtopics or themes.",
            "Ensure the tags align with popular search queries under 5 keywords.",
            "Rewrite 5 tags or keywords to make them more specific and targeted.",
            "Match tags to similar content for cross-promotion opportunities under 5 keywords.",
        ]
    )

    position: Optional[FieldPosition] = PydanticField(default_factory=FieldPosition)

    @model_validator(mode="after")
    def _process_after(self):
        def _normalize(sentences: list[str], append: str = ".") -> list[str]:
            if sentences:
                outputs = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence[-1] in punctuation:
                        sentence = sentence[:-1]

                    outputs.append(sentence + append)
                return list(set(outputs))
            return []

        self.system_prompts = _normalize(self.system_prompts)
        self.prompts = _normalize(self.prompts, ".")
        self.title = _normalize(self.title)
        self.description = _normalize(self.description)
        self.document = _normalize(self.document)
        self.main_points = _normalize(self.main_points)
        self.categories = _normalize(self.categories)
        self.tags = _normalize(self.tags)
        return self

    def load_dataset(
        self,
        fp: Union[str, Path, list[dict], Dataset, DatasetDict],
        *,
        output_format: Union[str, Literal["text", "alpaca", "openai"]] = "text",
        excluded_fields: Optional[Sequence[str]] = (),
        max_hidden_ratio: Union[float] = 0,
        max_hidden_words: Union[int, float] = 0,
        min_chars_length: int = 2,
        max_chars_length: int = 0,
        max_concurrency: int = 4,
        is_remove_data: bool = True,
        is_close_async_loop: bool = True,
        **kwargs,
    ) -> Union[Dataset, DatasetDict]:
        """
        Processes and loads a dataset, generating prompts based on the provided templates.

        This function supports various input formats such as file paths, dictionaries, or Hugging Face Dataset objects.
        It uses templates to create structured prompts and supports concurrent processing for efficiency.

        Args:
            fp (Union[str, list[dict], Dataset, DatasetDict]):
                Input data as a file path, a list of dictionaries, or a Hugging Face Dataset/DatasetDict object.
            output_format (Union[str, Literal["text", "alpaca", "openai"]]):
                Specifies the format for the generated prompts. Default is "text".
            excluded_fields (Optional[Sequence[str]]):
                Fields excluded to response. Default is empty sequence.
            max_hidden_ratio (Union[float]):
                Percentage of documents that need to be word masked. Min: 0, Max: 1. Default: 0.
            max_hidden_words (Optional[str]):
                Replace words in the document with '____'. The `max_hidden` parameter must be greater than 0.
                Use `int`: exact number of words to be masked, `float`: percentage of number of words to be masked.
            min_chars_length (int):
                Minimum character of a word, used to create unigrams, bigrams, and trigrams. Default is 2.
            max_chars_length (int):
                Maximum character of a word, used to create unigrams, bigrams and trigrams. Default is 0.
            max_concurrency (int):
                Maximum number of concurrent threads for processing data. Default is 4.
            is_remove_data (bool):
                True will remove the original data from the dataset, otherwise it will keep the field as `data` in the dataset.
            is_close_async_loop (bool):
                By default it will close the asyncio event loop every time I finish processing the dataset data.
                Although it has handled the `RuntimeError` exception. However, you should set it to False if running on Kaggle Notebooks and Colab.
            **kwargs: Additional parameters, including:
                - `token` (Optional[str]): Hugging Face authentication token.
                - `split` (Optional[list[str]]): Dataset split for Hugging Face Dataset loading.
                - `additional parameters` see also: `Template.template`.

        Returns:
            Dataset: A Hugging Face Dataset or DatasetDict object containing the processed prompts.

        Raises:
            DatasetError: If the input data type is not supported or the `max_hidden_ratio` value is incorrect.

        Example:
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
            dataset = prompt_instance.load_dataset(data_dict, output_format='text')   # enum: text, alpaca and openai.
            print(dataset['text'][0])
            ```
        """  # noqa: E501

        async def create_task(config, max_hidden_count: int = 0, hidden_count: int = 0):
            async with semaphore:
                config.update(kwargs)
                config.update(
                    dict(
                        min_chars_length=min_chars_length,
                        max_chars_length=max_chars_length,
                        excluded_fields=excluded_fields,
                        is_remove_data=is_remove_data,
                    )
                )
                if max_hidden_ratio > 0 and hidden_count < max_hidden_count:
                    config["max_hidden_words"] = max_hidden_words
                else:
                    config["max_hidden_words"] = 0

                if output_format == "alpaca":
                    items.append(
                        self.to_alpaca(**config,)
                    )
                elif output_format == "openai":
                    items.append(
                        self.to_openai(**config,)
                    )
                else:
                    items.append(
                        self.to_text(**config,)
                    )

                pbar.update(1)
                hidden_count += 1

        async def run_task(ds):
            max_hidden_count = ceil(len(ds) * max_hidden_ratio)
            await asyncio.wait(
                [
                    loop.create_task(create_task(config, max_hidden_count, idx))
                    for idx, config in enumerate(ds)
                ]
            )

        def _close():
            """Closed Asyncio event loop"""
            if is_close_async_loop:
                try:
                    loop.close()
                except RuntimeError:
                    pass

        if max_hidden_ratio:
            if 0 > max_hidden_ratio > 1:
                raise MaxHiddenRatioError(
                    "Maximum hidden ratio must be between 0 and 1."
                )

        dataset = fp
        if isinstance(dataset, (str, Path)):
            fp = Path(dataset)
            if fp.exists():
                try:
                    with open(dataset, "r", encoding="utf-8") as fp:
                        dataset = json.load(fp)

                except json.decoder.JSONDecodeError:
                    pass
            else:
                dataset = load_dataset(
                    **dict(
                        path=dataset,
                        split=kwargs.get("split", None),
                        token=kwargs.get("token", None),
                    )
                )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)
        semaphore = asyncio.Semaphore(max_concurrency)
        if isinstance(dataset, (list, Dataset)):
            items = []
            with tqdm(total=len(dataset)) as pbar:
                _ = loop.run_until_complete(run_task(dataset))

            _close()
            return Dataset.from_list(items)

        elif isinstance(dataset, DatasetDict):
            mapping = {}
            with tqdm(total=len(dataset)) as pbar:
                for field in dataset.column_names:
                    items = []
                    _ = loop.run_until_complete(run_task(dataset[field]))
                    mapping[field] = Dataset.from_list(items)

            _close()
            return DatasetDict(mapping)

        raise DatasetError("Invalid dataset type.")

    def to_text(self, **kwargs) -> dict:
        """
        Generate Text Format

        Args:
            **kwargs: see also `Template.apply_template`.
        """

        input_str, output_str, attr = self._build_template(**kwargs)

        text = JinjaTemplate.from_string(self._position_value("template")).render(
            input=input_str,
            output=output_str,
        )

        return dict(
            text=text,
            analysis=attr.analysis.model_dump(mode="json"),
            is_masked=attr.is_masked,
            origin_data=self._get_origin_data(**kwargs),
        )

    def to_alpaca(self, **kwargs) -> dict:
        """
        Generate Alpaca Format

        Args:
            **kwargs: see also `Template.apply_template`.
        """

        input_str, output_str, attr = self._build_template(**kwargs)

        return dict(
            instruction="\n\n".join(
                [
                    p
                    for p in [
                        attr.system_prompt,
                        attr.instruction,
                    ]
                    if p.strip()
                ]
            ).strip(),
            input="\n\n".join(
                [
                    p
                    for p in [
                        attr.prompt_structure or attr.prompt,
                        attr.input,
                    ]
                    if p.strip()
                ]
            ).strip(),
            output=attr.output,
            analysis=attr.analysis.model_dump(mode="json"),
            is_masked=attr.is_masked,
            origin_data=self._get_origin_data(**kwargs),
        )

    def to_openai(self, **kwargs) -> dict:
        """
        Generate Open AI Format

        Args:
            **kwargs: see also `Template.apply_template`.
        """

        input_str, output_str, attr = self._build_template(**kwargs)

        return dict(
            messages=[
                {
                    "role": "developer",
                    "content": "\n\n".join(
                        [
                            p
                            for p in [
                                attr.system_prompt,
                                attr.instruction,
                            ]
                            if p.strip()
                        ]
                    ).strip()
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            p
                            for p in [
                                attr.prompt_structure or attr.prompt,
                                attr.input,
                            ]
                            if p.strip()
                        ]
                    ).strip()
                },
                {"role": "assistant", "content": output_str},
            ],
            analysis=attr.analysis.model_dump(mode="json"),
            is_masked=attr.is_masked,
            origin_data=self._get_origin_data(**kwargs),
        )

    def apply_template(
        self,
        **kwargs,
    ):
        """
        Generates a complete prompt by integrating system, user, and structural elements.

        Args:
            **kwargs: Additional parameters including:
                - output: Optional[str] = Model response output same as document field.
                - title: Optional[list[str]] = List of title to include in the prompt.
                - description: Optional[list[str]] = List of description to include in the prompt.
                - document: Optional[list[str]] = The main text content or article to be processed.
                - main_points: Optional[list[str]] = List of main points to include in the prompt.
                - categories: Optional[list[str]] = List of categories/categories to include in the prompt.
                - tags: Optional[list[str]] = List of tags/tags to include in the prompt.
                - bullet_style: (Optional[Literal['dash', 'number', 'asterisk']]):
                    Bullet list style start dash, asterisk, number or blockquote. Default is asterisk.
                - additional parameters: see also `Template.template`.

        Returns:
            str: A formatted prompt string combining multiple components.

        Example:
            >>> template_instance = Template(...)
            >>> response = template_instance.apply_template(
            ...     document="Sample document",
            ...     output="Generated response",
            ... )
            >>> print(response)
        """  # noqa: E501

        input_str, output_str, _ = self._build_template(**kwargs)
        return JinjaTemplate.from_string(self._position_value("template")).render(
            input=input_str,
            output=output_str,
        )

    def generate_prompt(
        self,
        input_template: Optional[TemplateTypes] = INPUT_TEMPLATE,
        *,
        prompt_template: Optional[TemplateTypes] = GEMMA_PROMPT_TEMPLATE,
        **kwargs,
    ):
        """Generates a prompt to predict."""

        template_attr = self.get_template_attr(**kwargs)
        if isinstance(prompt_template, Callable):
            return prompt_template(self, template_attr)

        if isinstance(input_template, Callable):
            input_str = input_template(self, template_attr)
        else:
            input_str = JinjaTemplate.from_string(input_template).render(
                **template_attr.model_dump(mode="json")
            )

        return JinjaTemplate.from_string(prompt_template).render(input=input_str)

    def generate_user_prompt(
        self,
        **kwargs,
    ) -> str:
        """
        Generates a user-specific prompt by combining multiple user-defined inputs.

        This method collects and formats user prompts, optionally including structural elements, to create
        a coherent and complete prompt.

        Args:
            **kwargs: see also `Template.template`.

        Returns:
            str: A formatted string combining all user-defined prompts.

        Example:
            >>> prompt_instance = Template()
            >>> response = prompt_instance.generate_user_prompt(
            ...     document="Sample document",
            ... )
            >>> print(response)
        """  # noqa: E501

        attr = self.get_template_attr(**kwargs)
        return JinjaTemplate.from_string(self._position_value("input_template")).render(
            **attr.model_dump(mode="json")
        )

    def generate_model_prompt(
        self,
        **kwargs,
    ) -> str:
        """
        Generates a model-specific prompt by formatting output and document content according to structural rules.

        This method organizes output content and any additional fields into a structured format
        for model processing.

        Args:
            **kwargs: See also `Template.template`.

        Returns:
            str: A structured prompt string tailored for model input.

        Example:
            >>> template_instance = Template()
            >>> response = template_instance.generate_model_prompt(
            ...     document="Sample document",
            ...     output="Generated response",
            ... )
            >>> print(response)
        """  # noqa: E501

        return self._build_output(
            self._build_structure_fields(**kwargs), **kwargs
        )

    def get_template_attr(self, **kwargs) -> Attr:
        """
        Generates an kwargs for the Attr

        This method customizes the instruction prompt by combining system prompts, user prompts, and optional structural prompts.
        It allows for dynamic language customization and response structure formatting.

        Args:
            **kwargs: see also `Template.template`.

        Returns:
            Attr: Attr instance.
        """  # noqa: E501

        document = kwargs.pop("document", "")
        analysis = self._build_analytics(document, **kwargs)
        document = mask_hidden(document, language_code=analysis.language_code, **kwargs)
        system_prompt = self._position_value("system_prompts").strip()
        prompt = self._position_value("prompts").strip()
        structure_fields = self._build_structure_fields(document=document, **kwargs)
        return Attr.model_validate(
            dict(
                system_prompt=system_prompt,
                prompt=prompt,
                prompt_structure=self._build_prompt_structure(structure_fields, prompt, **kwargs),
                instruction=self._build_instruction(document, analysis, **kwargs),
                structure_fields=structure_fields,
                input=document,
                output=self._build_output(structure_fields, **kwargs),
                is_masked=bool(kwargs.get("max_hidden_words")),
                analysis=analysis,
            )
        )

    def _build_template(self, **kwargs) -> tuple[str, str, Attr]:
        attr = self.get_template_attr(**kwargs)
        template_kwargs = attr.model_dump(mode="json")
        input_str = JinjaTemplate.from_string(
            self._position_value("input_template")
        ).render(**template_kwargs)
        output_str = JinjaTemplate.from_string(
            self._position_value("output_template")
        ).render(**template_kwargs)
        return input_str.strip(), output_str.strip(), attr

    def _build_output(
        self,
        structure_fields: list[Field] = (),
        **kwargs,
    ):
        """Build Output Template"""

        output_str = kwargs.get("output", "")
        excluded_fields = kwargs.pop("excluded_fields", []) or []
        for item in structure_fields:
            if item.key in excluded_fields:
                continue

            if item.key == "document":
                item.value = output_str

        return JinjaTemplate.from_string(
            self._position_value("output_template")
        ).render(
            structure_fields=[
                item for item in structure_fields if item.key not in excluded_fields
            ],
            output=output_str,
        )

    def _build_instruction(
        self,
        document: str,
        analysis: Analysis,
        **kwargs,
    ) -> str:
        if self.instruction_template:
            template = JinjaTemplate.from_string(
                self._position_value("instruction_template")
            )
            return template.render(
                document=document,
                topic_value=analysis.topic_value,
                keyword_value=analysis.keyword_value,
                unigrams=analysis.unigrams,
                bigrams=analysis.bigrams,
                trigrams=analysis.trigrams,
                language=analysis.language,
                **kwargs,
            )
        return ""

    def _build_prompt_structure(
        self,
        structure_fields: list[Field],
        prompt: str = "",
        **kwargs,
    ) -> str:
        if self.prompt_template:
            excluded_fields = kwargs.get("excluded_fields", [])
            return (
                JinjaTemplate.from_string(self._position_value("prompt_template"))
                .render(prompt=prompt, structure_fields=[field for field in structure_fields if field.key not in excluded_fields])
                .strip()
            )
        return ""

    def _build_analytics(self, document: str, **kwargs) -> Analysis:
        language_code = "auto"
        language = kwargs.get("language")
        if language is None:
            language_code, language = get_language(document)

        unigrams, bigrams, trigrams = self._get_n_grams(
            document,
            number_common_words=kwargs.get("number_common_words", 5),
            language_code=language_code,
            **kwargs,
        )

        return Analysis(
            unigrams=unigrams,
            bigrams=bigrams,
            trigrams=trigrams,
            language=language,
            language_code=language_code,
            topic_value=", ".join(kwargs.get("categories", []) or []),
            keyword_value=", ".join(kwargs.get("tags", []) or []),
        )

    def _build_structure_fields(
        self, bullet_style: str = "asterisk", **kwargs
    ) -> list[Field]:
        structure_fields = []
        for label in self.position.labels():
            label.value = self._position_value(label.key)
            if label.key not in kwargs:
                continue

            if isinstance(kwargs[label.key], list):
                value = self._generate_bullet_style(kwargs[label.key], bullet_style)
            else:
                value = kwargs[label.key]

            structure_fields.append(
                Field(
                    key=label.key,
                    value=value,
                    label=label,
                )
            )

        return structure_fields

    def _get_n_grams(
        self,
        document: str = "",
        number_common_words: int = 5,
        language_code: str = "auto",
        **kwargs,
    ) -> tuple[Sequence[str], Sequence[str], Sequence[str]]:
        if not document.strip():
            return [], [], []

        unigrams = self._get_common_words(
            document,
            n=1,
            response_n=number_common_words,
            language_code=language_code,
            **kwargs,
        )
        bigrams = self._get_common_words(
            document,
            n=2,
            response_n=number_common_words,
            language_code=language_code,
            excluded_words=unigrams,
        )
        trigrams = self._get_common_words(
            document,
            n=3,
            response_n=number_common_words,
            language_code=language_code,
            excluded_words=unigrams,
        )
        return unigrams, bigrams, trigrams

    def _get_common_words(
        self,
        document: str,
        *,
        n: int = 1,
        response_n: int = 10,
        language_code: str = None,
        min_chars_length: int = 2,
        max_chars_length: int = 0,
        excluded_words: list[str] = (),
        **kwargs,
    ) -> list[str]:
        if not language_code:
            language_code, _ = get_language(document)

        return [
            word
            for word in get_common_words(
                document,
                n=n,
                response_n=response_n,
                language_code=language_code,
                excluded_words=excluded_words,
            )
        ]

    def _generate_bullet_style(self, words: list[str], bullet_style: str = None) -> str:
        if words:
            if bullet_style is None:
                return "\n".join(words)

            return "\n".join(
                [
                    "{} {}".format(
                        f"{idx+1}." if bool(bullet_style == "number") else BULLET_STYLE_MAPPING.get(bullet_style, ""),
                        word.strip(),
                    )
                    for idx, word in enumerate(words)
                ]
            ).strip()
        return ""

    def _get_origin_data(self, **kwargs) -> dict:
        if kwargs.get("is_remove_data", True) is False:
            return {k: v for k, v in kwargs.items() if hasattr(self, k)}
        return {}


gemma_template = Template()
vietnamese_gemma_template = Template(
    input_template=[VIETNAMESE_INPUT_TEMPLATE],
    output_template=[VIETNAMESE_OUTPUT_TEMPLATE],
    instruction_template=[VIETNAMESE_INSTRUCTION_TEMPLATE],
    prompt_template=[VIETNAMESE_PROMPT_TEMPLATE],
    end_sep="và",
    system_prompts=[
        (
            "Bạn là một nhà sáng tạo nội dung, viết nội dung chuyên nghiệp biết nhiều"
            " ngôn ngữ."
        ),
    ],
    prompts=[
        "Viết lại nội dung này để thân thiện với SEO. Bao gồm các từ khóa có liên quan, tối ưu hóa tiêu đề và tiêu đề phụ, và đảm bảo văn bản trôi chảy tự nhiên cho các công cụ tìm kiếm và người đọc.",
        "Viết lại bài viết này để làm cho nó đơn giản hơn và dễ hiểu hơn đối với đối tượng chung. Sử dụng ngôn ngữ rõ ràng và súc tích trong khi vẫn giữ nguyên ý nghĩa ban đầu và các chi tiết chính.",
        "Tái hiện bài viết này với giọng điệu hấp dẫn và sáng tạo hơn. Thêm phép ẩn dụ, phép so sánh hoặc các yếu tố kể chuyện để làm cho nó hấp dẫn hơn đối với người đọc.",
        "Viết lại bài viết này để làm cho nó thuyết phục và hấp dẫn hơn. Tập trung vào việc củng cố các lập luận, thu hút cảm xúc và sử dụng các kỹ thuật tu từ để thuyết phục người đọc.",
        "Viết lại bài viết này để phù hợp với đối tượng cụ thể về văn hóa hoặc khu vực. Điều chỉnh thành ngữ, tài liệu tham khảo và ví dụ để tạo được tiếng vang tốt hơn với độc giả mục tiêu trong khi vẫn giữ nguyên thông điệp cốt lõi.",
        "Viết lại bài viết này để làm nổi bật đề xuất giá trị độc đáo của nó trong khi đảm bảo nó được xếp hạng tốt cho các từ khóa mục tiêu.",
        "Viết lại nội dung này với giọng điệu có thẩm quyền, kết hợp các nguồn, dữ liệu và tài liệu tham khảo đáng tin cậy để tăng độ tin cậy và thứ hạng SEO.",
        "Viết lại bài viết này để đối tượng chuyên nghiệp, tập trung vào các chi tiết kỹ thuật, thuật ngữ chuyên ngành và thông tin chi tiết có thể thực hiện được.",
    ],
    position=FieldPosition(
        title=["Tiêu đề"],
        description=["Mô tả"],
        document=["Bài viết chỉnh sửa"],
        main_points=["Điểm nổi bật", "Điểm chính"],
        categories=["Danh mục", "Chủ đề"],
        tags=["Từ khoá"],
    ),
    title=[
        "Viết lại tiêu đề để phản ánh từ khóa và chủ đề chính.",
        "Viết lại tiêu đề để làm cho nó ngắn gọn, dễ nhớ và được tối ưu hóa cho SEO.",
        "Tạo một tiêu đề ngắn gọn, rõ ràng, thu hút sự chú ý và được tối ưu hóa cho SEO.",
        "Phát triển một tiêu đề hấp dẫn, thân thiện với SEO và thể hiện chính xác nội dung.",
        "Sửa đổi tiêu đề để đảm bảo từ khóa liên quan, hấp dẫn và dễ hiểu.",
        "Soạn một tiêu đề truyền tải rõ ràng chủ đề và được tối ưu hóa cho các công cụ tìm kiếm.",
        "Viết lại tiêu đề để tối đa hóa sự rõ ràng, hấp dẫn và liên quan đến nội dung.",
        "Tạo một tiêu đề bổ sung cho tiêu đề nhưng thêm chi tiết hơn. Làm cho tiêu đề mang tính đối thoại để thu hút người đọc.",
        "Tập trung vào một góc độ gây ngạc nhiên hoặc độc đáo trong tiêu đề. Bao gồm các con số hoặc số liệu thống kê trong tiêu đề để tạo sự cụ thể.",
        "Kết hợp các từ khóa hoặc cụm từ thịnh hành vào tiêu đề. Đảm bảo tiêu đề có liên quan và gắn chặt với nội dung.",
        "Viết lại tiêu đề sao cho ngắn gọn, rõ ràng và tối ưu hóa SEO.",
        "Thêm từ ngữ mạnh mẽ vào tiêu đề để gợi sự tò mò hoặc cảm xúc.",
        "Tập trung vào những lợi ích trong tiêu đề để thu hút sự chú ý.",
        "Sử dụng động từ hành động để tạo tiêu đề hấp dẫn và năng động.",
        "Viết lại tiêu đề để phản ánh từ khóa và chủ đề chính.",
    ],
    description=[
        "Viết lại phần mô tả bằng một tuyên bố hoặc số liệu thống kê táo bạo để thu hút sự chú ý.",
        "Viết phần mô tả bài viết trong một hoặc hai câu, đồng thời tập trung vào lợi ích của người đọc và khơi gợi sự tò mò.",
        "Bắt đầu phần mô tả bằng một giai thoại hoặc câu chuyện hấp dẫn để tối ưu hóa SEO.",
        "Viết lại phần mô tả để làm nổi bật một sự thật đáng ngạc nhiên hoặc hiểu biết độc đáo khiến người đọc tò mò.",
        "Soạn thảo phần mô tả trong một hoặc hai câu, nhấn mạnh giá trị mà người đọc sẽ nhận được từ bài viết.",
        "Bắt đầu phần mô tả bằng một câu hỏi gợi mở để khơi dậy sự tò mò và khuyến khích nhấp chuột.",
        "Viết phần mô tả bắt đầu bằng một mẹo hoặc lời khuyên hữu ích để thu hút ngay lập tức đối tượng mục tiêu.",
        "Tạo phần mô tả tập trung vào cách bài viết giải quyết một vấn đề hoặc thách thức phổ biến mà người đọc gặp phải.",
        "Viết lại phần mô tả bằng ngôn ngữ khơi gợi cảm xúc, truyền cảm hứng cho người đọc khám phá sâu hơn.",
    ],
    document=[
        "Viết lại bài viết này để đối tượng chuyên nghiệp, tập trung vào các chi tiết kỹ thuật, thuật ngữ chuyên ngành và thông tin chi tiết có thể thực hiện được.",
        "Viết lại nội dung này với giọng điệu có thẩm quyền, kết hợp các nguồn, dữ liệu và tài liệu tham khảo đáng tin cậy để tăng độ tin cậy và thứ hạng SEO.",
        "Viết lại bài viết này để làm nổi bật đề xuất giá trị độc đáo của nó trong khi đảm bảo nó được xếp hạng tốt cho các từ khóa mục tiêu.",
        "Viết lại bài viết này để phù hợp với đối tượng cụ thể về văn hóa hoặc khu vực. Điều chỉnh thành ngữ, tài liệu tham khảo và ví dụ để tạo được tiếng vang tốt hơn với độc giả mục tiêu trong khi vẫn giữ nguyên thông điệp cốt lõi.",
        "Viết lại bài viết này để làm cho nó thuyết phục và hấp dẫn hơn. Tập trung vào việc củng cố các lập luận, thu hút cảm xúc và sử dụng các kỹ thuật tu từ để thuyết phục người đọc.",
        "Tái hiện bài viết này với giọng điệu hấp dẫn và sáng tạo hơn. Thêm phép ẩn dụ, phép so sánh hoặc các yếu tố kể chuyện để làm cho nó hấp dẫn hơn đối với người đọc.",
        "Viết lại bài viết này để làm cho nó đơn giản hơn và dễ hiểu hơn đối với đối tượng chung. Sử dụng ngôn ngữ rõ ràng và súc tích trong khi vẫn giữ nguyên ý nghĩa ban đầu và các chi tiết chính.",
        "Viết lại nội dung này để thân thiện với SEO. Bao gồm các từ khóa có liên quan, tối ưu hóa tiêu đề và tiêu đề phụ, và đảm bảo văn bản trôi chảy tự nhiên cho các công cụ tìm kiếm và người đọc.",
    ],
    main_points=[
        "Tóm tắt các ý chính thành các điểm chính ngắn gọn, có thể hành động để thêm ngữ cảnh nhằm khiến chúng hấp dẫn hơn.",
        "Đơn giản hóa các điểm chính ban đầu để làm cho chúng rõ ràng hơn và thân thiện hơn với người đọc.",
        "Đảm bảo tất cả các điểm chính đều có mạch lạc hợp lý từ điểm này sang điểm khác.",
        "Tóm tắt những điểm chính từ văn bản này thành các điểm chính, đảm bảo tính rõ ràng và súc tích.",
        "Tạo một tài liệu tóm tắt chắt lọc các chủ đề chính và các điểm chính hỗ trợ từ văn bản này.",
        "Viết lại các điểm chính để súc tích hơn và dễ thực hiện hơn.",
        "Nhóm các điểm chính liên quan để tổ chức tốt hơn.",
        "Thêm ví dụ hoặc giải thích ngắn gọn cho mỗi điểm chính.",
        "Đơn giản hóa các ý tưởng phức tạp thành các điểm chính dễ hiểu.",
        "Viết lại các điểm chính dưới dạng câu hỏi để làm cho chúng hấp dẫn hơn.",
        "Đảm bảo tất cả các điểm chính đều có sự liên kết hợp lý từ điểm này sang điểm khác.",
        "Biến các khái niệm trừu tượng thành các hành động cụ thể trong các điểm chính.",
    ],
    categories=[
        "Viết lại các danh mục để phù hợp với chủ đề phổ biến theo bài viết.",
        "Tạo danh sách danh mục để phù hợp với các từ khóa được sử dụng trong bài viết.",
        "Chọn các danh mục cải thiện SEO và khả năng khám phá theo nội dung bài viết.",
        "Chỉ định các danh mục phản ánh chủ đề chính của bài viết.",
        "Viết lại các danh mục để phù hợp với các tiêu chuẩn của ngành hoặc các chủ đề phổ biến.",
        "Tập trung vào các danh mục rộng nhưng cụ thể để tổ chức tốt hơn.",
        "Đảm bảo các danh mục phản ánh sở thích của đối tượng mục tiêu.",
        "Viết lại các danh mục để phù hợp với các từ khóa được sử dụng trong bài viết.",
        "Chọn các danh mục cải thiện SEO và khả năng khám phá.",
        "Sử dụng các danh mục phù hợp với các bài viết tương tự về chủ đề này.",
        "Tránh các danh mục quá rộng hoặc mơ hồ bằng cách cụ thể.",
        "Viết lại các danh mục để làm nổi bật các lĩnh vực trọng tâm chính của bài viết.",
    ],
    tags=[
        "Tạo danh sách 5 từ khóa thịnh hành giúp SEO tốt hơn.",
        "Tạo danh sách 5 từ khóa có liên quan phù hợp với truy vấn tìm kiếm phổ biến.",
        "Tập trung vào các từ khóa phổ biến trong bài viết để SEO tốt hơn dưới 5 từ khoá.",
        "Viết lại 3 đến 5 từ khóa để bao gồm các từ khóa có liên quan.",
        "Thêm các cụm từ khóa thịnh hành để tăng khả năng hiển thị trong khoảng 3 đến 5 từ khoá.",
        "Sử dụng các từ khóa phản ánh các chủ đề phụ hoặc chủ đề của bài viết trong phạm vi dưới 5 từ khoá.",
        "Đảm bảo các từ khóa phù hợp với các truy vấn tìm kiếm phổ biến dưới 5 từ khoá.",
        "Viết lại từ 3 đến 5 từ khóa để làm cho chúng cụ thể và có mục tiêu hơn.",
        "Tạo 5 từ khóa phù hợp với nội dung tương tự để có cơ hội quảng cáo chéo.",
    ],
)
