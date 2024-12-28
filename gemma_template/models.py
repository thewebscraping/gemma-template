from __future__ import annotations

import asyncio
import json
from pathlib import Path
from string import punctuation
from typing import Callable, ClassVar, Literal, Optional, Union, get_origin

import nest_asyncio
from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm

from .constants import (GEMMA_TEMPLATE, INSTRUCTION_TEMPLATE,
                        STRUCTURE_TEMPLATE, USER_TEMPLATE)
from .types import TemplateTypes
from .utils import get_frequently_words, get_language

nest_asyncio.apply()


class BaseTemplate(BaseModel):
    """
    A foundational class that encapsulates common functionalities for structured fields.

    Attributes:
        end_sep: Default separator for concatenating values.

    Methods:
        _process_before: Preprocesses input data.
        _get_value_by_position: Retrieves a value from a list based on position.
        _flatten_values: Flattens list values into a string format.
    """

    _positions: ClassVar[dict[str, int]] = {}
    _exclude_process_fields: ClassVar[list[str]] = ["end_sep", "language"]

    end_sep: str = "and"

    @model_validator(mode="before")
    @classmethod
    def _process_before(cls, data: dict) -> dict:
        if not cls._positions:
            for field, field_info in cls.model_fields.items():
                origin_type = get_origin(field_info.annotation)
                if isinstance(origin_type, (list, type, set)):
                    cls._positions[field] = 0

        kwargs = {k: [v] if isinstance(v, str) else v for k, v in data.items() if v}
        for exclude_field in cls._exclude_process_fields:
            if exclude_field in data:
                kwargs[exclude_field] = data[exclude_field]

        return kwargs

    def _get_value_by_position(
        self, field: str, *, positions: Optional[dict] = None, values: list[str] = None
    ) -> str:
        positions = self._positions if not isinstance(positions, dict) else positions
        field = str(field).lower().strip()
        value = ""
        if field in positions:
            pos = positions[field]
            values = values or getattr(self, field)
            if values:
                try:
                    value = values[pos]
                except IndexError:
                    pos = 0
                    value = values[pos]

            positions[field] = pos + 1
        return value

    def _flatten_values(
        self, value: Union[str, list[str]], *, end_sep: str = None, **kwargs
    ):
        if isinstance(value, list):
            values = [str(v) for v in value]
        else:
            values = [str(value)] if value else ""

        end_sep = " ".join(["", end_sep or self.end_sep.strip(), ""])
        flatten_value = values[0]
        if len(values) == 2:
            flatten_value = end_sep.join(values)
        elif len(values) > 2:
            flatten_value = end_sep.join([", ".join(values[:-1]), values[-1]])

        return flatten_value


class StructureField(BaseTemplate):
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

    _default_tag_positions: ClassVar[dict[str, int]] = {
        "title": 0,
        "description": 0,
        "document": 0,
        "main_points": 0,
        "categories": 0,
        "tags": 0,
    }
    _default_tag: ClassVar[dict] = {
        "title": ["Title"],
        "description": [
            "Description",
            "Introduction",
            "Summary",
            "Intro",
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

    def items(self) -> list[tuple[str, str, str]]:
        items = []
        for field, values in self._default_tag.items():
            default_value = self._get_value_by_position(
                field, positions=self._default_tag_positions, values=values
            ).title()
            custom_value = self._get_value_by_position(field).title()
            items.append((field, custom_value, default_value))
        return items


class Template(BaseTemplate):
    """
    Extends the BaseTemplate class to provide specialized functionality for generating structured prompts.
    This class combines system, user, and structural prompt templates to create flexible, multi-purpose
    content generation workflows.

    Attributes:
        system_prompts (list[str]): A list of templates for system-level prompts, defining the role or behavior of the model.
        user_prompts (list[str]): A list of templates for user-level prompts, specifying user queries or requests.
        title (list[str]): A collection of title templates designed for SEO optimization and clear messaging.
        description (list[str]): Templates for crafting compelling introductions or meta descriptions.
        document (list[str]): Prompts aimed at refining and enhancing the language of main content.
        structure_field (Optional[StructureField]): An instance of `StructureField` to manage structured fields in the prompt.
        main_points (list[str]): Templates for summarizing or emphasizing main points.
        categories (list[str]): Prompts for identifying or refining article categories or themes.
        tags (list[str]): Templates for selecting or enhancing tags and tags for SEO.
        _structure_items (dict[str, tuple[str, str, str]]): An internal dictionary mapping structured fields to their definitions
            (name, custom label, default label).

    Example Usage:
        >>> prompt_instance = Template(
        ...         structure_field=StructureField(
        ...         title=["Custom Title"],
        ...         description=["Custom Description"],
        ...         document=["Custom Article"],
        ...         main_points=["Custom Main Points"],
        ...         categories=["Custom Categories"],
        ...         tags=["Custom Tags"],
        ...    ),
        ... )   # Create fully customized structured reminders.
        >>> response = prompt_instance.template(
        ...    template=GEMMA_TEMPLATE,
        ...    user_template=USER_TEMPLATE,
        ...    instruction_template=INSTRUCTION_TEMPLATE,
        ...    structure_template=STRUCTURE_TEMPLATE,
        ...    title="Gemma open models",
        ...    description="Gemma: Introducing new state-of-the-art open models.",
        ...    main_points=["Main point 1", "Main point 2"],
        ...    categories=["Artificial Intelligence", "Gemma"],
        ...    tags=["AI", "LLM", "Google"],
        ...    input="Gemma open models are built from the same research and technology as Gemini models. Gemma 2 comes in 2B, 9B and 27B and Gemma 1 comes in 2B and 7B sizes.",
        ...    output="A new family of open language models demonstrating strong performance across academic benchmarks for language understanding, reasoning, and safety.",
        ... )  # remove kwargs if not used.
        >>> print(response)
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
        comes in => English
        technology as => English
        Text Analysis 2: Frequent bigrams in Vietnamese confirm the language context.

        Example 3: Trigrams (three words)
        technology as Gemini => English
        Text Analysis 3: Trigrams further validate the linguistic analysis and the necessity to respond in English.

        # Conclusion of Text Analysis:
        The linguistic analysis confirms the text is predominantly in English. Consequently, the response should be structured and written in English to align with the original text and context.

        # Response Structure Format:
        You must follow the response structure outlined below to ensure clarity and alignment with user expectations:
        **Custom Title (Title):**
        Rewrite the title to reflect the main keyword and topic.
        **Custom Description (Description):**
        Write description of the article in one or two sentences while focusing on reader benefits and engage curiosity.
        **Custom Article (Article):**
        Reimagine this article with a more engaging and creative tone. Add metaphors, analogies, or storytelling elements to make it more captivating for readers.
        **Custom Main Points (Main Points):**
        Ensure all key points flow logically from one to the next.
        **Custom Categories (Categories):**
        Assign appropriate categories to the article based text or target audience.
        **Custom Tags (Tags):**
        Create tags to include relevant keywords. Ensure the tags align with popular search queries.

        By adhering to this format, the response will maintain linguistic integrity while enhancing professionalism and structure.

        # Text:
        Gemma open models are built from the same research and technology as Gemini models. Gemma 2 comes in 2B, 9B and 27B and Gemma 1 comes in 2B and 7B sizes.<end_of_turn>
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

    """  # noqa: E501

    _structure_items: dict[str, tuple[str, str, str]] = {}
    system_prompts: list[str] = ["You are a multilingual professional writer."]
    user_prompts: list[str] = [
        (
            "Rewrite the text to be more search engine friendly. Incorporate relevant"
            " keywords naturally, improve readability, and ensure it aligns with SEO best"
            " practices."
        ),
        (
            "Rewrite the text with a more engaging and creative tone. Use vivid imagery,"
            " descriptive language, and a conversational style to captivate the reader."
        ),
        (
            "Rewrite the text to make it more concise without losing its meaning or"
            " impact. Remove unnecessary words and phrases while preserving the core"
            " message."
        ),
    ]
    title: list[str] = [
        "Rewrite the title to reflect the main keyword and topic.",
        "Rewrite the title to make it concise, memorable, and optimized for SEO.",
        "Create a title that is concise, clear, attention-grabbing, and SEO-optimized.",
    ]
    description: list[str] = [
        "Rewrite the description with a bold claim or statistic to grab attention.",
        (
            "Write description of the article in one or two sentences while focusing on"
            " reader benefits and engage curiosity."
        ),
        "Begin the description with an engaging anecdote or story for SEO optimization.",
    ]
    document: list[str] = [
        (
            "Transform this text into a formal, professional tone suitable for business"
            " communication or an academic audience. Focus on improving vocabulary,"
            " grammar, and structure."
        ),
        (
            "Rewrite this content to be SEO-friendly. Include relevant tags, optimize"
            " the title and subheadings, and ensure the text flows naturally for search"
            " engines and readers."
        ),
        (
            "Reimagine this article with a more engaging and creative tone. Add"
            " metaphors, analogies, or storytelling elements to make it more captivating"
            " for readers."
        ),
    ]
    structure_field: Optional[StructureField] = Field(default_factory=StructureField)
    main_points: list[str] = [
        (
            "Summarize the main ideas into concise, actionable key points for added"
            " context to make them more engaging."
        ),
        "Simplify the original key points to make them clearer and more reader-friendly.",
        "Ensure all key points flow logically from one to the next.",
    ]
    categories: list[str] = [
        "Assign appropriate categories to the article based text or target audience.",
        "Rewrite categories to align with industry standards or popular topics.",
        (
            "Use categories that align with similar articles on the topic and improve SEO"
            " and discoverability."
        ),
    ]
    tags: list[str] = [
        "Add trending keyword terms or phrases to the tags for increased visibility.",
        (
            "Create tags to include relevant keywords. Ensure the tags align with"
            " popular search queries."
        ),
        "Focus use tags that reflect the article’s subtopics or themes for better SEO.",
    ]

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
        self.user_prompts = _normalize(self.user_prompts, ".")
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
        template: Optional[TemplateTypes] = GEMMA_TEMPLATE,
        user_template: Optional[TemplateTypes] = USER_TEMPLATE,
        instruction_template: Optional[TemplateTypes] = None,
        structure_template: Optional[TemplateTypes] = None,
        output_format: Union[str, Literal["text", "alpaca", "gpt"]] = "text",
        eos_token_str: Optional[str] = "",
        max_concurrency: int = 4,
        **kwargs,
    ) -> Union[Dataset, DatasetDict]:
        """
        Processes and loads a dataset, generating prompts based on the provided templates.

        This function supports various input formats such as file paths, dictionaries, or Hugging Face Dataset objects.
        It uses templates to create structured prompts and supports concurrent processing for efficiency.

        Args:
            fp (Union[str, list[dict], Dataset, DatasetDict]):
                Input data as a file path, a list of dictionaries, or a Hugging Face Dataset/DatasetDict object.
            template (Optional[TemplateTypes]):
                The base template for constructing prompts. Defaults to `GEMMA_TEMPLATE`.
            user_template (Optional[TemplateTypes]):
                The base user_template for constructing prompts. Defaults to `USER_TEMPLATE`.
            instruction_template (Optional[TemplateTypes]):
                Template for including specific instructions in the prompts.
            structure_template (Optional[TemplateTypes]):
                Template for structuring the response content.
            output_format (Union[str, Literal["text", "alpaca", "gpt"]]):
                Specifies the format for the generated prompts. Default is "text".
            eos_token_str (Optional[str]):
                Append eos token to the end of the model output.
            max_concurrency (int):
                Maximum number of concurrent threads for processing data. Default is 4.
            **kwargs: Additional parameters, including:
                - `token` (Optional[str]): Hugging Face authentication token.
                - `split` (Optional[list[str]]): Dataset split for Hugging Face Dataset loading.
                - `additional parameters` see also: `Template.template`.

        Returns:
            Dataset: A Hugging Face Dataset or DatasetDict object containing the processed prompts.

        Raises:
            TypeError: If the input data type is not supported.

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
            dataset = prompt_instance.load_dataset(data_dict, output_format='text')   # enum: text, gpt, alpaca
            print(dataset['text'][0])
            ```
        """  # noqa: E501

        async def create_task(config):
            async with semaphore:
                config.update(kwargs)
                if output_format == "alpaca":
                    items.append(
                        self.to_alpaca(
                            user_template,
                            instruction_template,
                            structure_template,
                            eos_token_str,
                            **config,
                        )
                    )
                elif output_format == "gpt":
                    items.append(
                        self.to_openai(
                            user_template,
                            instruction_template,
                            structure_template,
                            eos_token_str,
                            **config,
                        )
                    )
                else:
                    items.append(
                        self.to_text(
                            template,
                            user_template,
                            instruction_template,
                            structure_template,
                            eos_token_str,
                            **config,
                        )
                    )

                pbar.update(1)

        async def run_task(ds):
            await asyncio.wait([loop.create_task(create_task(config)) for config in ds])

        def _close():
            """Notebook Error"""
            try:
                loop.close()
            except RuntimeError:
                pass

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

        items = []
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

        raise TypeError("Invalid dataset type.")

    def get_system_prompt(self) -> str:
        """
        Retrieves a Round-Robin system-level prompt from the predefined list.

        Returns:
            str: A Round-Robin selected system-level prompt.
        """
        return self._get_value_by_position("system_prompts")

    def get_user_prompt(self) -> str:
        """
        Retrieves a Round-Robin user-level prompt from the predefined list.

        Returns:
            str: A Round-Robin selected user-level prompt.
        """
        return self._get_value_by_position("user_prompts")

    def get_user_kwargs(
        self,
        instruction_template: Optional[TemplateTypes] = INSTRUCTION_TEMPLATE,
        structure_template: Optional[TemplateTypes] = STRUCTURE_TEMPLATE,
        *,
        n_words: int = 5,
        bullet_style: Optional[Union[str, Literal["dash", "number"]]] = "dash",
        **kwargs,
    ):
        """
        Generates an kwargs for the user template

        This method customizes the instruction prompt by combining system prompts, user prompts, and optional structural prompts.
        It allows for dynamic language customization and response structure formatting.

        Args:
            instruction_template (Optional[Union[str, Callable]]): Instruction template for generating instruction prompt.
            structure_template (Optional[Union[str, Callable]]): Structure template for generating structure prompt.
            n_words (int): Number of words frequently used to create unigrams, bigrams and trigrams.
            bullet_style (Optional[str]): Bullet list style start dash or number. Default is dash.
            **kwargs: see also `Template.template`.

        Returns:
            dict: instruction data dict.
        """  # noqa: E501

        system_template_str, prompt_template_str, structure_template_str, document = (
            self._get_prompts(structure_template, **kwargs)
        )
        language_code, language = get_language(document)
        unigrams = self._get_frequently_words(
            document, n=1, response_n=n_words, language_code=language_code
        )
        bigrams = self._get_frequently_words(
            document,
            n=2,
            response_n=n_words,
            language_code=language_code,
            excluded_words=unigrams,
        )
        trigrams = self._get_frequently_words(
            document,
            n=3,
            response_n=n_words,
            language_code=language_code,
            excluded_words=unigrams,
        )
        instruction_kwargs = dict(
            document=document,
            topic_values=", ".join(kwargs.get("categories", []) or []),
            keyword_values=", ".join(kwargs.get("tags", []) or []),
            unigrams=unigrams,
            bigrams=bigrams,
            trigrams=trigrams,
            n_words=n_words,
            language=language,
            bullet_style=bullet_style,
        )
        if isinstance(instruction_template, Callable):
            instruction_template_str = instruction_template(
                self, **instruction_kwargs, **kwargs
            )
        else:
            instruction_template_str = self._formatting_instruction_fn(
                instruction_template, **instruction_kwargs
            )

        if structure_template_str:
            if isinstance(structure_template, Callable):
                structure_template_str = structure_template(
                    self, self._get_structure_attrs(**kwargs), **kwargs
                )
            else:
                structure_template_str = self._formatting_structure_user_fn(
                    structure_template,
                    **kwargs,
                )

        return dict(
            system_template=system_template_str,
            prompt_template=prompt_template_str,
            instruction_template=instruction_template_str,
            structure_template=structure_template_str,
            **instruction_kwargs,
        )

    def get_structure_prompt(self, **kwargs) -> str:
        """
        Constructs a structured response user_template based on provided keyword arguments.

        Args:
            **kwargs: Key-value pairs for customizing the structured response user_template.

        Returns:
            str: A formatted string representing the structured response user_template.
        """

        prompts = []
        structure_fields = {}
        for field, custom_label, default_label in self.structure_field.items():
            if field in kwargs:
                if custom_label:
                    bold_label = "**%s (%s):**" % (custom_label, default_label)
                else:
                    bold_label = "**%s:**" % default_label

                structure_fields[field] = (bold_label, custom_label, default_label)
                prompts.append(
                    " ".join([bold_label, self._get_value_by_position(field)])
                )

        self._structure_items = structure_fields
        return "\n\n".join(prompts)

    def template(
        self,
        template: Optional[TemplateTypes] = GEMMA_TEMPLATE,
        user_template: Optional[TemplateTypes] = USER_TEMPLATE,
        instruction_template: Optional[TemplateTypes] = None,
        structure_template: Optional[TemplateTypes] = None,
        eos_token_str: Optional[str] = "",
        **kwargs,
    ):
        """
        Generates a complete prompt by integrating system, user, and structural elements.

        Args:
            template (Optional[Union[str, Callable]]): Base template for constructing the final prompt.
            user_template (Optional[Union[str, Callable]]): User Template for user prompt.
            instruction_template (Optional[Union[str, Callable]]): Instruction template for instruction prompt, if applicable.
            structure_template (Optional[Union[str, Callable]]): Structuring template for structuring prompt, if applicable.
            eos_token_str (Optional[str]): Append eos token to the end of the model output.
            **kwargs: Additional parameters including:
                - output: Optional[str] = Model response output.
                - title: Optional[list[str]] = List of title to include in the prompt.
                - description: Optional[list[str]] = List of description to include in the prompt.
                - document: Optional[list[str]] = The main text content or article to be processed.
                - main_points: Optional[list[str]] = List of main points to include in the prompt.
                - categories: Optional[list[str]] = List of categories/categories to include in the prompt.
                - tags: Optional[list[str]] = List of tags/tags to include in the prompt.
                - bullet_style: (Optional[Literal['dash', 'number']]): Bullet list style start dash or number. Default is dash.
                - additional parameters: see also `Template.template`.

        Returns:
            str: A formatted prompt string combining multiple components.

        Example:
            >>> prompt_instance = Template(...)
            >>> response = prompt_instance.template(
            ...     document="Sample document",
            ...     output="Generated response",
            ... )
            >>> print(response)
        """  # noqa: E501

        user_template, model_template = self._get_templates(
            user_template, instruction_template, structure_template, **kwargs
        )
        if isinstance(template, Callable):
            return template(user_template=user_template, model_template=model_template)

        return template.format(
            user_template=user_template,
            model_template=model_template,
        )

    def generate_user_prompt(
        self,
        user_template: Optional[TemplateTypes] = USER_TEMPLATE,
        instruction_template: Optional[TemplateTypes] = None,
        structure_template: Optional[TemplateTypes] = None,
        **kwargs,
    ) -> str:
        """
        Generates a user-specific prompt by combining multiple user-defined inputs.

        This method collects and formats user prompts, optionally including structural elements, to create
        a coherent and complete prompt.

        Args:
            user_template (Optional[Union[str, Callable]]): User template for generating final prompt. Default is USER_TEMPLATE.
            instruction_template (Optional[Union[str, Callable]]): Instruction template for generating instruction prompt.
            structure_template (Optional[Union[str, Callable]]): Structure template for generating structure prompt.
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

        if instruction_template is not None:
            user_kwargs = self.get_user_kwargs(
                instruction_template, structure_template, **kwargs
            )
            return user_template.format(**user_kwargs)

        return "\n\n".join(
            [
                p.strip()
                for p in self._get_prompts(structure_template, **kwargs)
                if p.strip()
            ]
        )

    def generate_model_prompt(
        self,
        structure_template: Optional[TemplateTypes] = None,
        eos_token_str: Optional[str] = "",
        bullet_style: Optional[Union[str, Literal["dash", "number"]]] = "dash",
        **kwargs,
    ) -> str:
        """
        Generates a model-specific prompt by formatting output and document content according to structural rules.

        This method organizes output content and any additional fields into a structured format
        for model processing.

        Args:
            structure_template (Optional[Union[str, Callable]]): A structure template defining the generating structure prompt.
            eos_token_str (Optional[str]): Append eos token to the end of the model output.
            bullet_style (Optional[str]): Bullet list style start dash or number. Default is dash.
            **kwargs: See also `Template.template`.

        Returns:
            str: A structured prompt string tailored for model input.

        Example:
            >>> prompt_instance = Template()
            >>> response = prompt_instance.generate_model_prompt(
            ...     document="Sample document",
            ...     output="Generated response",
            ... )
            >>> print(response)
        """  # noqa: E501

        output_document = kwargs.get("output", "")
        kwargs["document"] = output_document
        if isinstance(structure_template, Callable):
            if isinstance(structure_template, Callable):
                output_document = structure_template(self._structure_items, **kwargs)

        elif isinstance(structure_template, str):
            output_document = self._formatting_structure_model_fn(
                self._structure_items, bullet_style, **kwargs
            )

        return output_document.strip() + eos_token_str

    def to_text(
        self,
        template: Optional[TemplateTypes] = GEMMA_TEMPLATE,
        user_template: Optional[TemplateTypes] = USER_TEMPLATE,
        instruction_template: Optional[TemplateTypes] = INSTRUCTION_TEMPLATE,
        structure_template: Optional[TemplateTypes] = STRUCTURE_TEMPLATE,
        eos_token_str: Optional[str] = "",
        **kwargs,
    ) -> dict:
        """Generate SFT Text Template format"""

        return {
            "text": self.template(
                template,
                user_template,
                instruction_template,
                structure_template,
                eos_token_str,
                **kwargs,
            )
        }

    def to_alpaca(
        self,
        user_template: Optional[TemplateTypes] = USER_TEMPLATE,
        instruction_template: Optional[TemplateTypes] = INSTRUCTION_TEMPLATE,
        structure_template: Optional[TemplateTypes] = STRUCTURE_TEMPLATE,
        eos_token_str: Optional[str] = "",
        **kwargs,
    ) -> dict:
        """Generate Alpaca Template format"""
        instruction_kwargs = self.get_user_kwargs(
            instruction_template, structure_template, **kwargs
        )
        instruction_template = instruction_kwargs["instruction_template"]
        model_template = self.generate_model_prompt(
            structure_template, eos_token_str, **kwargs
        )
        return dict(
            instruction=instruction_template,
            input=kwargs.get("document", ""),
            output=model_template,
        )

    def to_openai(
        self,
        user_template: Optional[TemplateTypes] = USER_TEMPLATE,
        instruction_template: Optional[TemplateTypes] = INSTRUCTION_TEMPLATE,
        structure_template: Optional[TemplateTypes] = STRUCTURE_TEMPLATE,
        eos_token_str: Optional[str] = "",
        **kwargs,
    ) -> dict:
        """Generate Open AI Template format"""

        user_template, model_template = self._get_templates(
            user_template,
            instruction_template,
            structure_template,
            eos_token_str,
            **kwargs,
        )
        return dict(
            human=user_template,
            gpt=model_template,
        )

    def _get_templates(
        self,
        user_template: Optional[TemplateTypes] = USER_TEMPLATE,
        instruction_template: Optional[TemplateTypes] = None,
        structure_template: Optional[TemplateTypes] = None,
        eos_token_str: Optional[str] = "",
        **kwargs,
    ) -> tuple[str, str]:
        user_template = self.generate_user_prompt(
            user_template, instruction_template, structure_template, **kwargs
        )
        model_template = self.generate_model_prompt(
            structure_template, eos_token_str, **kwargs
        )
        return user_template.strip(), model_template.strip()

    def _get_prompts(
        self,
        structure_template: TemplateTypes = None,
        *args,
        **kwargs,
    ):
        prompts = [self.get_system_prompt(), self.get_user_prompt()]
        structure_template_str = ""
        if isinstance(structure_template, (str, Callable)):
            structure_template_str = self.get_structure_prompt(**kwargs)

        prompts.append(structure_template_str)
        prompts.append(kwargs.get("document", ""))
        return [p.strip() for p in prompts]

    def _gen_bullet_list_style(self, words: list[str], bullet_style: str = None) -> str:
        if words:
            if bullet_style is None:
                return "\n".join(words)

            return "\n".join(
                [
                    "{} {}".format(
                        f"{idx+1}." if bool(bullet_style == "number") else "-",
                        word.strip(),
                    )
                    for idx, word in enumerate(words)
                ]
            ).strip()
        return ""

    def _get_frequently_words(
        self,
        document: str,
        *,
        n: int = 1,
        response_n: int = 10,
        language_code: str = None,
        min_chars_length: int = 2,
        max_chars_length: int = 0,
        excluded_words: list[str] = (),
    ) -> list[str]:
        if not language_code:
            language_code, _ = get_language(document)

        return [
            word
            for word in get_frequently_words(
                document,
                n=n,
                response_n=response_n,
                language_code=language_code,
                excluded_words=excluded_words,
            )
        ]

    def _formatting_instruction_fn(
        self,
        instruction_template: str,
        document: str,
        topic_values: str,
        keyword_values: str,
        unigrams: list[str],
        bigrams: list[str],
        trigrams: list[str],
        language: str,
        *args,
        **kwargs,
    ) -> str:
        def _ftm_template(word):
            return f"{word} => {language}"

        return instruction_template.format(
            document=document,
            topic_values=topic_values,
            keyword_values=keyword_values,
            unigrams="\n".join([_ftm_template(word) for word in unigrams]),
            bigrams="\n".join([_ftm_template(word) for word in bigrams]),
            trigrams="\n".join([_ftm_template(word) for word in trigrams]),
            language=language,
            **kwargs,
        )

    def _formatting_structure_user_fn(
        self,
        structure_template: str = "",
        **kwargs,
    ) -> str:
        prompts = []
        for _, data in self._get_structure_attrs(**kwargs).items():
            prompts.append(
                "{field}\n{prompt}".format(
                    field=data["bold_value"], prompt=data["prompt"]
                )
            )

        return structure_template.format(structure_template="\n".join(prompts))

    def _formatting_structure_model_fn(
        self,
        structure_data: dict,
        bullet_style: str = None,
        *args,
        **kwargs,
    ) -> str:
        prompts = []
        for field, (
            bold_value,
            custom_label,
            default_label,
        ) in structure_data.items():
            if field not in kwargs:
                continue

            value = kwargs[field]
            if not value:
                continue

            if isinstance(value, list):
                value = self._gen_bullet_list_style(value, bullet_style)

            if field == "title":
                if not value.strip().startswith("#"):
                    value = "### " + value.strip()

            label = custom_label or default_label
            template = "## **{}**:\n{}".format(label, value)
            prompts.append(template)
        return "\n\n".join(prompts).strip()

    def _get_structure_attrs(self, **kwargs):
        mapping = {}
        for field, (
            bold_value,
            custom_value,
            default_value,
        ) in self._structure_items.items():
            if field in kwargs:
                mapping[field] = {
                    "prompt": self._get_value_by_position(field),
                    "bold_value": bold_value,
                    "custom_value": custom_value,
                    "default_value": default_value,
                }
        return mapping


gemma_template = Template()
vietnamese_template = Template(
    end_sep="và",
    system_prompts=[
        (
            "Bạn là một nhà sáng tạo nội dung, viết nội dung chuyên nghiệp biết nhiều"
            " ngôn ngữ."
        ),
    ],
    user_prompts=[
        (
            "Viết lại văn bản để thân thiện hơn với công cụ tìm kiếm. Kết hợp các từ khóa"
            " có liên quan một cách tự nhiên, cải thiện khả năng đọc và đảm bảo phù hợp"
            " với các phương pháp hay nhất của SEO."
        ),
        (
            "Viết lại văn bản với giọng văn hấp dẫn và sáng tạo hơn. Sử dụng hình ảnh"
            " sống động, ngôn ngữ mô tả và phong cách đàm thoại để thu hút người đọc."
        ),
        (
            "Viết lại văn bản để làm cho nó súc tích hơn mà không làm mất đi ý nghĩa hoặc"
            " tác động của nó. Loại bỏ các từ và cụm từ không cần thiết trong khi vẫn giữ"
            " nguyên thông điệp cốt lõi."
        ),
    ],
    structure_field=StructureField(
        title=["Tiêu đề"],
        description=["Mô tả"],
        document=["Bài viết chỉnh sửa"],
        main_points=["Điểm nổi bật", "Điểm chính"],
        categories=["Danh mục", "Chủ đề"],
        tags=["Từ khoá"],
    ),
    title=[
        (
            "Viết lại tiêu đề để ngắn gọn, hấp dẫn và được tối ưu hóa cho SEO bằng các từ"
            " khóa có liên quan."
        ),
        "Tạo tiêu đề ngắn gọn, thu hút sự chú ý và được tối ưu hóa cho SEO.",
        "Viết lại tiêu đề, kết hợp các từ khóa hoặc cụm từ thịnh hành vào tiêu đề.",
    ],
    description=[
        "Tóm tắt bài viết trong một câu, làm nổi bật so và thu hút sự tò mò.",
        (
            "Tạo mô tả bằng một sự thật hoặc số liệu thống kê đáng ngạc nhiên để thu hút"
            " sự chú ý để thu hút người đọc."
        ),
        (
            "Tóm tắt bài viết trong một hoặc hai câu tập trung vào ý chính, kết hợp các"
            " từ khóa chính một cách tự nhiên."
        ),
    ],
    document=[
        (
            "Viết lại bài viết với giọng điệu chuyên nghiệp hơn và cấu trúc hợp lý, dễ"
            " đọc hơn."
        ),
        "Viết lại bài viết để làm cho chúng hấp dẫn hơn và có phong cách chuyên nghiệp.",
        "Đơn giản hóa thuật ngữ kỹ thuật để làm cho bài viết dễ hiểu với tất cả độc giả.",
    ],
    main_points=[
        (
            "Tạo điểm chính dưới dạng danh sách, thêm ví dụ hoặc giải thích ngắn gọn cho"
            " từng điểm chính."
        ),
        "Tóm tắt các ý chính thành các điểm chính ngắn gọn, thu hút người đọc.",
        (
            "Đảm bảo tất cả các điểm chính đều có sự liên kết hợp lý từ điểm này sang"
            " điểm khác."
        ),
    ],
    categories=[
        "Viết lại các danh mục để phù hợp với chủ đề phổ biến theo bài viết.",
        "Tạo danh sách danh mục để phù hợp với các từ khóa được sử dụng trong bài viết.",
        "Chọn các danh mục cải thiện SEO và khả năng khám phá theo nội dung bài viết.",
    ],
    tags=[
        "Tạo danh sách từ khóa thịnh hành giúp SEO tốt hơn.",
        "Tạo danh sách từ khóa có liên quan phù hợp với truy vấn tìm kiếm phổ biến.",
        "Tập trung vào các từ khóa phổ biến trong bài viết để SEO tốt hơn.",
    ],
)
