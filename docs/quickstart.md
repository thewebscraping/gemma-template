# Quickstart Guide for Gemma Template

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

### Output

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

## Local Language Configuration
Here is how to customize Vietnamese language configuration. Template formatting using Jinja2.

### Input Template
```python
VIETNAMESE_INPUT_TEMPLATE = """{{ system_prompt }}
{% if instruction %}\n{{ instruction }}\n{% endif %}
{% if prompt_structure %}{{ prompt_structure }}\n{% else %}{{ prompt }}\n{% endif %}
# Văn Bản:
{{ input }}
{% if topic_value %}\nDanh Mục: {{ topic_value }}\n{% endif %}{% if keyword_value %}Từ Khoá: {{ keyword_value }}\n{% endif %}
"""
```

### Output Template
```python
VIETNAMESE_OUTPUT_TEMPLATE = """{% if structure_fields %}{% for field in structure_fields %}## **{{ field.label.custom or field.label.default }}:**\n{% if field.key == 'title' %}### {% endif%}{{ field.value }}\n\n{% endfor %}{% else %}{{ output }}{% endif %}"""
```

### Instruction Template
```python
VIETNAMESE_INSTRUCTION_TEMPLATE = """# Vai trò:
Bạn là một biên tập viên nội dung chuyên nghiệp, nhà phân tích ngôn ngữ và chuyên gia đa ngôn ngữ, chuyên về viết có cấu trúc và xử lý văn bản nâng cao.

# Nhiệm Vụ:
Mục tiêu chính của bạn là:
1. Nhiệm vụ chính của bạn là viết lại nội dung được cung cấp theo định dạng có cấu trúc, chuyên nghiệp hơn, đồng thời vẫn giữ nguyên ý định và ý nghĩa ban đầu.
2. Nâng cao khả năng hiểu từ vựng bằng cách phân tích văn bản với unigrams (từ đơn), bigrams (hai từ) và trigrams (ba từ).
3. Đảm bảo phản hồi của bạn tuân thủ nghiêm ngặt định dạng cấu trúc được quy định.
4. Phản hồi bằng ngôn ngữ chính của văn bản đầu vào trừ khi có hướng dẫn thay thế rõ ràng.

# Kỳ Vọng Bổ Sung:
1. Cung cấp phiên bản văn bản đầu vào được viết lại, nâng cao, đảm bảo tính chuyên nghiệp, rõ ràng và cấu trúc được cải thiện.
2. Tập trung vào khả năng đa ngôn ngữ, sử dụng vốn từ vựng phức tạp, ngữ pháp để cải thiện phản hồi của bạn.
3. Giữ nguyên ngữ cảnh và sắc thái văn hóa của văn bản gốc khi viết lại.
{% if topic_value %}\nTopics: {{ topic_value }}\n{% endif %}{% if keyword_value %}Keywords: {{ keyword_value }}\n{% endif %}

# Phân Tích Văn Bản:
Ví Dụ 1: Unigrams (nhóm 1 chữ cái){% for word in unigrams %}\n{{ word }} => Tiếng Việt ({{ language }}){% endfor %}

Phân Tích Văn Bản 1: đây là những từ thông dụng trong tiếng Việt ({{ language }}), cho biết văn bản được viết bằng tiếng Việt ({{ language }}).

Ví Dụ 2: Bigrams (nhóm 2 chữ cái){% for word in bigrams %}\n{{ word }} => Tiếng Việt ({{ language }}){% endfor %}
Phân Tích Văn Bản 2: các từ ghép thường gặp trong Tiếng Việt ({{ language }}) xác nhận bối cảnh ngôn ngữ.

Ví Dụ 3: Trigrams (nhóm 3 chữ cái)\n{% for word in trigrams %}{{ word }} => Tiếng Việt ({{ language }}){% endfor %}
Phân Tích Văn Bản 3: các từ ghép 3 chữ liên tiếp là những từ tiếng Việt sử dụng thường xuyên, xác nhận sự cần thiết phải phản hồi bằng Tiếng Việt ({{ language }}).

# Kết Luận Phân Tích Văn Bản:
Phân tích ngôn ngữ xác nhận văn bản chủ yếu bằng Tiếng Việt ({{ language }}). Do đó, phản hồi phải được cấu trúc và viết bằng Tiếng Việt ({{ language }}). để phù hợp với văn bản và ngữ cảnh gốc.
"""
```

### Prompt Template
```python
VIETNAMESE_PROMPT_TEMPLATE = """{% if prompt %}\n\n# Đầu Vào Văn Bản:\n{{ prompt }}\n\n{% endif %}{% if structure_fields %}# Định Dạng Cấu Trúc Phản Hồi:
Bạn phải tuân theo cấu trúc phản hồi:
{% for field in structure_fields %}{{ field.label }}\n{% endfor %}

Bằng cách tuân thủ định dạng này, phản hồi sẽ duy trì tính toàn vẹn về mặt ngôn ngữ đồng thời tăng cường tính chuyên nghiệp, cấu trúc và sự phù hợp với mong đợi của người dùng.
{% endif %}"""
```

### New Template Instance

```python
from gemma_template import Template, FieldPosition

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

response = vietnamese_gemma_template.apply_template(
    title="Gemma open models",
    description="Gemma: Introducing new state-of-the-art open models.",
    main_points=["Main point 1", "Main point 2"],
    categories=["Artificial Intelligence", "Gemma"],
    tags=["AI", "LLM", "Google"],
    document="Các mẫu mở Gemma được xây dựng từ cùng một nghiên cứu và công nghệ như các mẫu Gemini. Gemma 2 có các kích cỡ 2B, 9B và 27B và Gemma 1 có các kích cỡ 2B và 7B.",
    output="A new family of open language models demonstrating strong performance across academic benchmarks for language understanding, reasoning, and safety.",
    max_hidden_words=.05,  # Hide 5% words of per documents.
    min_chars_length=2,  # Minimum character of a word, used to create unigrams, bigrams, and trigrams. Default is 2.
    max_chars_length=0,  # Maximum character of a word, used to create unigrams, bigrams and trigrams. Default is 0.
)  # remove kwargs if not used.

print(response)
```

### Output
```shell
<start_of_turn>user
Bạn là một nhà sáng tạo nội dung, viết nội dung chuyên nghiệp biết nhiều ngôn ngữ.

# Vai trò:
Bạn là một biên tập viên nội dung chuyên nghiệp, nhà phân tích ngôn ngữ và chuyên gia đa ngôn ngữ, chuyên về viết có cấu trúc và xử lý văn bản nâng cao.

# Nhiệm Vụ:
Mục tiêu chính của bạn là:
1. Nhiệm vụ chính của bạn là viết lại nội dung được cung cấp theo định dạng có cấu trúc, chuyên nghiệp hơn, đồng thời vẫn giữ nguyên ý định và ý nghĩa ban đầu.
2. Nâng cao khả năng hiểu từ vựng bằng cách phân tích văn bản với unigrams (từ đơn), bigrams (hai từ) và trigrams (ba từ).
3. Đảm bảo phản hồi của bạn tuân thủ nghiêm ngặt định dạng cấu trúc được quy định.
4. Phản hồi bằng ngôn ngữ chính của văn bản đầu vào trừ khi có hướng dẫn thay thế rõ ràng.

# Kỳ Vọng Bổ Sung:
1. Cung cấp phiên bản văn bản đầu vào được viết lại, nâng cao, đảm bảo tính chuyên nghiệp, rõ ràng và cấu trúc được cải thiện.
2. Tập trung vào khả năng đa ngôn ngữ, sử dụng vốn từ vựng phức tạp, ngữ pháp để cải thiện phản hồi của bạn.
3. Giữ nguyên ngữ cảnh và sắc thái văn hóa của văn bản gốc khi viết lại.

Topics: Artificial Intelligence, Gemma
Keywords: AI, LLM, Google


# Phân Tích Văn Bản:
Ví Dụ 1: Unigrams (nhóm 1 chữ cái)
và => Tiếng Việt (Vietnamese)
các => Tiếng Việt (Vietnamese)
mẫu => Tiếng Việt (Vietnamese)
có => Tiếng Việt (Vietnamese)
cỡ => Tiếng Việt (Vietnamese)

Phân Tích Văn Bản 1: đây là những từ thông dụng trong tiếng Việt (Vietnamese), cho biết văn bản được viết bằng tiếng Việt (Vietnamese).

Ví Dụ 2: Bigrams (nhóm 2 chữ cái)
mở Gemma => Tiếng Việt (Vietnamese)
Gemma được => Tiếng Việt (Vietnamese)
được xây => Tiếng Việt (Vietnamese)
xây dựng => Tiếng Việt (Vietnamese)
dựng từ => Tiếng Việt (Vietnamese)
Phân Tích Văn Bản 2: các từ ghép thường gặp trong Tiếng Việt (Vietnamese) xác nhận bối cảnh ngôn ngữ.

Ví Dụ 3: Trigrams (nhóm 3 chữ cái)
mở Gemma được => Tiếng Việt (Vietnamese)Gemma được xây => Tiếng Việt (Vietnamese)được xây dựng => Tiếng Việt (Vietnamese)xây dựng từ => Tiếng Việt (Vietnamese)dựng từ cùng => Tiếng Việt (Vietnamese)
Phân Tích Văn Bản 3: các từ ghép 3 chữ liên tiếp là những từ tiếng Việt sử dụng thường xuyên, xác nhận sự cần thiết phải phản hồi bằng Tiếng Việt (Vietnamese).

# Kết Luận Phân Tích Văn Bản:
Phân tích ngôn ngữ xác nhận văn bản chủ yếu bằng Tiếng Việt (Vietnamese). Do đó, phản hồi phải được cấu trúc và viết bằng Tiếng Việt (Vietnamese). để phù hợp với văn bản và ngữ cảnh gốc.

# Đầu Vào Văn Bản:
Viết lại bài viết này để làm nổi bật đề xuất giá trị độc đáo của nó trong khi đảm bảo nó được xếp hạng tốt cho các từ khóa mục tiêu.

# Định Dạng Cấu Trúc Phản Hồi:
Bạn phải tuân theo cấu trúc phản hồi:
**Tiêu Đề (Title):** Tập trung vào một góc độ gây ngạc nhiên hoặc độc đáo trong tiêu đề. Bao gồm các con số hoặc số liệu thống kê trong tiêu đề để tạo sự cụ thể.
**Mô Tả (Description):** Viết lại phần mô tả bằng một tuyên bố hoặc số liệu thống kê táo bạo để thu hút sự chú ý.
**Bài Viết Chỉnh Sửa (Article):** Viết lại bài viết này để đối tượng chuyên nghiệp, tập trung vào các chi tiết kỹ thuật, thuật ngữ chuyên ngành và thông tin chi tiết có thể thực hiện được.
**Điểm Nổi Bật (Main Points):** Tóm tắt các ý chính thành các điểm chính ngắn gọn, có thể hành động để thêm ngữ cảnh nhằm khiến chúng hấp dẫn hơn.
**Danh Mục (Categories):** Đảm bảo các danh mục phản ánh sở thích của đối tượng mục tiêu.
**Từ Khoá (Tags):** Tạo 5 từ khóa phù hợp với nội dung tương tự để có cơ hội quảng cáo chéo.


Bằng cách tuân thủ định dạng này, phản hồi sẽ duy trì tính toàn vẹn về mặt ngôn ngữ đồng thời tăng cường tính chuyên nghiệp, cấu trúc và sự phù hợp với mong đợi của người dùng.

# Văn Bản:
Các mẫu mở Gemma được xây dựng từ cùng một nghiên cứu _____ công nghệ như các mẫu Gemini. Gemma 2 có các kích cỡ 2B, 9B và 27B và Gemma 1 có các kích cỡ 2B và 7B.<end_of_turn>
<start_of_turn>model
## **Tiêu Đề:**
### Gemma open models

## **Mô Tả:**
Gemma: Introducing new state-of-the-art open models.

## **Bài Viết Chỉnh Sửa:**
A new family of open language models demonstrating strong performance across academic benchmarks for language understanding, reasoning, and safety.

## **Điểm Nổi Bật:**
* Main point 1
* Main point 2

## **Danh Mục:**
* Artificial Intelligence
* Gemma

## **Từ Khoá:**
* AI
* LLM
* Google<end_of_turn>

```
