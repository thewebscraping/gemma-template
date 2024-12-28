# fmt: off
from __future__ import annotations

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English", "af": "Afrikaans", "sq": "Albanian", "ar": "Arabic", "bn": "Bangla", "bg": "Bulgarian",
    "ca": "Catalan", "zh-cn": "Chinese", "zh-tw": "Chinese Traditional", "hr": "Croatian", "cs": "Czech",
    "da": "Danish", "nl": "Dutch", "et": "Estonian", "fi": "Finnish", "fr": "French", "de": "German",
    "el": "Greek", "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi",  "hu": "Hungarian", "id": "Indonesian",  # noqa: E241
    "it": "Italian", "ja": "Japanese", "kn": "Kannada", "ko": "Korean", "lv": "Latvian", "lt": "Lithuanian",
    "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali", "no": "Norwegian", "fa": "Persian",
    "pt": "Portuguese", "pl": "Polish", "pa": "Punjabi", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "es": "Spanish", "sw": "Swahili", "sv": "Swedish", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tr": "Turkish", "tl": "Tagalog", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "cy": "Welsh",
}
# fmt: on

GEMMA_TEMPLATE = """<start_of_turn>user
{user_template}<end_of_turn>
<start_of_turn>model
{model_template}<end_of_turn>"""

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

STRUCTURE_TEMPLATE = """# Response Structure Format:
You must follow the response structure outlined below to ensure clarity and alignment with user expectations:
{structure_template}

By adhering to this format, the response will maintain linguistic integrity while enhancing professionalism and structure.
"""  # noqa: E501

USER_TEMPLATE = """{system_template}

{prompt_template}

{instruction_template}
{structure_template}
# Text:
{document}
"""

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

Danh mục: {topic_values}
Từ khoá: {keyword_values}

# Phân Tích Văn Bản:
Ví Dụ 1: Unigrams (nhóm 1 chữ cái)
{unigrams}

Phân Tích Văn Bản 1: đây là những từ thông dụng trong tiếng Việt ({language}), cho biết văn bản được viết bằng Tiếng Việt ({language}).

Ví Dụ 2: Bigrams (nhóm 2 chữ cái)
{bigrams}

Phân Tích Văn Bản 2: các từ ghép thường gặp trong tiếng Việt ({language}) xác nhận bối cảnh ngôn ngữ.

Ví Dụ 3: Trigrams (nhóm 3 chữ cái)
{trigrams}

Phân Tích Văn Bản 3: các từ ghép 3 chữ liên tiếp là những từ tiếng Việt sử dụng thường xuyên, xác nhận sự cần thiết phải phản hồi bằng Tiếng Việt ({language}).

# Kết Luận Phân Tích Văn Bản:
Phân tích ngôn ngữ xác nhận văn bản chủ yếu bằng Tiếng Việt ({language}). Do đó, phản hồi phải được cấu trúc và viết bằng Tiếng Việt ({language}). để phù hợp với văn bản và ngữ cảnh gốc.
"""  # noqa: E501

VIETNAMESE_STRUCTURE_TEMPLATE = """# Định Dạng Cấu Trúc Phản Hồi:
{structure_template}

Bằng cách tuân thủ định dạng này, phản hồi sẽ duy trì được tính toàn vẹn về mặt ngôn ngữ đồng thời nâng cao tính chuyên nghiệp và cấu trúc phản hồi.
"""  # noqa: E501

VIETNAMESE_USER_TEMPLATE = """{system_template}

{prompt_template}

{instruction_template}
{structure_template}
# Văn Bản:
{document}
"""
