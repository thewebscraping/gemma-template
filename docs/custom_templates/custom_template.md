# Custom Templates to Vietnamese Language
Gemma Template uses Jinja2 template.

See also: [`models.Attr`](../../models/#attributes_5)

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
# Văn Bản:
{{ input }}
{% if topic_value %}\nDanh Mục: {{ topic_value }}\n{% endif %}{% if keyword_value %}Từ Khoá: {{ keyword_value }}\n{% endif %}
```

* * *

# Output Template
```text
{% if structure_fields %}{% for field in structure_fields %}## **{{ field.label.custom or field.label.default }}:**\n{% if field.key == 'title' %}### {% endif%}{{ field.value }}\n\n{% endfor %}{% else %}{{ output }}{% endif %}
```

* * *

# Instruction Template
```text
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
```

* * *

# Prompt Template
```text
{% if prompt %}\n\n# Đầu Vào Văn Bản:\n{{ prompt }}\n\n{% endif %}{% if structure_fields %}# Định Dạng Cấu Trúc Phản Hồi:
Bạn phải tuân theo cấu trúc phản hồi:

{% for field in structure_fields %}{{ field.label }}\n{% endfor %}

Bằng cách tuân thủ định dạng này, phản hồi sẽ duy trì tính toàn vẹn về mặt ngôn ngữ đồng thời tăng cường tính chuyên nghiệp, cấu trúc và sự phù hợp với mong đợi của người dùng.
{% endif %}
```
