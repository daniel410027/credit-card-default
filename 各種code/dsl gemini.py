import google.generativeai as genai
import os

# 1.configuration
genai.configure(api_key = os.getenv("gemini_api"))
generation_config = {"temperature": 0.9, "top_p" : 1, "max_output_tokens" : 99999}

# 2.initialise model
model = genai.GenerativeModel("gemini-pro", generation_config = generation_config)
content = 'classify above object into 16 career'
# 定義目標字符串和列表
target_string = content
list_to_insert = df.iloc[:, 7]

# 將列表元素插入到目標字符串中的特定位置
# 這裡我們將列表元素插入到字符串中的第 5 個字符後面
insert_position = 0
result_string = target_string[:insert_position] + ' '.join(list_to_insert) + target_string[insert_position:]

# 輸出結果
# print(result_string)


# 3.generate content
response = model.generate_content([content])
print(response.text )
a = response.text

