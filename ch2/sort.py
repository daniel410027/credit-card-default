# 刪除頭尾的星號
original_str = a.strip('*')

# 按照 '**' 分割字符串，得到 key-value 對列表
pairs = original_str.split('**')

# 將列表中的每個 key-value 對分割成 key 和 value
key_value_pairs = [pair.split('**') for pair in pairs]

# 將列表中的每個 key-value 對分割成 key 和 value
result_dict = {}
for i in range(0, len(pairs), 2):
    key = pairs[i]
    value = pairs[i+1]
    result_dict[key] = value

# 輸出字典
print(result_dict)

# 定義轉換函數
def transform_value(value):
    # 將句子拆分成單字
    # words = result_dict.values()
    # 將字串轉換為小寫並拆分成單字
    value_lower = value.lower().split()
    # 檢查每個單字是否與分類中的 value 中的任一單字相匹配
    for key, sentence in result_dict.items():
        for word in sentence.split():
            if word.lower() in value_lower:
                return key
    # 如果沒有匹配的，返回原值
    # return value
    return '17. other'

# 將轉換應用於 DataFrame 中的第 7 列
df.iloc[:, 7] = df.iloc[:, 7].apply(lambda x: transform_value(x))

# 再來要把key丟進value裡面,再打單字切乾淨一點,然後就完成了(?

