from sklearn.feature_extraction.text import CountVectorizer
# 假设 x 是一个 pandas Series 包含了文本数据
x = df['job']
# 创建 CountVectorizer 实例，并设置 min_df=0.5
vect = CountVectorizer(min_df=0.002)
# 将文本数据转换为词频矩阵
word_count_matrix = vect.fit_transform(x)
# 输出词频矩阵的形状
print(word_count_matrix.shape)
# 获取保留下来的单词列表
feature_names = vect.get_feature_names_out()

# 输出保留下来的单词列表
print(feature_names)

df2 = df['job'].copy()

for i in range(len(df2)):
    df2[i] = df2[i].lower().split()
    # 保留 df2[i] 中在 feature_names 中出现的单词
    df2[i] = [word for word in df2[i] if word in feature_names]

# 假设 df2 是你的 Series 对象
count_empty_lists = sum(1 for item in df2 if item == [])
print("空列表的数量：", count_empty_lists)

