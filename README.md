本代码是多模态RAG的示例，通过语言检索图像。并跟本地多模态大模型进行交互获取最终响应。

alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE"]

for letter in alphabet:
    代码逐个字母查询 “怎么用ASL手势表示字母{letter}”

代码逐个字母对比所有多模态图片的Embedding内容，并且给出相似度分数。上面例子可以看出来27个图片的多模态查询准确率 可以达到100%
