import torch
import json
import os
from FlagEmbedding.visual.modeling import Visualized_BGE

# 加载模型
model = Visualized_BGE(
    model_name_bge="/work/my_models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
    model_weight="/work/my_models/bge_visualized/Visualized_m3.pth"
)

# 读取JSON文件获取图片描述
json_path = "./asl_data/asl_text_descriptions.json"
image_folder = os.path.join(os.path.dirname(json_path), "images")

with open(json_path, "r") as json_file:
    asl_descriptions = json.load(json_file)

# 获取图片名称及路径
image_paths = {key: os.path.join(image_folder, f"{key}.jpg") for key in asl_descriptions.keys()}

# 编码所有图片
image_embeds = {}
image_embeds_with_text = {}
with torch.no_grad():
    for key, img_path in image_paths.items():
        # 仅使用图片进行编码
        img_emb = model.encode(image=img_path)
        image_embeds[key] = img_emb

        # 获取图片的描述，并将图片+描述一起编码
        description = asl_descriptions[key]
        img_emb_with_text = model.encode(image=img_path, text=description)
        image_embeds_with_text[key] = img_emb_with_text        

# 定义相似度计算函数（使用余弦相似度）
def calculate_cosine_similarity(text_emb, img_emb):
    return torch.nn.functional.cosine_similarity(text_emb, img_emb, dim=-1)

# 定义相似度计算函数：点乘相似度
def calculate_dot_product_similarity(text_emb, img_emb):
    return torch.matmul(text_emb, img_emb.T)

# 循环遍历字母A到Z，再加上SPACE
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE"]

for letter in alphabet:
    # 生成中文和英文的文本
    text_emb1 = f"怎么用ASL手势表示字母{letter}？"
    candi_text_emb1 = model.encode(text=text_emb1)
    text_emb1_en = f"How can I sign a {letter}?"
    candi_text_emb1_en = model.encode(text=text_emb1_en)

    print(f"\n========================= 现在处理字母：{letter} =========================")
    
    # 1. 中文文本与图片的 **余弦相似度** 排序 (不带文本描述)
    similarities_cn_cosine = []
    for key, img_emb in image_embeds.items():
        sim = calculate_cosine_similarity(candi_text_emb1, img_emb)
        similarities_cn_cosine.append((f"中文与图片 {key} 的余弦相似度", sim.item()))

    sorted_similarities_cn_cosine = sorted(similarities_cn_cosine, key=lambda x: x[1], reverse=True)
    print(f"中文与图片的余弦相似度排序（不带文本描述），中文内容：{text_emb1}")
    for label, sim in sorted_similarities_cn_cosine:
        print(f"{label}: {sim}")

    # 2. 中文文本与图片的 **余弦相似度** 排序 (携带文本描述)
    similarities_cn_cosine_with_text = []
    for key, img_emb_with_text in image_embeds_with_text.items():
        sim = calculate_cosine_similarity(candi_text_emb1, img_emb_with_text)
        similarities_cn_cosine_with_text.append((f"中文与图片 {key} 的余弦相似度 (携带文本描述)", sim.item()))

    sorted_similarities_cn_cosine_with_text = sorted(similarities_cn_cosine_with_text, key=lambda x: x[1], reverse=True)
    print("-------------------------------------------")
    print(f"中文与图片的余弦相似度排序（携带文本描述），中文内容：{text_emb1}")
    for label, sim in sorted_similarities_cn_cosine_with_text:
        print(f"{label}: {sim}")

    # 3. 英文文本与图片的 **余弦相似度** 排序 (不带文本描述)
    similarities_en_cosine = []
    for key, img_emb in image_embeds.items():
        sim = calculate_cosine_similarity(candi_text_emb1_en, img_emb)
        similarities_en_cosine.append((f"英文与图片 {key} 的余弦相似度", sim.item()))

    sorted_similarities_en_cosine = sorted(similarities_en_cosine, key=lambda x: x[1], reverse=True)
    print("-------------------------------------------")
    print(f"英文与图片的余弦相似度排序（不带文本描述），英文内容：{text_emb1_en}")
    for label, sim in sorted_similarities_en_cosine:
        print(f"{label}: {sim}")

    # 4. 英文文本与图片的 **余弦相似度** 排序 (携带文本描述)
    similarities_en_cosine_with_text = []
    for key, img_emb_with_text in image_embeds_with_text.items():
        sim = calculate_cosine_similarity(candi_text_emb1_en, img_emb_with_text)
        similarities_en_cosine_with_text.append((f"英文与图片 {key} 的余弦相似度 (携带文本描述)", sim.item()))

    sorted_similarities_en_cosine_with_text = sorted(similarities_en_cosine_with_text, key=lambda x: x[1], reverse=True)
    print("-------------------------------------------")
    print(f"英文与图片的余弦相似度排序（携带文本描述），英文内容：{text_emb1_en}")
    for label, sim in sorted_similarities_en_cosine_with_text:
        print(f"{label}: {sim}")

    # 5. 中文文本与图片的 **点乘相似度** 排序 (不带文本描述)
    similarities_cn_dot = []
    for key, img_emb in image_embeds.items():
        sim = calculate_dot_product_similarity(candi_text_emb1, img_emb)
        similarities_cn_dot.append((f"中文与图片 {key} 的点乘相似度", sim.item()))

    sorted_similarities_cn_dot = sorted(similarities_cn_dot, key=lambda x: x[1], reverse=True)
    print("-------------------------------------------")
    print(f"中文与图片的点乘相似度排序（不带文本描述），中文内容：{text_emb1}")
    for label, sim in sorted_similarities_cn_dot:
        print(f"{label}: {sim}")

    # 6. 中文文本与图片的 **点乘相似度** 排序 (携带文本描述)
    similarities_cn_dot_with_text = []
    for key, img_emb_with_text in image_embeds_with_text.items():
        sim = calculate_dot_product_similarity(candi_text_emb1, img_emb_with_text)
        similarities_cn_dot_with_text.append((f"中文与图片 {key} 的点乘相似度 (携带文本描述)", sim.item()))

    sorted_similarities_cn_dot_with_text = sorted(similarities_cn_dot_with_text, key=lambda x: x[1], reverse=True)
    print("-------------------------------------------")
    print(f"中文与图片的点乘相似度排序（携带文本描述），中文内容：{text_emb1}")
    for label, sim in sorted_similarities_cn_dot_with_text:
        print(f"{label}: {sim}")

    # 7. 英文文本与图片的 **点乘相似度** 排序 (不带文本描述)
    similarities_en_dot = []
    for key, img_emb in image_embeds.items():
        sim = calculate_dot_product_similarity(candi_text_emb1_en, img_emb)
        similarities_en_dot.append((f"英文与图片 {key} 的点乘相似度", sim.item()))

    sorted_similarities_en_dot = sorted(similarities_en_dot, key=lambda x: x[1], reverse=True)
    print("-------------------------------------------")
    print(f"英文与图片的点乘相似度排序（不带文本描述），英文内容：{text_emb1_en}")
    for label, sim in sorted_similarities_en_dot:
        print(f"{label}: {sim}")

    # 8. 英文文本与图片的 **点乘相似度** 排序 (携带文本描述)
    similarities_en_dot_with_text = []
    for key, img_emb_with_text in image_embeds_with_text.items():
        sim = calculate_dot_product_similarity(candi_text_emb1_en, img_emb_with_text)
        similarities_en_dot_with_text.append((f"英文与图片 {key} 的点乘相似度 (携带文本描述)", sim.item()))

    sorted_similarities_en_dot_with_text = sorted(similarities_en_dot_with_text, key=lambda x: x[1], reverse=True)
    print("-------------------------------------------")
    print(f"英文与图片的点乘相似度排序（携带文本描述），英文内容：{text_emb1_en}")
    for label, sim in sorted_similarities_en_dot_with_text:
        print(f"{label}: {sim}")
