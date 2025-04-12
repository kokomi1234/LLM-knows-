from transformers.utils import TRANSFORMERS_CACHE
import os

print(f"默认缓存位置: {TRANSFORMERS_CACHE}")

# 检查缓存目录是否存在
if os.path.exists(TRANSFORMERS_CACHE):
    print(f"目录存在，内容:")
    for item in os.listdir(TRANSFORMERS_CACHE):
        print(f" - {item}")
    
    # 检查具体模型目录
    models_dir = os.path.join(TRANSFORMERS_CACHE, "models--mistralai--Mistral-7B-Instruct-v0.2")
    if os.path.exists(models_dir):
        print(f"\n找到模型目录: {models_dir}")
        print("包含文件:")
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                print(f" - {os.path.join(root, file)}")
else:
    print("缓存目录不存在")

# 检查环境变量
for env_var in ["TRANSFORMERS_CACHE", "HF_HOME", "HF_DATASETS_CACHE"]:
    value = os.environ.get(env_var)
    print(f"{env_var}: {value if value else '未设置'}")