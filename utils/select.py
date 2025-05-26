import os
import random
import shutil

# 数据集目录路径
celeba_hq_dir = '/data/IP-retrieval/dataset/celeba-hq'
ffhq_dir = '/data/IP-retrieval/dataset/ffhq'
output_dir = '/data/IP-retrieval/dataset/new'

# 确保目标文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 总共选择的图片数量
total_images = 30000

# 比例计算
celeba_hq_count = int(total_images * 0.3)  # 30% 来自 celeba-hq
ffhq_count = total_images - celeba_hq_count  # 70% 来自 ffhq

# 获取两个数据集中的所有图片文件
celeba_hq_images = [f for f in os.listdir(celeba_hq_dir) if os.path.isfile(os.path.join(celeba_hq_dir, f))]
ffhq_images = [f for f in os.listdir(ffhq_dir) if os.path.isfile(os.path.join(ffhq_dir, f))]

# 随机选择图片
selected_celeba_hq = random.sample(celeba_hq_images, celeba_hq_count)
selected_ffhq = random.sample(ffhq_images, ffhq_count)

# 函数：复制选中的图片到目标文件夹
def copy_images(image_list, source_dir):
    for image in image_list:
        src_path = os.path.join(source_dir, image)
        dest_path = os.path.join(output_dir, image)
        shutil.copy(src_path, dest_path)
        print(f'复制: {src_path} -> {dest_path}')

# 复制选中的图片
copy_images(selected_celeba_hq, celeba_hq_dir)
copy_images(selected_ffhq, ffhq_dir)

print("图片复制完成！")
