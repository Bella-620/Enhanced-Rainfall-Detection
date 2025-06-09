import os
import random
import shutil


org_img_folder = '/root/autodl-tmp/handan_cwt_picture/train_dry_all'  # 待处理源文件夹路径
# 获取源文件夹及其子文件夹中图片列表
Filelist = []
for s in os.listdir(org_img_folder):
    newDir = os.path.join(org_img_folder, s)
    Filelist.append(newDir)

tar_img_folder = '/root/autodl-tmp/handan_cwt_picture/train/dry'     # 移动到新文件夹路径
picknumber = 1126  # 需要从源文件夹中抽取的图片数量
img_format = 'jpg'  # 需要处理的图片后缀

print(Filelist)
print(len(Filelist))

samplelist = random.sample(Filelist, picknumber)  # 获取随机抽样后的图片列表

print('本次执行检索到 ' + str(len(Filelist)) + ' 张图像\n')
print('本次共随机抽取 ' + str(len(samplelist)) + ' 张图像\n')

# 复制选取好的图片到新文件夹中
new_img_folder = tar_img_folder
for imgpath in samplelist:
    shutil.copy(imgpath, new_img_folder)  # 复制图片到新文件夹