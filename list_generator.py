import os
 
# 定义变量存储文件目录
# 此时文件是存储在项目下 路径直接写文件名即可
file_path = '/data/CamVid/train'  # filename根据具体的文件名修改
 
# # 如果文件不在本项目目录中，则可采用下面的方式
# file_path = 'E:/1/2/3'  # 填写文件的绝对路径
 
# 获取文件名
filename_list = os.listdir(file_path)
 
# # 输出获取的文件名 输出的是list类型数据
# print('filename:', filename_list)
 
# 打开TXT文件，并将路径写入
# 读写模式有很多种，比如：r,r+,w,w+,a,a+ w只对文件进行全覆盖写入，如果文件不存在则创建
# 加入编码格式，防止读取中文时乱码
with open('/data/CamVid/train.txt', 'w', encoding='utf-8') as f:  # txtname 根据具体所需的命名即可
    for val_name in filename_list:
        f.write(val_name.split(".")[0] + '\n')
 
# 最后输出finish 表示任务已经执行完成
print('finish')