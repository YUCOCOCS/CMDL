

def extract_filenames(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.split()  # 以空格分隔
            if len(parts) == 2:
                filename = parts[0].split('/')[-1].split('.')[0]  # 提取文件名部分
                outfile.write(filename + '\n')  # 写入到输出文件中

# 示例使用
input_file = '/data/CamVid/val.txt'  # 输入文件名
output_file = '/data/CamVid/Val.txt'  # 输出文件名
extract_filenames(input_file, output_file)