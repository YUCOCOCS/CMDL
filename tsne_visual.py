from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


embeddings = np.load('/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/test_picture/s/seg_embeddings_1.npy') # 加载特征
labels = np.load('/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/test_picture/s/seg_labels_1.npy')  # 加载标签

np.random.seed(0)
uni_labels = np.unique(labels)
print(uni_labels)
dictx = {}

overall_embeddings = []
overall_labels = []

for key in uni_labels: # 对每个类别进行遍历
    index = (labels == key).squeeze(axis=-1) # 得到属于类别key的位置
    num = index.sum().item() # 计算有多少个类别像素属于key
    di ={}
    di['key'] = num
    print(di)
    class_embed = embeddings[index]
    class_labels = labels[index]

    index = np.random.choice(num, size=200, replace=True) # 指的是从0-num中随机抽取1000个数字形成一个一维数组
    overall_embeddings.append(class_embed[index])# 每个类对应的特征
    overall_labels.append(class_labels[index]) # 每个类对应的索引下标

overall_embeddings = np.concatenate(overall_embeddings,axis=0)
overall_labels = np.concatenate(overall_labels,axis=0)

np.save('/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/test_picture/s/seg_class_embed_1.npy', overall_embeddings)
np.save('/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/test_picture/s/seg_class_label_1.npy', overall_labels)

print(overall_embeddings.shape)
print(overall_labels.shape)



matplotlib.rcParams['font.family']='Times New Roman'
matplotlib.rcParams['font.size']=30


np.random.seed(1968081)

net1_embeddings = np.load('/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/test_picture/s/seg_class_embed_1.npy') # size: (sample_num, embedding_dim)
net1_target = np.load('/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/test_picture/s/seg_class_label_1.npy') # size: (sample_num, )

net1_target = net1_target.squeeze(axis=-1)
target_value = list(set(net1_target))  # 标签变成一个列表


color_dict = {}
colors = ['black', 'red', 'gold', 'green', 'orange', 'pink','magenta', 'slategray', 'greenyellow', 'lightgreen',
           'brown', 'chocolate', 'mediumvioletred', 'navy', 'lightseagreen',]
# colors = ['black', 'red', 'gold', 'green', 'orange', 'pink', 'magenta', 'slategray', 'greenyellow', 'lightgreen',
#           'brown', 'chocolate', 'mediumvioletred', 'navy', 'lightseagreen', 'aqua', 'olive', 'maroon', 'yellow']
for i, t in enumerate(target_value):
    color_dict[t] = colors[i]
print(color_dict)

net1 = TSNE(early_exaggeration=4).fit_transform(net1_embeddings) #early_exaggeration的值指代的是簇之间的间距大小
np.save('/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/test_picture/s/tsne_1.npy', net1)


net1 = np.load('/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/test_picture/s/tsne_1.npy')

for i in range(len(target_value)): # 把对应类别的像素选出来
    tmp_X1 = net1[net1_target==target_value[i]]
    plt.scatter(tmp_X1[:, 0], tmp_X1[:, 1], c=color_dict[target_value[i]], marker='o')

plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("/home/y212202015/SSEG/pre/sssegmentation-main/ssseg/test_picture/s/tsne_1.png")

