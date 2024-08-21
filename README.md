# Semantic Segmentation with Multiple Distributions
  Abstract:— Semantic segmentation methods that utilize contextual information to enhance pixel feature representations have made significant progress recently. However, these methods rely solely on a single class feature to describe the overall distribution of the corresponding class, which prevents the network from learning the rich intra-class variance information. To address this limitation, we propose a Class-wise Multiple Distributions Learning (CMDL) method to capture the diversity of intra-class features. Specifically, we design a class multiple distributions consistency strategy to extract the multiple distributions feature of each class and enhance the discriminability of classification boundaries in the feature embedding space, enabling the backbone network to produce more robust features.  To fully leverage the rich information within the multiple distributions feature of each class, we further propose a feature voting and a multiple distributions semantic aggregation module. The former injects the feature information of each class's single distribution into the basic pixel feature representations to obtain several single distribution features, while the latter  semantically aggregates these single distribution features to achieve more  fine-grained multiple distribution feature representations. Additionally, our method can serve as a plug-and-play component that can be integrated into existing semantic segmentation frameworks, resulting in significant performance improvements. Experimental results on four standard datasets—Cityscapes, ADE20K, COCO-Stuff, and CamVid—demonstrate that our proposed method achieves state-of-the-art performance.
  

# Pipeline
 ![Network](https://github.com/YUCOCOCS/CMDL/blob/main/Net.png)



# Datasets
We have demonstrated state-of-the-art experimental performance of our method on ADE20K, COCO-Stuff, and CamVid and Cityscapes datasets.
You can download the ADE20K on [this](https://groups.csail.mit.edu/vision/datasets/ADE20K).

You can download the COCO-Stuff on [this](https://github.com/nightrome/cocostuff10k).

You can download the CamVid on [this](https://download.csdn.net/download/u011406896/87764336?utm_source=bbsseo&ydreferer=aHR0cHM6Ly9iYnMuY3Nkbi5uZXQvdG9waWNzLzYxNTQ4OTU2Ng%3D%3D).

You can download the Cityscapes on [this](https://www.cityscapes-dataset.com/).


# Training 
     CUDA_VISIBLE_DEVICES=0,1,2,3 nohup  python -m torch.distributed.launch --nproc_per_node=4 --master_port=1001   train.py >train.log &


