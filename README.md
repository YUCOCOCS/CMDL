# Semantic Segmentation with Multiple Distributions
  Abstract:— Semantic segmentation methods that utilize contextual information to enhance pixel feature representations have made significant progress recently. However, these methods rely solely on a single class feature to describe the overall distribution of the corresponding class, which prevents the network from learning the rich intra-class variance information. To address this limitation, we propose a Class-wise Multiple Distributions Learning (CMDL) method to capture the diversity of intra-class features. Specifically, we design a class multiple distributions consistency strategy to extract the multiple distributions feature of each class and enhance the discriminability of classification boundaries in the feature embedding space, enabling the backbone network to produce more robust features.  To fully leverage the rich information within the multiple distributions feature of each class, we further propose a feature voting and a multiple distributions semantic aggregation module. The former injects the feature information of each class's single distribution into the basic pixel feature representations to obtain several single distribution features, while the latter  semantically aggregates these single distribution features to achieve more  fine-grained multiple distribution feature representations. Additionally, our method can serve as a plug-and-play component that can be integrated into existing semantic segmentation frameworks, resulting in significant performance improvements. Experimental results on four standard datasets—Cityscapes, ADE20K, COCO-Stuff, and CamVid—demonstrate that our proposed method achieves state-of-the-art performance.
  

# Pipeline
 ![Network](https://github.com/MKSAQW/MLCL/blob/main/Network.png)

 # Installation
> pip install -r requirements.txt






