# CTRRecommenderModels (ongoing)

## 经验总结和前沿研究调研

对学术界和工业界的推荐系统进行了系统性总结，形成了《特征工程》、《召回》和《排序》三个章节，包括技术要点和前沿研究。

### 《特征工程》前沿论文：

Multi-modal Representation Learning for Short Video Understanding and Recommendation. ICME Workshops 2019.
An Embedding Learning Framework for Numerical Features in CTR Prediction, KDD 2021.

Dynamic Explicit Embedding Representation for Numerical Features in Deep CTR Prediction, CIKM 2022.

Numerical Feature Representation with Hybrid 𝑁 -ary Encoding, CIKM 2022.

AutoFeature: Searching for Feature Interactions and Their Architectures for Click-through Rate Prediction, CIKM 2020.

Towards Automated Neural Interaction Discovery for Click-Through Rate Prediction, KDD 2020.

AutoGroup: Automatic Feature Grouping for Modelling Explicit High-Order Feature Interactions in CTR Prediction, SIGIR 2020.

Cognitive Evolutionary Search to Select Feature Interactions for Click-Through Rate Prediction, KDD 2023.

AdnFM: An Attentive DenseNet based Factorization Machine for Click-Through-Rate Prediction, ICCDE 2022.

CAN:Feature Co-Action Network for Click-Through Rate Prediction, WSDM 2022.

Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models , DLP-KDD 2021.

FINAL: Factorized Interaction Layer for CTR Prediction, SIGIR 2023.

AdaFS: Adaptive Feature Selection in Deep Recommender System, KDD 2022.

LPFS:Learnable Polarizing Feature Selection for Click-Through Rate Prediction, 2022.

Optimizing Feature Set for Click-Through Rate Prediction, WWW 2023.

Automatic Feature Selection By One-Shot Neural Architecture Search In Recommendation Systems, WWW 2023.

Catch: Collaborative Feature Set Search for Automated Feature Engineering, WWW 2023.



### 我实现25个推荐CTR经典模型代码(开箱即用-你需要进一步调优，我的运行环境是mac m1 + python 3.9，所有代码都本地完成测试)，这个库后续继续更新；

I have implemented some common-used CTR / recommender models for reusage, including 25 models as follows:

#### a. 4个常用机器学习集成模型：随机森林、Xgboost、lightgbm和catboost，以及使用hyperopt和bayesian-optimization进行超参数调优。（这部分基于sklearn包和相应python包实现调用）

#### b. 5个基础模型：Matrix Factorizatin (MF)、SVD、Factorization Machine（FM）、NeuralCF（WWW 2017）、AutoencoderRec（WWW 2015）。

#### c. 8个深度网络模型：DeepFM（IJCAI 2017）、DSSM（CIKM 2013）、Wide & Deep（RS 2016）、DeepCross（DCN，KDD 2016）、Attentive Factorization Machine（AFM，IJCAI 2017）、Product-based Neural Network（PNN，ICDM 2016）、Neural Factorization Machine（NFM，SIGIR 2017）、FiBiNET（RS 2019）。

#### d. 5个序列推荐模型：GRU4Rec（ICLR 2016）、Deep Interest Network（DIN，KDD 2018）、Deep Interest Evolution Network（DIEN，AAAI 2018）、Self-attentive Sequential Recommendation（SASRec，ICDM 2018）、Behavior Sequence Transformer（BSTransformer，2019）。

#### e. 3个多兴趣偏好模型：Multi-interest network with dynamic routing（MIND，2019）、Controllable Multi-Interest Framework for Recommendation（Comirec，KDD 2020）、Sparse-Interest Network（SINE，WSDM 2021）。
解决一个用户兴趣向量很难捕获用户多方面兴趣的问题（特别是从用户历史长行为序列中捕捉多方面兴趣偏好），从用户历史行为序列中得到多个兴趣偏好。当用户历史行为序列较短时（<50）可以采用各种常规序列模型（如GRU、attention序列模型之类），当用户历史行为序列较长时，需要考虑效率，如利用target item来检索相似相近的历史items并进行序列建模。建模用户多方面偏好类似于聚类效果，采用胶囊网络、多个选择通路(如top-k个激活兴趣)等等手段，每次激活一个通路或多个通路（即兴趣点），采用hard方式或者soft方式（如注意力）。

#### f. 4个多任务学习模型：Entire-space multi-task model（ESSM，SIGIR 2020）、Multi-gate MoE Mixture-of-Experts（MMOE，KDD 2018）、Customized Gate Control（CGC，RS 2020）、Audience Multi-step Conversions with Multi-task Learning（AITM，KDD 2021）。

多任务学习一般在实际工作中效果提升比较大的一种方式，找到场景下一些相关的任务，基于任务之间的关系特点来设计多任务共享结构，这里面有很多设计的空间，如共享模块可以是底层embedding共享、中间层共享或高层共享，共享程度大小等等，不同任务loss之间的比例，还有提高采样的效率等。共享的方式是hard还是soft等。注意一点就是根据不同任务之间相关性强弱，设计合理架构，避免负迁移。


根据这几年大厂论文，主要集中在挖掘用户超长行为序列（同时考虑效率和效益，用于精排）、多兴趣偏好（用于召回）、多任务学习（模型sharing结构设计，主要用于精排）等，特征工程（特征离散化和特征交互）的文章相对较少。


AutoRec - Autoencoders Meet Collaborative Filtering, WWW 2015.

Factorization Machines: Fast Context-aware Recommendations with Factorization Machines, SIGIR 2011.

DSSM: Learning deep structured semantic models for web search using clickthrough data, CIKM 2013.

<img width="1036" alt="DSSM" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/a6181c3f-108c-44b1-944c-8bfb15777b7e">

NeuralCF: Neural Collaborative Filtering, WWW 2017.

<img width="706" alt="NeuralCF" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/1dd5dcc1-390c-4e01-abc9-f8aaeb25f05f">


Wide&Deep: Wide & deep learning for recommender systems, RS 2016.

<img width="683" alt="Wide Deep" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/d8fa6431-8230-4f0b-bda6-0ac3e277ae36">


DeepFM: Deepfm: a factorization-machine based neural network for ctr prediction,  IJCAI 2017.

<img width="688" alt="DeepFM" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/a2ccc855-3d96-481d-b2e8-e9c93d325181">

DeepCross: Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features, KDD 2016.

![DeepCross DCN模型](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/96f05c1d-21a4-46d8-9c7a-951ccf147dd5)


AFM: Attentive Factorization Machine, IJCAI 2017.

![AFM-2](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/1b2a3962-4049-4cce-a0d1-2a9182da85c7)

NFM: Neural Factorization Machines for Sparse Predictive Analytics, SIGIR 2017.

![NFM](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/9c19319f-ea62-4218-a442-275de422f784)

FiBiNET: FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction, RS 2019.

<img width="720" alt="FiBiNET" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/863183f3-aa8b-4bf3-bbf9-0a2970e66aa2">


PNN: Product-based Neural Networks for User Response Prediction, ICDM 2016.

<img width="531" alt="PNN" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/72b2abc0-b72d-49a9-845b-8116d278ef5b">


GRU4Rec: Session-based Recommendations with Recurrent Neural Networks, ICLR 2016.

<img width="566" alt="GRU4REC" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/871915c8-192e-4922-ba97-7947ddc98f01">


Caser：Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, WSDM 2018.

<img width="690" alt="Caser" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/e671ac1b-b58e-43d9-9331-181685c23c72">


DIN: Deep Interest Network for Click-Through Rate Prediction, KDD 2018.

<img width="692" alt="DIN" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/dafb6b0e-bd64-4c05-9ce6-3fa16d8f32cc">


DIEN: Deep Interest Evolution Network for Click-Through Rate Prediction, AAAI 2018.

<img width="689" alt="DIEN" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/55e7c6b9-e8f2-4594-9195-ddff4c42c73d">


SASRec: Self-attentive Sequential Recommendation, ICDM 2018.

<img width="588" alt="SASRec" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/4ab1aad5-77b4-40b0-835e-a1a118f18b3c">

BSTransformer: Behavior Sequence Transformer for E-commerce Recommendation in Alibaba, 2019.

![BST](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/0e14913b-e770-4ce0-b87b-1a6b62dcaad9)


MIND：Multi-interest network with dynamic routing for recommendation at Tmall, 2019.

![MIND](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/3089993c-d291-47b8-95af-e61e81e6d86e)


Comirec：Controllable Multi-Interest Framework for Recommendation， KDD 2020.

![Comirec](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/08884e82-b3c2-4c7a-b0c8-949c6c9150ff)


SINE: Sparse-Interest Network for Sequential Recommendation, WSDM 2021.

<img width="1308" alt="SINE" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/ed8f9505-00b2-487a-822e-ecedf8de5f20">



ESSM：Entire Space Multi-task Modeling via Post-Click Behavior Decomposition for Conversion Rate Prediction, SIGIR 2020.

![ESSM](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/a7217768-d1f1-43af-9db0-2ed3768c199c)

MMOE：Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts, KDD 2018.

![MMOE](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/b32c61dd-cef2-4fc1-913f-802580d3741c)

CGC：Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations, RS 2020.

![CGC](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/1aade055-a884-4723-b868-ff87d557f1f5)


AITM: Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising, KDD 2021.

<img width="1625" alt="AITM" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/737a5eac-d50a-4842-a5e3-46b1dc9d6807">


The project is ongoing ......
