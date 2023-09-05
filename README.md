# CTRRecommenderModels (ongoing)

我实现23个推荐CTR经典模型代码(开箱即用-你需要进一步调优，我的运行环境是mac m1 + python 3.9，所有代码都本地完成测试)，这个库后续继续更新；

I have implemented some common-used CTR / recommender models for reusage, including 23 models as follows:

# 3个常用机器学习集成模型：随机森林、Xgboost和lightgbm，以及使用hyperopt和bayesian-optimization进行超参数调优。（这部分基于sklearn包和相应python包实现调用）

# 5个基础模型：Matrix Factorizatin (MF)、SVD、Factorization Machine（FM）、NeuralCF、AutoencoderRec。

# 8个深度网络模型：DeepFM、DSSM、Wide & Deep、DeepCross（DCN）、Attentive Factorization Machine（AFM）、Product-based Neural Network（PNN）、Neural Factorization Machine（NFM）、FiBiNET。

# 5个序列推荐模型：GRU4Rec、Deep Interest Network（DIN）、Deep Interest Evolution Network（DIEN）、Self-attentive Sequential Recommendation（SASRec）、Behavior Sequence Transformer（BSTransformer）。

# 2个多兴趣偏好模型：Multi-interest network with dynamic routing（MIND）、Controllable Multi-Interest Framework for Recommendation（Comirec）。

# 3个多任务学习模型：Entire-space multi-task model（ESSM）、Multi-gate MoE Mixture-of-Experts（MMOE）、Customized Gate Control（CGC）。

NeuralCF:

<img width="706" alt="NeuralCF" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/1dd5dcc1-390c-4e01-abc9-f8aaeb25f05f">


DeepFM:

<img width="688" alt="DeepFM" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/a2ccc855-3d96-481d-b2e8-e9c93d325181">

DeepCross:

![DeepCross DCN模型](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/96f05c1d-21a4-46d8-9c7a-951ccf147dd5)


AFM:

![AFM-2](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/1b2a3962-4049-4cce-a0d1-2a9182da85c7)

NFM:

![NFM](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/9c19319f-ea62-4218-a442-275de422f784)

FiBiNET:

<img width="720" alt="FiBiNET" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/863183f3-aa8b-4bf3-bbf9-0a2970e66aa2">



DIN:

<img width="692" alt="DIN" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/dafb6b0e-bd64-4c05-9ce6-3fa16d8f32cc">


DIEN:

<img width="689" alt="DIEN" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/55e7c6b9-e8f2-4594-9195-ddff4c42c73d">


SASRec:

<img width="588" alt="SASRec" src="https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/4ab1aad5-77b4-40b0-835e-a1a118f18b3c">

BSTransformer:

![BST](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/0e14913b-e770-4ce0-b87b-1a6b62dcaad9)


MIND：

![MIND](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/3089993c-d291-47b8-95af-e61e81e6d86e)


Comirec：

![Comirec](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/08884e82-b3c2-4c7a-b0c8-949c6c9150ff)




ESSM：

![ESSM](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/a7217768-d1f1-43af-9db0-2ed3768c199c)

MMOE：

![MMOE](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/b32c61dd-cef2-4fc1-913f-802580d3741c)

CGC：

![CGC](https://github.com/BinFuPKU/CTRRecommenderModels/assets/29002864/1aade055-a884-4723-b868-ff87d557f1f5)


The project is ongoing ......
