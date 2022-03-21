([中文文档](https://paddlerec.readthedocs.io/en/latest/)|[简体中文](./README_CN.md)|English)
<p align="center">
<img align="center" src="doc/imgs/logo.png">
<p>
<p align="center">
<img align="center" src="doc/imgs/overview_en.png">
<p>

<h2 align="center">News<img src="./doc/imgs/rec_new_icon.png" width="40"/></h2>

* [2022/3/21] Add a new [paper](./paper) directory , show our analysis of the top meeting papers of the recommendation system in 2021 years and the list of recommendation system papers in the industry for your reference.  
* [2022/3/10] Add 5 algorithms: [DCN_V2](models/rank/dcn_v2), [MHCN](models/recall/mhcn), [FLEN](models/rank/flen), [Dselect_K](models/multitask/dselect_k)，[AutoFIS](models/rank/autofis)。  
* [2022/1/12] Add AI Studio [Online running](https://aistudio.baidu.com/aistudio/projectdetail/3240640) function, you can easily and quickly online experience our model on AI studio platform.

<h2 align="center">What is recommendation system ?</h2>
<p align="center">
<img align="center" src="doc/imgs/rec-overview-en.png">
<p>

- Recommendation system helps users quickly find useful and interesting information from massive data.

- Recommendation system is also a silver bullet to attract users, retain users, increase users' stickness or conversionn.

  > Who can better use the recommendation system, who can gain more advantage in the fierce competition.
  >
  > At the same time, there are many problems in the process of using the recommendation system, such as: huge data, complex model, inefficient distributed training, and so on.

<h2 align="center">What is PaddleRec ?</h2>


- A quick start tool of search & recommendation algorithm based on [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/index_en.html)
- A complete solution of recommendation system for beginners, developers and researchers.
- Recommendation algorithm library including content-understanding, match, recall, rank, multi-task, re-rank etc.[Support model list](#Support_Model_List)

<h2 align="center">Getting Started</h2>

### Online running

- **[AI Studio Online Running](https://aistudio.baidu.com/aistudio/projectdetail/3240640)**

### Environmental requirements
* Python 2.7/ 3.5 / 3.6 / 3.7 , Python 3.7 is recommended ,Python in example represents Python 3.7 by default
* PaddlePaddle >=2.0 
* operating system: Windows/Mac/Linux

  > Linux is recommended for distributed training
  
### Installation

- Install by pip in GPU environment
  ```bash
  python -m pip install paddlepaddle-gpu==2.0.0 
  ```
- Install by pip in CPU environment
  ```bash
  python -m pip install paddlepaddle # gcc8 
  ```
For download more versions, please refer to the installation tutorial [Installation Manuals](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)

### Download PaddleRec

```bash
git clone https://github.com/PaddlePaddle/PaddleRec/
cd PaddleRec
```

### Quick Start

We take the `dnn` algorithm as an example to get start of `PaddleRec`, and we take 100 pieces of training data from [Criteo Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge/):

```bash
python -u tools/trainer.py -m models/rank/dnn/config.yaml # Training with dygraph model
python -u tools/static_trainer.py -m models/rank/dnn/config.yaml #  Training with static model
```


<h2 align="center">Documentation</h2>

### Background
* [Recommendation System](doc/rec_background.md)
* [Distributed deep Learning](doc/ps_background.md)

### Introductory Tutorial
* [PaddleRec Function Introduction](doc/introduction.md)
* [Dygraph Train](doc/dygraph_mode.md)
* [Static Train](doc/static_mode.md)
* [Distributed Train](doc/fleet_mode.md)


### Advanced Tutorial
* [Submit Specification](doc/contribute.md)
* [Custom Reader](doc/custom_reader.md)
* [Custom Model](doc/model_develop.md)
* [Configuration Description of Yaml](doc/yaml.md)
* [Training Visualization](doc/visualization.md)
* [Serving](doc/serving.md)
* [Python Inference](doc/inference.md)
* [Benchmark](doc/benchmark.md)
* [The latest reserch trends of RS](paper/readme.md)

### FAQ
* [Common Problem FAQ](doc/faq.md)

### Acknowledgements
* [Contributions From External Developer](contributor.md)

#### Support_Model_List
<h2 align="center">Support Model List</h2>


  |         Type          |                                 Algorithm                                 | Online Environment | Parameter-Server | Multi-GPU | version | Paper                                                                                                                                                                                                       |
  | :-------------------: | :-----------------------------------------------------------------------: | :---: | :--------------: | :-------: | :-------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | Content-Understanding | [TextCnn](models/contentunderstanding/textcnn/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/contentunderstanding/textcnn.html)) |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3238415)  |       ✓         |     x     |      >=2.1.0     | [EMNLP 2014][Convolutional neural networks for sentence classication](https://www.aclweb.org/anthology/D14-1181.pdf)                                                                                        |
  | Content-Understanding |         [TagSpace](models/contentunderstanding/tagspace/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/contentunderstanding/tagspace.html))         |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3238891)  |         ✓         |     x     |      >=2.1.0     | [EMNLP 2014][TagSpace: Semantic Embeddings from Hashtags](https://www.aclweb.org/anthology/D14-1194.pdf)                                                                                                    |
  |         Match         |                    [DSSM](models/match/dssm/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/match/dssm.html))                     |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3217658?contributionType=1)  |         ✓         |     x     |      >=2.1.0     | [CIKM 2013][Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)             |
  |         Match         |        [MultiView-Simnet](models/match/multiview-simnet/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/match/multiview-simnet.html))         |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3238206)  |         ✓         |     x     |      >=2.1.0     | [WWW 2015][A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf)             |
  |         Match         |        [Match-Pyramid](models/match/match-pyramid/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/match/match-pyramid.html))         |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3238192)  |         ✓         |     x     |      >=2.1.0     | [2016][Text Matching as Image Recognition](https://arxiv.org/pdf/1602.06359.pdf)             |
  |        Recall         |                   [TDM](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/treebased/tdm/)                    |  -  |        ✓         |  >=1.8.0  | [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [KDD 2018][Learning Tree-based Deep Model for Recommender Systems](https://arxiv.org/pdf/1801.02294.pdf)                                                                                                    |
  |        Recall         |                [FastText](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/recall/fasttext/)                |  -  |         x         |     x     |[1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [EACL 2017][Bag of Tricks for Efficient Text Classification](https://www.aclweb.org/anthology/E17-2068.pdf)                                                                                                 |
  |        Recall         |                [MIND](models/recall/mind/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/recall/mind.html))                |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3239088)  |     x     |     x     | >=2.1.0 | [2019][Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/pdf/1904.08030.pdf)                                                                                                 |
  |        Recall         |                [Word2Vec](models/recall/word2vec/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/recall/word2vec.html))                |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240153)  |         ✓         |     x     |      >=2.1.0     | [NIPS 2013][Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) |
  |        Recall         |                [DeepWalk](models/recall/deepwalk/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/recall/deepwalk.html))                |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3239086)  |     x     |     x     | >=2.1.0 | [SIGKDD 2014][DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf) |
  |        Recall         |                     [SSR](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/recall/ssr/)                     |  -  |         ✓         |     ✓     | [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [SIGIR 2016][Multi-Rate Deep Learning for Temporal Recommendation](http://sonyis.me/paperpdf/spr209-song_sigir16.pdf)                                                                                       |
  |        Recall         |                 [Gru4Rec](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/recall/gru4rec/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/recall/gru4rec.html))                |  -  |         ✓         |     ✓     | [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [2015][Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)                                                                                                      |
  |        Recall         |             [Youtube_dnn](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/recall/youtube_dnn/)             |  -  |         ✓         |     ✓     | [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [RecSys 2016][Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)                                               |
  |        Recall         |                     [NCF](models/recall/ncf/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/recall/ncf.html))                     |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240152)  |         ✓         |     ✓     | >=2.1.0 | [WWW 2017][Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)                                                                                                                            |
  |   Recall   |                     [TiSAS](models/recall/tisas/)            |   -   |    ✓    |     ✓     | >=2.1.0 | [WSDM 2020][Time Interval Aware Self-Attention for Sequential Recommendation](https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf)                                                                                               |
  |   Recall   |                     [ENSFM](models/recall/ensfm/)                     |  -  |     ✓     |     ✓     | >=2.1.0 | [IW3C2 2020][Eicient Non-Sampling Factorization Machines for Optimal Context-Aware Recommendation](http://www.thuir.cn/group/~mzhang/publications/TheWebConf2020-Chenchong.pdf)                                                               |
  |   Recall   |                     [MHCN](models/recall/mhcn/)                     |  -  |     ✓     |     ✓     | >=2.1.0 | [WWW 2021][Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation](https://arxiv.org/pdf/2101.06448v3.pdf)                                                               |
  |        Recall         |                     [GNN](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/recall/gnn/)                     |  -  |         ✓         |     ✓     | [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [AAAI 2019][Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855)                                                                                                      |
  |        Recall         |                     [RALM](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/recall/look-alike_recall/)                     |  -  |         ✓         |     ✓     | [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [KDD 2019][Real-time Attention Based Look-alike Model for Recommender System](https://arxiv.org/pdf/1906.05022.pdf)                                                                                                      |
  |         Rank          |      [Logistic Regression](models/rank/logistic_regression/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/logistic_regression.html))      |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240481)  |         ✓         |     x     |      >=2.1.0     | /                                                                                                                                                                                                           |
  |         Rank          |                      [Dnn](models/rank/dnn/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/dnn.html))                      |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240347)  |         ✓         |     ✓     |      >=2.1.0     | /                                                                                                                                                                                                           |
  |         Rank          |                       [FM](models/rank/fm/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/fm.html))                       |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240371)  |         ✓         |     x     |      >=2.1.0     | [IEEE Data Mining 2010][Factorization machines](https://analyticsconsultores.com.mx/wp-content/uploads/2019/03/Factorization-Machines-Steffen-Rendle-Osaka-University-2010.pdf)                             |
  |         Rank          |                       [BERT4REC](models/rank/bert4rec/)                       |  -  |         ✓         |     x     |      >=2.1.0     | [CIKM 2019][BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/pdf/1904.06690.pdf)                             |
  |         Rank          |                       [FAT_DeepFFM](models/rank/fat_deepffm/)                       |  -  |         ✓         |     x     |      >=2.1.0     | [2019][FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine](https://arxiv.org/pdf/1905.06336.pdf)                             |
  |         Rank          |                      [FFM](models/rank/ffm/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/ffm.html))                      |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240369)  |         ✓         |     x     | >=2.1.0 | [RECSYS 2016][Field-aware Factorization Machines for CTR Prediction](https://dl.acm.org/doi/pdf/10.1145/2959100.2959134)                                                                                    |
  |         Rank          |                      [FNN](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rank/fnn/)                      |  -  |         ✓         |     x     | [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [ECIR 2016][Deep Learning over Multi-field Categorical Data](https://arxiv.org/pdf/1601.02376.pdf)                                                                                                          |
  |         Rank          |            [Deep Crossing](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rank/deep_crossing/)            |  -  |         ✓         |     x     | [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [ACM 2016][Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)                                                   |
  |         Rank          |                      [Pnn](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rank/pnn/)                      |  -  |         ✓         |     x     | [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [ICDM 2016][Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                                                               |
  |         Rank          |                      [DCN](models/rank/dcn/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/dcn.html))                      |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240207)  |         ✓         |     x     | >=2.1.0 | [KDD 2017][Deep & Cross Network for Ad Click Predictions](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754)                                                                                               |
  |         Rank          |                      [NFM](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rank/nfm/)                      |  -  |         ✓         |     x     | [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/doi/pdf/10.1145/3077136.3080777)                                                                             |
  |         Rank          |                      [AFM](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rank/afm/)                      |  -  |         ✓         |     x     | [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)                                                  |
  |         Rank          |                   [DMR](models/rank/dmr/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/dmr.html))                   |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240346)  |     x     |     x     | >=2.1.0 | [AAAI 2020][Deep Match to Rank Model for Personalized Click-Through Rate Prediction](https://github.com/lvze92/DMR/blob/master/%5BDMR%5D%20Deep%20Match%20to%20Rank%20Model%20for%20Personalized%20Click-Through%20Rate%20Prediction-AAAI20.pdf)                                                                                 |
  |         Rank          |                   [DeepFM](models/rank/deepfm/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/deepfm.html))                   |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3238581)  |         ✓         |     x     |      >=2.1.0     | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)                                                                                 |
  |         Rank          |                  [xDeepFM](models/rank/xdeepfm/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/xdeepfm.html))                  |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240533)  |         ✓         |     x     | >=2.1.0 | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/3219819.3220023)                                                       |
  |         Rank          |                      [DIN](models/rank/din/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/din.html))                      |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240307)  |         ✓         |     x     | >=2.1.0 | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://dl.acm.org/doi/pdf/10.1145/3219819.3219823)                                                                                     |
  |         Rank          |                     [DIEN](models/rank/dien/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/dien.html))                     |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240212)  |         ✓         |     x     | >=2.1.0 | [AAAI 2019][Deep Interest Evolution Network for Click-Through Rate Prediction](https://www.aaai.org/ojs/index.php/AAAI/article/view/4545/4423)                                                              |
  |         Rank          |                     [GateNet](models/rank/gatenet/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/gatenet.html))                      |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240375)  |         ✓         |     x     | >=2.1.0 | [SIGIR 2020][GateNet: Gating-Enhanced Deep Network for Click-Through Rate Prediction](https://arxiv.org/pdf/2007.03519.pdf)                                                              |
  |         Rank          |                     [DLRM](models/rank/dlrm/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/dlrm.html))                     |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240343)  |         ✓         |     x     | >=2.1.0 | [CoRR 2019][Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091)                                                              |
  |         Rank          |                     [NAML](models/rank/naml/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/naml.html))                     |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240375)  |         ✓         |     x     | >=2.1.0 | [IJCAI 2019][Neural News Recommendation with Attentive Multi-View Learning](https://www.ijcai.org/proceedings/2019/0536.pdf)                                                              |
  |         Rank          |                     [DIFM](models/rank/difm/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/difm.html))                     |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240286)  |         ✓         |     x     | >=2.1.0 | [IJCAI 2020][A Dual Input-aware Factorization Machine for CTR Prediction](https://www.ijcai.org/proceedings/2020/0434.pdf)                                                              |
  |         Rank          |                     [DeepFEFM](models/rank/deepfefm/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/deepfefm.html))                     |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240209)  |         ✓         |     x     | >=2.1.0 | [arXiv 2020][Field-Embedded Factorization Machines for Click-through rate prediction](https://arxiv.org/abs/2009.09931)                                                              |
  |         Rank          |                      [BST](models/rank/bst/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/bst.html))                      |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3240205)  |         ✓         |     x     |  >=2.1.0 | [DLP-KDD 2019][Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1905.06874v1.pdf)                                                                              |
  |         Rank          |                  [AutoInt](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rank/AutoInt/)                  |  -  |         ✓         |     x     | [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [CIKM 2019][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)                                                                       |
  |         Rank          |                [Wide&Deep](models/rank/wide_deep/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/rank/wide_deep.html))                |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3238421)  |         ✓         |     x     |      >=2.1.0     | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454)                                                                                               |
  |         Rank          |                    [FGCNN](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rank/fgcnn/)                    |  -  |         ✓         |     ✓     |  [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [WWW 2019][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1904.04447.pdf)                                                                      |
  |         Rank          |                  [Fibinet](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rank/fibinet/)                  |  -  |         ✓         |     ✓     |  [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [RecSys19][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction]( https://arxiv.org/pdf/1905.09433.pdf)                                                 |
  |         Rank          |                     [FLEN](models/rank/flen/)                     |  -  |         ✓         |     ✓     |  >=2.1.0 | [2019][FLEN: Leveraging Field for Scalable CTR Prediction]( https://arxiv.org/pdf/1911.04690.pdf)                                                                                                           |
  |   Rank   |                     [DeepRec](models/rank/deeprec/)                     |  -  |       ✓     |     ✓     | >=2.1.0 | [2017][Training Deep AutoEncoders for Collaborative Filtering](https://arxiv.org/pdf/1708.01715v3.pdf)                                                                                                          |
  |   Rank   |                     [AutoFIS](models/rank/autofis/)                     |  -  |       ✓     |     ✓     | >=2.1.0 | [KDD 2020][AutoFIS: Automatic Feature Interaction Selection in Factorization Models for Click-Through Rate Prediction](https://arxiv.org/pdf/2003.11235v3.pdf)                                                                                                          |
  |   Rank   |                     [DCN_V2](models/rank/dcn_v2/)                     |  -  |       ✓     |     ✓     | >=2.1.0 | [WWW 2021][DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/pdf/2008.13535v2.pdf) 
  |      Multi-Task       |                  [PLE](models/multitask/ple/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/multitask/ple.html))                   |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3238938)  |     ✓     |     ✓     |  >=2.1.0 | [RecSys 2020][Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/abs/10.1145/3383313.3412236)                                                              |
  |      Multi-Task       |                  [ESMM](models/multitask/esmm/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/multitask/esmm.html))                   |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3238583)  |         ✓         |     ✓     |      >=2.1.0     | [SIGIR 2018][Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931)                                                              |
  |      Multi-Task       |                  [MMOE](models/multitask/mmoe/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/multitask/mmoe.html))                   |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3238934)  |         ✓         |     ✓     |      >=2.1.0     | [KDD 2018][Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007)                                                       |
  |      Multi-Task       |           [ShareBottom](models/multitask/share_bottom/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/multitask/share_bottom.html))           |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3238943)  |         ✓         |     ✓     |  >=2.1.0 | [1998][Multitask learning](http://reports-archive.adm.cs.cmu.edu/anon/1997/CMU-CS-97-203.pdf)                                                                                                               |
  |      Multi-Task       |           [Maml](models/multitask/maml/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/multitask/maml.html))           |  [Python CPU/GPU](https://aistudio.baidu.com/aistudio/projectdetail/3238412)  |    x      |     x     | >=2.1.0 | [PMLR 2017][Model-agnostic meta-learning for fast adaptation of deep networks](https://arxiv.org/pdf/1703.03400.pdf)                                                                                                               |
  |  Multi-Task  |           [DSelect_K](models/multitask/dselect_k/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/multitask/dselect_k.html))           |  -  |      x      |     x     | >=2.1.0 | [NeurIPS 2021][DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning](https://arxiv.org/pdf/2106.03760v3.pdf)                                                                                                               |
  |        Re-Rank        |                [Listwise](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rerank/listwise/)                |  -  |         ✓         |     x     |  [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [2019][Sequential Evaluation and Generation Framework for Combinatorial Recommender System](https://arxiv.org/pdf/1902.00245.pdf)                                                                           |

<h2 align="center">Community</h2>

<p align="center">
    <br>
    <img alt="Release" src="https://img.shields.io/badge/Release-0.1.0-yellowgreen">
    <img alt="License" src="https://img.shields.io/github/license/PaddlePaddle/PaddleRec">
    <img alt="Slack" src="https://img.shields.io/badge/Join-Slack-green">
    <br>
<p>

### Version history
- 2021.11.19 - PaddleRec v2.2.0
- 2021.05.19 - PaddleRec v2.1.0
- 2021.01.29 - PaddleRec v2.0.0
- 2020.10.12 - PaddleRec v1.8.5
- 2020.06.17 - PaddleRec v0.1.0
- 2020.06.03 - PaddleRec v0.0.2
- 2020.05.14 - PaddleRec v0.0.1
  
### License
[Apache 2.0 license](LICENSE)

### Contact us

For any feedback, please propose a [GitHub Issue](https://github.com/PaddlePaddle/PaddleRec/issues)

You can also communicate with us in the following ways：

- QQ group id：`861717190`
- Wechat account：`wxid_0xksppzk5p7f22`
- Remarks `REC` add group automatically

<p align="center"><img width="200" height="200" margin="500" src="./doc/imgs/QQ_group.png"/>&#8194;&#8194;&#8194;&#8194;&#8194<img width="200" height="200"  src="doc/imgs/weixin_supporter.png"/></p>
<p align="center">PaddleRec QQ Group&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;&#8194;PaddleRec Wechat account</p>
