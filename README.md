```
@author：Yan Gu; Jicong Duan; Hualong Yu; Xibei Yang; Shang Gao
#Title: PLVI-CE: A multi-label active learning algorithm with simultaneously considering uncertainty and diversity
#Time:2022/9/24
#Institution：Jiangsu University of Science and Technology
```
# PLVI-CE
In this study, we consider both uncertainty and diversity measures, and combine them to construct a query strategy called predicted label vectors inconsistency and cross entropy measure (PLVI-CE). In PLVI-CE, the uncertainty is measured by the inconsistency between two predicted label vectors from the same unlabeled instance, and the diversity is measured by the average discrepancy in posterior probabilities between each unlabeled instance and all instances in the labeled set
# Requirement
1. Python 3.8
2. 12 datasets
# Results
Experimental results on 20 benchmark multi-label datasets indicate the effectiveness and superiority of the proposed PLVI-CE algorithm in comparison with several current state-of-the-art MLAL algorithms.
# Reference
1. Ren P, Xiao Y, Chang X, et al (2022) A survey of Deep Active Learning. ACM Computing Surveys 54(9):1–40
2. Yu H, Yang X, Zheng S, Sun C (2018) Active learning from imbalanced data: A solution of online weighted Extreme Learning Machine. IEEE Transactions on Neural Networks and Learning Systems 30(4):1088–1103
3. Gui X, Lu X, Yu G (2021) Cost-effective batch-mode multi-label active learning. Neurocomputing 463:355–367
4. Chakraborty S, Balasubramanian V, Panchanathan S (2014) Adaptive Batch Mode Active Learning. IEEE Transactions on Neural Networks and Learning Systems 26(8):1747–1760
5. Yu H, Sun C, Yang W, et al (2015) Al-Elm: One uncertainty-based active learning algorithm using Extreme Learning Machine. Neurocomputing 166:140–150



