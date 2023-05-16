# Python基于Bert实现的情感分析模型
基于Bert实现情感分析的代码示例。使用预训练的BERT模型，并提供了一个简单的接口，可以对给定的中文文本进行情感分类，判断是积极的评论还是消极的评论。
# 准备工作
1. 克隆项目到本地
2. 导入必要的库和模块：
```python
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
```
