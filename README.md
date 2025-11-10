# 新冠感染人数预测系统

## 项目描述
基于深度学习的新冠感染人数预测模型，使用PyTorch框架实现。

## 功能特点
- 特征选择与相关性分析
- 神经网络回归模型
- 正则化防止过拟合
- 数据标准化处理

## 使用方法
1. 安装依赖：`pip install -r requirements.txt`
2. 运行训练：`python regression_predict.py`
3. 查看结果：预测结果保存在`pred_submit.csv`

## 文件结构
- `regression_predict.py` - 主程序文件
- `covid.train.csv` - 训练数据
- `covid.test.csv` - 测试数据
- `requirements.txt` - 依赖包
