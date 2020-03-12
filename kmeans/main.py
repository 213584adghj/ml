# -*- coding: utf-8 -*-
import get_data as gt
import train

if __name__ == "__main__":
    data = gt.data()  # 得到数据
    model = train.trainer(data.all_data)  # 训练模型
    result_model = model.model  # 聚类结果
    print(result_model.predict(data.all_data))
