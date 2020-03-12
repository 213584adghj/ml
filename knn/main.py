# -*- coding: utf-8 -*-
import roc
import sys
sys.path.append('...')
from conf import knn as kn
import train as tr
if __name__ == "__main__":
    trainer=tr.trainer()
    predict_result=trainer.model.predict(trainer.data.x_test)
    #print(predict_result)

    for value in kn.CONFIG['data']['parameter']['category_set']:
        roc.roc_picture(predict_result, trainer.data.y_test, value)