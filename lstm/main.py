# -*- coding:utf-8 -*-
import sin_train as st

if __name__ == "__main__":
    train_model = st.sin_train()
    test_x = train_model.test_x
    test_y = train_model.test_y
