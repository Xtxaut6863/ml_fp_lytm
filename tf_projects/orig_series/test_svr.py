import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge,LinearRegression,ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
import sys
sys.path.append(
    'F:/ml_fp_lytm/tf_projects/test/')
from import_tanmiao import load_normalized_data


# laod data from local disk.
(x_train, y_train), (x_dev, y_dev), (x_dev, y_dev), (
        series_mean, series_max,
        series_min) = load_normalized_data("orig_day_full_X.xlsx")

# Get the SVR model with RBF kernal
# svr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1)

# Hyperparameter
n_folds = 6  # Cross validation folds
model_br = BayesianRidge()
model_lr = LinearRegression()
model_etc = ElasticNet()
model_svr = SVR()
model_gbr = GradientBoostingRegressor()

model_names = [
    'BayesianRidge',
    'LinearRegression',
    'ElasticNet',
    'SVR',
    'GBR'
]

model_dic = [
    model_br,
    model_lr,
    model_etc,
    model_svr,
    model_gbr
]

cv_score_list = []

pred_y_list = []

for model in model_dic:
    scores = cross_val_score(model,x_train,y_train,cv=n_folds)
    cv_score_list.append(scores)
    pred_y_list.append(model.fit(x_train,y_train).predict(x_train))

# Get the feature num
n_samples, n_features = x_train.shape

# model metrics for model evaluate
model_metrics_name = [
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
]

model_metrics_list = []

for i in range(5):
    temp_list = []
    for m in  model_metrics_name:
        temp_score = m(y_train,pred_y_list[i])
        temp_list.append(temp_score)
    model_metrics_list.append(temp_list)

df1 = pd.DataFrame(cv_score_list,index=model_names)
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])



print('samples: %d \t features: %d'%(n_samples,n_features))
print(70*'-')
print('cross validation result:')
print(df1)
print(70*'-')
print('regression metrics:')
print(df2)
print(70*'-')
print('short name \t full name')
print('ev \t explained_variance')
print('mae \t mean_absolute_error')
print('mse \t mean_square_error')
print('r2 \t r2')
print(70*'-')

# 模型效果可视化
plt.figure()  # 创建画布
plt.plot(np.arange(x_train.shape[0]), y_train, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(pred_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(
        np.arange(x_train.shape[0]),
        pred_y_list[i],
        color_list[i],
        label=model_names[i])  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像



# y_rbf = svr_rbf.fit(x_train,y_train).predict(x_train)
# t = np.linspace(1,y_train.size,y_train.size)
# plt.plot(t,y_train,color='darkorange',label='records')
# plt.plot(t,y_rbf,color='navy',label='predictions')
# plt.xlabel('Time(day)')
# plt.ylabel('flow(' + r'$m^3$' + '/s)')
# plt.title('Support Vector Regression')
# plt.legend()
# plt.show()