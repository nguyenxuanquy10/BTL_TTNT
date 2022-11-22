# -*- coding: utf-8 -*-


## Hình ảnh hóa cho hồi quy tuyến tính bội
#pandas được sư dụng để xử lí dữ liệu 
import pandas
from sklearn import linear_model
# numpy để thực hiện các xử lí toán học trong mảng 
import numpy as np
## Thư viện dùng để trực quan hóa dữ liệu 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

df = pandas.read_csv("/content/BTL_TTNT/apartment - Copy.csv")


## chuẩn bị data set 
# Xóa trường dự liệu không dùng tới 
df=df.drop(columns="ID")
# Lấy 2  trường dữ liệu kích thước và vị trí tầng 
X = df[['Size', 'Floor']]
# Lấy trường dữ liệu kết quả 
Y = df['SalePrice']


# Tạo model 
model = smf.ols(formula='SalePrice ~ Size + Floor', data=df)
results_formula = model.fit()
results_formula.params

variableDependent = np.array([results_formula.params[1],results_formula.params[2]])
variableIndependent = np.array([600, 3])


print(np.sum(variableDependent*variableIndependent))
print(results_formula.params)

## Chuẩn bị dữ liệu để trực quan hóa 

x_surf, y_surf = np.meshgrid(np.linspace(df.Size.min(), df.Size.max(), 100),np.linspace(df.Floor.min(), df.Floor.max(), 100))
onlyX = pd.DataFrame({'Size': x_surf.ravel(), 'Floor': y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)


## Chuyển kết quả dự đoán thành 1 mảng 

fittedY=np.array(fittedY)


# Trực quan hóa dữ liệu cho hồi quy tuyến tính bội 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Size'],df['Floor'],df['SalePrice'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('Size')
ax.set_ylabel('Floor')
ax.set_zlabel('SalePrice')
plt.show()


