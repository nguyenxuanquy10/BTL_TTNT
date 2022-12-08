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

df = pandas.read_csv("/content/BTL_TTNT/Daegu_Real_Estate_data.csv")


## chuẩn bị data set 

# Lấy 2  trường dữ liệu kích thước và vị trí tầng 
X = df[['Size_sqf', 'Floor','N_Parkinglot_Ground','N_elevators','N_FacilitiesNearBy_PublicOffice','N_FacilitiesNearBy_Hospital','N_FacilitiesNearBy_Dpartmentstore','N_FacilitiesNearBy_Mall','N_FacilitiesNearBy_Park','N_SchoolNearBy_University']]
# Lấy trường dữ liệu kết quả 
Y = df['SalePrice']


# Tạo model 
model = smf.ols(formula='SalePrice ~ Size_sqf + Floor + N_Parkinglot_Ground + N_elevators + N_FacilitiesNearBy_PublicOffice + N_FacilitiesNearBy_Hospital + N_FacilitiesNearBy_Dpartmentstore + N_FacilitiesNearBy_Mall + N_FacilitiesNearBy_Park + N_SchoolNearBy_University', data=df)
results_formula = model.fit()
results_formula.params

variableDependent = np.array([results_formula.params[1],results_formula.params[2],
                              results_formula.params[3],results_formula.params[4],
                              results_formula.params[5],results_formula.params[6],
                              results_formula.params[7],results_formula.params[8],
                              results_formula.params[9],results_formula.params[10]])
# variableIndependent = np.array([587,8,80,2,5,1,2,1,1,0])
variableIndependent = np.array([814,3,111,0,2,1,1,1,0,2])
# variableIndependent = np.array([814,3,111,0,2,1,1,1,0,2])
print(np.sum(variableDependent*variableIndependent))
print(results_formula.params)

## Chuẩn bị dữ liệu để trực quan hóa 

# x_surf, y_surf = np.meshgrid(np.linspace(df.Size.min(), df.Size.max(), 100),np.linspace(df.Floor.min(), df.Floor.max(), 100))
# onlyX = pd.DataFrame({'Size': x_surf.ravel(), 'Floor': y_surf.ravel()})
# fittedY=results_formula.predict(exog=onlyX)


# ## Chuyển kết quả dự đoán thành 1 mảng 

# fittedY=np.array(fittedY)


# # Trực quan hóa dữ liệu cho hồi quy tuyến tính bội 

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df['Size'],df['Floor'],df['SalePrice'],c='red', marker='o', alpha=0.5)
# ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
# ax.set_xlabel('Size')
# ax.set_ylabel('Floor')
# ax.set_zlabel('SalePrice')
# plt.show()


