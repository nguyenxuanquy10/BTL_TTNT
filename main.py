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


df = pandas.read_csv("/content/BTL_TTNT/Daegu_Real_Estate_data - Copy.csv")

## chuẩn bị data set 

# Lấy tất cả trường dữ liệu 
X = df[['SalePrice','YearBuilt','YrSold','Size_sqf', 'Floor','N_Parkinglot_Ground','N_elevators','N_FacilitiesNearBy_PublicOffice','N_FacilitiesNearBy_Hospital','N_FacilitiesNearBy_Dpartmentstore','N_FacilitiesNearBy_Mall','N_FacilitiesNearBy_Park','N_SchoolNearBy_University']]

Y = df['SalePrice']
df['NumberSold']=df['YrSold']-df['YearBuilt']


# Tạo model 
# full 
model = smf.ols(formula='SalePrice ~ Size_sqf + Floor + N_Parkinglot_Ground + N_elevators + N_FacilitiesNearBy_PublicOffice + N_FacilitiesNearBy_Hospital + N_FacilitiesNearBy_Dpartmentstore + N_FacilitiesNearBy_Mall + N_FacilitiesNearBy_Park + N_SchoolNearBy_University + NumberSold', data=df)

results_formula = model.fit()
results_formula.params


variableDependent = np.array([results_formula.params[1],results_formula.params[2],
                              results_formula.params[3],results_formula.params[4],
                              results_formula.params[5], results_formula.params[6],
                              results_formula.params[7], results_formula.params[8],
                              results_formula.params[9], results_formula.params[10],
                              results_formula.params[11]])
## in hằng số đại diện cho độ dốc hồi quy 
# print(results_formula.params)

## giá gốc 
realPrice =51327;

## nhập đầu vào 
variableIndependentPre = np.array([1985,2007,587,8,80,2,5,1,2,1,1,0])
# Dữ liệu sau khi xử lí 
numberSold=variableIndependentPre[1]-variableIndependentPre[0];
variableIndependent=np.array([587,8,80,2,5,1,2,1,1,0,numberSold])
## giá nhà dự đoán 
predictPrice=np.sum(variableDependent*variableIndependent)
## in kết quả 
print("Giá nhà dự đoán: ",predictPrice)
print("Giá thật : ",realPrice )
## độ chính xác so với giá nhà thật 
percentPredict=(predictPrice/realPrice)*100; 
if(percentPredict>100):
    percentPredict=200-percentPredict

print("Tỷ lệ chính xác : ", percentPredict);





# # Chuẩn bị dữ liệu để trực quan hóa 
# dfVisualLize=df[['Size_sqf', 'Floor','NumberSold']]
# x_surf, y_surf = np.meshgrid(np.linspace(dfVisualLize.Size_sqf.min(), dfVisualLize.Size_sqf.max(), 2100),np.linspace(dfVisualLize.NumberSold.min(), dfVisualLize.NumberSold.max(), 30))
# onlyX = pandas.DataFrame({'Size_sqf': x_surf.ravel(), 'NumberSold': y_surf.ravel()})

# fittedY=results_formula.predict(exog=onlyX)

# ## Chuyển kết quả dự đoán thành 1 mảng 

# fittedY=np.array(fittedY)


# # Trực quan hóa dữ liệu cho hồi quy tuyến tính bội 

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(dfVisualLize['Size_sqf'],dfVisualLize['NumberSold'],dfVisualLize['SalePrice'],c='red', marker='o', alpha=0.5)
# ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
# ax.set_xlabel('Size_sqf')
# ax.set_ylabel('NumberSold')
# ax.set_zlabel('SalePrice')
# plt.show()


