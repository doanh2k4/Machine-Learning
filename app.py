# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor  # Mô hình Neural Network

# Thu thập và xử lý dữ liệu
# Đọc dữ liệu từ file CSV vào dataframe của pandas
car_dataset = pd.read_csv('car data.csv')

# Kiểm tra 5 dòng đầu tiên của dữ liệu
print(car_dataset.head())

# Kiểm tra số hàng và cột
print(car_dataset.shape)

# Thông tin về dữ liệu
print(car_dataset.info())

# Kiểm tra giá trị bị thiếu
print(car_dataset.isnull().sum())

# Kiểm tra phân phối của các dữ liệu dạng danh mục (categorical)
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())

# Mã hóa dữ liệu dạng danh mục
# Mã hóa cột "Fuel_Type"
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)

# Mã hóa cột "Seller_Type"
car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)

# Mã hóa cột "Transmission"
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

print(car_dataset.head())

# Tách dữ liệu và nhãn
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

# Tách tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# Chuẩn hóa dữ liệu (chỉ cần cho mô hình mạng nơron)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### 1. Mô hình Linear Regression (Hồi quy tuyến tính)

# Tạo mô hình Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# Dự đoán trên tập huấn luyện và kiểm tra
train_predictions_lin = lin_reg_model.predict(X_train)
test_predictions_lin = lin_reg_model.predict(X_test)

# Tính toán các chỉ số đánh giá cho Linear Regression
train_r2_score_lin = r2_score(Y_train, train_predictions_lin)
test_r2_score_lin = r2_score(Y_test, test_predictions_lin)
train_mse_lin = mean_squared_error(Y_train, train_predictions_lin)
test_mse_lin = mean_squared_error(Y_test, test_predictions_lin)
train_rmse_lin = np.sqrt(train_mse_lin)
test_rmse_lin = np.sqrt(test_mse_lin)

print(f"Linear Regression - R² trên tập huấn luyện: {train_r2_score_lin}, R² trên tập kiểm tra: {test_r2_score_lin}")
print(f"Linear Regression - MSE trên tập huấn luyện: {train_mse_lin}, MSE trên tập kiểm tra: {test_mse_lin}")
print(f"Linear Regression - RMSE trên tập huấn luyện: {train_rmse_lin}, RMSE trên tập kiểm tra: {test_rmse_lin}")

# Vẽ biểu đồ giá thực tế và dự đoán trên tập kiểm tra
plt.scatter(Y_test, test_predictions_lin)
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Linear Regression: Giá thực tế vs Giá dự đoán (Tập kiểm tra)")
plt.show()

### 2. Mô hình Lasso Regression

# Tạo mô hình Lasso Regression
lass_reg_model = Lasso()
lass_reg_model.fit(X_train, Y_train)

# Dự đoán trên tập huấn luyện và kiểm tra
train_predictions_lasso = lass_reg_model.predict(X_train)
test_predictions_lasso = lass_reg_model.predict(X_test)

# Tính toán các chỉ số đánh giá cho Lasso Regression
train_r2_score_lasso = r2_score(Y_train, train_predictions_lasso)
test_r2_score_lasso = r2_score(Y_test, test_predictions_lasso)
train_mse_lasso = mean_squared_error(Y_train, train_predictions_lasso)
test_mse_lasso = mean_squared_error(Y_test, test_predictions_lasso)
train_rmse_lasso = np.sqrt(train_mse_lasso)
test_rmse_lasso = np.sqrt(test_mse_lasso)

print(f"Lasso Regression - R² trên tập huấn luyện: {train_r2_score_lasso}, R² trên tập kiểm tra: {test_r2_score_lasso}")
print(f"Lasso Regression - MSE trên tập huấn luyện: {train_mse_lasso}, MSE trên tập kiểm tra: {test_mse_lasso}")
print(f"Lasso Regression - RMSE trên tập huấn luyện: {train_rmse_lasso}, RMSE trên tập kiểm tra: {test_rmse_lasso}")

# Vẽ biểu đồ giá thực tế và dự đoán trên tập kiểm tra
plt.scatter(Y_test, test_predictions_lasso)
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Lasso Regression: Giá thực tế vs Giá dự đoán (Tập kiểm tra)")
plt.show()

### 3. Mô hình Neural Network (Mạng Nơron)

# Xây dựng mô hình mạng nơron
nn_model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=2)
nn_model.fit(X_train_scaled, Y_train)

# Dự đoán trên tập huấn luyện và kiểm tra
train_predictions_nn = nn_model.predict(X_train_scaled)
test_predictions_nn = nn_model.predict(X_test_scaled)

# Tính toán các chỉ số đánh giá cho Neural Network
train_r2_score_nn = r2_score(Y_train, train_predictions_nn)
test_r2_score_nn = r2_score(Y_test, test_predictions_nn)
train_mse_nn = mean_squared_error(Y_train, train_predictions_nn)
test_mse_nn = mean_squared_error(Y_test, test_predictions_nn)
train_rmse_nn = np.sqrt(train_mse_nn)
test_rmse_nn = np.sqrt(test_mse_nn)

print(f"Neural Network - R² trên tập huấn luyện: {train_r2_score_nn}, R² trên tập kiểm tra: {test_r2_score_nn}")
print(f"Neural Network - MSE trên tập huấn luyện: {train_mse_nn}, MSE trên tập kiểm tra: {test_mse_nn}")
print(f"Neural Network - RMSE trên tập huấn luyện: {train_rmse_nn}, RMSE trên tập kiểm tra: {test_rmse_nn}")

# Vẽ biểu đồ giá thực tế và dự đoán trên tập kiểm tra
plt.scatter(Y_test, test_predictions_nn)
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Neural Network: Giá thực tế vs Giá dự đoán (Tập kiểm tra)")
plt.show()

### 4. Stacking Regressor

# Tạo mô hình stacking
estimators = [
    ('linear', LinearRegression()),
    ('lasso', Lasso()),
    ('nn', MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=2))
]

stacked_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacked_model.fit(X_train, Y_train)

# Dự đoán trên tập huấn luyện và kiểm tra
train_predictions_stacked = stacked_model.predict(X_train)
test_predictions_stacked = stacked_model.predict(X_test)

# Tính toán các chỉ số đánh giá cho Stacking Regressor
train_r2_score_stacked = r2_score(Y_train, train_predictions_stacked)
test_r2_score_stacked = r2_score(Y_test, test_predictions_stacked)
train_mse_stacked = mean_squared_error(Y_train, train_predictions_stacked)
test_mse_stacked = mean_squared_error(Y_test, test_predictions_stacked)
train_rmse_stacked = np.sqrt(train_mse_stacked)
test_rmse_stacked = np.sqrt(test_mse_stacked)

print(f"Stacking Regressor - R² trên tập huấn luyện: {train_r2_score_stacked}, R² trên tập kiểm tra: {test_r2_score_stacked}")
print(f"Stacking Regressor - MSE trên tập huấn luyện: {train_mse_stacked}, MSE trên tập kiểm tra: {test_mse_stacked}")
print(f"Stacking Regressor - RMSE trên tập huấn luyện: {train_rmse_stacked}, RMSE trên tập kiểm tra: {test_rmse_stacked}")

# Vẽ biểu đồ giá thực tế và dự đoán trên tập kiểm tra
plt.scatter(Y_test, test_predictions_stacked)
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Stacking Regressor: Giá thực tế vs Giá dự đoán (Tập kiểm tra)")
plt.show()
