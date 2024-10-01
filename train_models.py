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
car_dataset = pd.read_csv('./CarData.csv')

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

# Tách tập dữ liệu thành tập huấn luyện, tập kiểm tra và tập xác thực
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=2)
X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=2)

# Chuẩn hóa dữ liệu (chỉ cần cho mô hình mạng nơron)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

### 1. Mô hình Linear Regression (Hồi quy tuyến tính)

# Tạo mô hình Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# Dự đoán trên tập huấn luyện, kiểm tra và xác thực
train_predictions_lin = lin_reg_model.predict(X_train)
test_predictions_lin = lin_reg_model.predict(X_test)
val_predictions_lin = lin_reg_model.predict(X_val)

# Tính toán các chỉ số đánh giá cho Linear Regression
train_r2_score_lin = r2_score(Y_train, train_predictions_lin)
test_r2_score_lin = r2_score(Y_test, test_predictions_lin)
val_r2_score_lin = r2_score(Y_val, val_predictions_lin)
train_mse_lin = mean_squared_error(Y_train, train_predictions_lin)
test_mse_lin = mean_squared_error(Y_test, test_predictions_lin)
val_mse_lin = mean_squared_error(Y_val, val_predictions_lin)
train_rmse_lin = np.sqrt(train_mse_lin)
test_rmse_lin = np.sqrt(test_mse_lin)
val_rmse_lin = np.sqrt(val_mse_lin)

print(
    f"Linear Regression - R² trên tập huấn luyện: {train_r2_score_lin:.4f}, R² trên tập kiểm tra: {test_r2_score_lin:.4f}, R² trên tập xác thực: {val_r2_score_lin:.4f}")
print(
    f"Linear Regression - MSE trên tập huấn luyện: {train_mse_lin:.4f}, MSE trên tập kiểm tra: {test_mse_lin:.4f}, MSE trên tập xác thực: {val_mse_lin:.4f}")
print(
    f"Linear Regression - RMSE trên tập huấn luyện: {train_rmse_lin:.4f}, RMSE trên tập kiểm tra: {test_rmse_lin:.4f}, RMSE trên tập xác thực: {val_rmse_lin:.4f}")

# Vẽ biểu đồ giá thực tế và dự đoán trên tập huấn luyện, kiểm tra và xác thực
plt.figure(figsize=(18, 6))

# Biểu đồ cho tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(Y_train, train_predictions_lin, label='Dự đoán', alpha=0.6)
plt.plot(Y_train, Y_train, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Linear Regression: Giá thực tế vs Giá dự đoán (Tập huấn luyện)")
plt.legend()

# Biểu đồ cho tập kiểm tra
plt.subplot(1, 3, 2)
plt.scatter(Y_test, test_predictions_lin, label='Dự đoán', alpha=0.6)
plt.plot(Y_test, Y_test, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Linear Regression: Giá thực tế vs Giá dự đoán (Tập kiểm tra)")
plt.legend()

# Biểu đồ cho tập xác thực
plt.subplot(1, 3, 3)
plt.scatter(Y_val, val_predictions_lin, label='Dự đoán', alpha=0.6)
plt.plot(Y_val, Y_val, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Linear Regression: Giá thực tế vs Giá dự đoán (Tập xác thực)")
plt.legend()

plt.tight_layout()
plt.savefig('graph/linear_regression.png')  # Lưu biểu đồ
plt.close()

### 2. Mô hình Lasso Regression

# Tạo mô hình Lasso Regression với các giá trị alpha khác nhau
alphas = np.logspace(-4, 4, 50)
best_alpha = None
best_r2_score = -np.inf

# Tìm alpha tốt nhất
for alpha in alphas:
    lass_reg_model = Lasso(alpha=alpha)
    lass_reg_model.fit(X_train, Y_train)

    # Dự đoán trên tập kiểm tra
    test_predictions_lasso = lass_reg_model.predict(X_test)

    # Tính toán chỉ số R² trên tập kiểm tra
    test_r2_score_lasso = r2_score(Y_test, test_predictions_lasso)

    # Lưu lại alpha tốt nhất
    if test_r2_score_lasso > best_r2_score:
        best_r2_score = test_r2_score_lasso
        best_alpha = alpha  # Cập nhật best_alpha

# Lấy 4 chữ số sau dấu phẩy cho alpha tốt nhất
best_alpha_rounded = round(best_alpha, 4)

# In ra alpha tốt nhất
print(f"Alpha tốt nhất cho Lasso Regression: {best_alpha_rounded}")

# Tạo mô hình Lasso Regression với alpha tốt nhất
best_lass_reg_model = Lasso(alpha=best_alpha)
best_lass_reg_model.fit(X_train, Y_train)

# Dự đoán trên tập huấn luyện, kiểm tra và xác thực
train_predictions_lasso = best_lass_reg_model.predict(X_train)
test_predictions_lasso = best_lass_reg_model.predict(X_test)
val_predictions_lasso = best_lass_reg_model.predict(X_val)

# Tính toán các chỉ số đánh giá cho Lasso Regression
train_r2_score_lasso = r2_score(Y_train, train_predictions_lasso)
test_r2_score_lasso = r2_score(Y_test, test_predictions_lasso)
val_r2_score_lasso = r2_score(Y_val, val_predictions_lasso)
train_mse_lasso = mean_squared_error(Y_train, train_predictions_lasso)
test_mse_lasso = mean_squared_error(Y_test, test_predictions_lasso)
val_mse_lasso = mean_squared_error(Y_val, val_predictions_lasso)
train_rmse_lasso = np.sqrt(train_mse_lasso)
test_rmse_lasso = np.sqrt(test_mse_lasso)
val_rmse_lasso = np.sqrt(val_mse_lasso)

print(
    f"Lasso Regression - R² trên tập huấn luyện: {train_r2_score_lasso:.4f}, R² trên tập kiểm tra: {test_r2_score_lasso:.4f}, R² trên tập xác thực: {val_r2_score_lasso:.4f}")
print(
    f"Lasso Regression - MSE trên tập huấn luyện: {train_mse_lasso:.4f}, MSE trên tập kiểm tra: {test_mse_lasso:.4f}, MSE trên tập xác thực: {val_mse_lasso:.4f}")
print(
    f"Lasso Regression - RMSE trên tập huấn luyện: {train_rmse_lasso:.4f}, RMSE trên tập kiểm tra: {test_rmse_lasso:.4f}, RMSE trên tập xác thực: {val_rmse_lasso:.4f}")

# Vẽ biểu đồ giá thực tế và dự đoán trên tập huấn luyện, kiểm tra và xác thực
plt.figure(figsize=(18, 6))

# Biểu đồ cho tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(Y_train, train_predictions_lasso, label='Dự đoán', alpha=0.6)
plt.plot(Y_train, Y_train, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Lasso Regression: Giá thực tế vs Giá dự đoán (Tập huấn luyện)")
plt.legend()

# Biểu đồ cho tập kiểm tra
plt.subplot(1, 3, 2)
plt.scatter(Y_test, test_predictions_lasso, label='Dự đoán', alpha=0.6)
plt.plot(Y_test, Y_test, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Lasso Regression: Giá thực tế vs Giá dự đoán (Tập kiểm tra)")
plt.legend()

# Biểu đồ cho tập xác thực
plt.subplot(1, 3, 3)
plt.scatter(Y_val, val_predictions_lasso, label='Dự đoán', alpha=0.6)
plt.plot(Y_val, Y_val, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Lasso Regression: Giá thực tế vs Giá dự đoán (Tập xác thực)")
plt.legend()

plt.tight_layout()
plt.savefig('graph/lasso_regression.png')  # Lưu biểu đồ
plt.close()

### 3. Mô hình Neural Network (Mạng Nơron)

# Xây dựng mô hình mạng nơron với cấu hình mới
nn_model = MLPRegressor(
    hidden_layer_sizes=(10, 5),  # Tăng số lượng nơ-ron và sử dụng hai lớp ẩn
    max_iter=1000,
    alpha=1.0,  # Giảm regularization
    activation='relu',  # Sử dụng hàm kích hoạt ReLU
    random_state=2,
    early_stopping=True,
    n_iter_no_change=20,
    learning_rate_init=0.01  # Tăng learning rate
)
nn_model.fit(X_train_scaled, Y_train)

# Dự đoán trên tập huấn luyện, kiểm tra và xác thực
train_predictions_nn = nn_model.predict(X_train_scaled)
test_predictions_nn = nn_model.predict(X_test_scaled)
val_predictions_nn = nn_model.predict(X_val_scaled)

# Tính toán các chỉ số đánh giá cho Neural Network
train_r2_score_nn = r2_score(Y_train, train_predictions_nn)
test_r2_score_nn = r2_score(Y_test, test_predictions_nn)
val_r2_score_nn = r2_score(Y_val, val_predictions_nn)
train_mse_nn = mean_squared_error(Y_train, train_predictions_nn)
test_mse_nn = mean_squared_error(Y_test, test_predictions_nn)
val_mse_nn = mean_squared_error(Y_val, val_predictions_nn)
train_rmse_nn = np.sqrt(train_mse_nn)
test_rmse_nn = np.sqrt(test_mse_nn)
val_rmse_nn = np.sqrt(val_mse_nn)

print(
    f"Neural Network - R² trên tập huấn luyện: {train_r2_score_nn:.4f}, R² trên tập kiểm tra: {test_r2_score_nn:.4f}, R² trên tập xác thực: {val_r2_score_nn:.4f}")
print(
    f"Neural Network - MSE trên tập huấn luyện: {train_mse_nn:.4f}, MSE trên tập kiểm tra: {test_mse_nn:.4f}, MSE trên tập xác thực: {val_mse_nn:.4f}")
print(
    f"Neural Network - RMSE trên tập huấn luyện: {train_rmse_nn:.4f}, RMSE trên tập kiểm tra: {test_rmse_nn:.4f}, RMSE trên tập xác thực: {val_rmse_nn:.4f}")

# Vẽ biểu đồ giá thực tế và dự đoán trên tập huấn luyện, kiểm tra và xác thực
plt.figure(figsize=(18, 6))

# Biểu đồ cho tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(Y_train, train_predictions_nn, label='Dự đoán', alpha=0.6)
plt.plot(Y_train, Y_train, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Neural Network: Giá thực tế vs Giá dự đoán (Tập huấn luyện)")
plt.legend()

# Biểu đồ cho tập kiểm tra
plt.subplot(1, 3, 2)
plt.scatter(Y_test, test_predictions_nn, label='Dự đoán', alpha=0.6)
plt.plot(Y_test, Y_test, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Neural Network: Giá thực tế vs Giá dự đoán (Tập kiểm tra)")
plt.legend()

# Biểu đồ cho tập xác thực
plt.subplot(1, 3, 3)
plt.scatter(Y_val, val_predictions_nn, label='Dự đoán', alpha=0.6)
plt.plot(Y_val, Y_val, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Neural Network: Giá thực tế vs Giá dự đoán (Tập xác thực)")
plt.legend()

plt.tight_layout()
plt.savefig('graph/neural_network.png')  # Lưu biểu đồ
plt.close()

### 4. Stacking Regressor

# Định nghĩa mô hình stacking regressor
base_models = [
    ('linear', lin_reg_model),
    ('lasso', best_lass_reg_model),
    ('nn', nn_model)
]
stacking_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
stacking_model.fit(X_train, Y_train)

# Dự đoán trên tập huấn luyện, kiểm tra và xác thực
train_predictions_stacked = stacking_model.predict(X_train)
test_predictions_stacked = stacking_model.predict(X_test)
val_predictions_stacked = stacking_model.predict(X_val)

# Tính toán các chỉ số đánh giá cho Stacking Regressor
train_r2_score_stacked = r2_score(Y_train, train_predictions_stacked)
test_r2_score_stacked = r2_score(Y_test, test_predictions_stacked)
val_r2_score_stacked = r2_score(Y_val, val_predictions_stacked)
train_mse_stacked = mean_squared_error(Y_train, train_predictions_stacked)
test_mse_stacked = mean_squared_error(Y_test, test_predictions_stacked)
val_mse_stacked = mean_squared_error(Y_val, val_predictions_stacked)
train_rmse_stacked = np.sqrt(train_mse_stacked)
test_rmse_stacked = np.sqrt(test_mse_stacked)
val_rmse_stacked = np.sqrt(val_mse_stacked)

print(
    f"Stacking Regressor - R² trên tập huấn luyện: {train_r2_score_stacked:.4f}, R² trên tập kiểm tra: {test_r2_score_stacked:.4f}, R² trên tập xác thực: {val_r2_score_stacked:.4f}")
print(
    f"Stacking Regressor - MSE trên tập huấn luyện: {train_mse_stacked:.4f}, MSE trên tập kiểm tra: {test_mse_stacked:.4f}, MSE trên tập xác thực: {val_mse_stacked:.4f}")
print(
    f"Stacking Regressor - RMSE trên tập huấn luyện: {train_rmse_stacked:.4f}, RMSE trên tập kiểm tra: {test_rmse_stacked:.4f}, RMSE trên tập xác thực: {val_rmse_stacked:.4f}")

# Vẽ biểu đồ giá thực tế và dự đoán trên tập huấn luyện, kiểm tra và xác thực
plt.figure(figsize=(18, 6))

# Biểu đồ cho tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(Y_train, train_predictions_stacked, label='Dự đoán', alpha=0.6)
plt.plot(Y_train, Y_train, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Stacking Regressor: Giá thực tế vs Giá dự đoán (Tập huấn luyện)")
plt.legend()

# Biểu đồ cho tập kiểm tra
plt.subplot(1, 3, 2)
plt.scatter(Y_test, test_predictions_stacked, label='Dự đoán', alpha=0.6)
plt.plot(Y_test, Y_test, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Stacking Regressor: Giá thực tế vs Giá dự đoán (Tập kiểm tra)")
plt.legend()

# Biểu đồ cho tập xác thực
plt.subplot(1, 3, 3)
plt.scatter(Y_val, val_predictions_stacked, label='Dự đoán', alpha=0.6)
plt.plot(Y_val, Y_val, color='red', linestyle='--', label='Giá thực tế')
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("Stacking Regressor: Giá thực tế vs Giá dự đoán (Tập xác thực)")
plt.legend()

plt.tight_layout()
plt.savefig('graph/stacking_regressor.png')  # Lưu biểu đồ
plt.close()

# Lưu các mô hình
import joblib

joblib.dump(lin_reg_model, 'models/linear_regression_model.joblib')
joblib.dump(best_lass_reg_model, 'models/lasso_regression_model.joblib')
joblib.dump(nn_model, 'models/neural_network_model.joblib')
joblib.dump(stacking_model, 'models/stacking_regressor_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')