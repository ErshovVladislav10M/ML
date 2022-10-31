import random

import numpy as np
import numpy.linalg as la
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

import loader

dataset1 = loader.load_normalized_dataset("dataset1.csv")
test_dataset1 = loader.load_normalized_dataset("test_dataset1.csv")

dataset2 = loader.load_normalized_dataset("dataset2.csv")
test_dataset2 = loader.load_normalized_dataset("test_dataset2.csv")

dataset3 = loader.load_normalized_dataset("dataset3.csv")
test_dataset3 = loader.load_normalized_dataset("test_dataset3.csv")

dataset4 = loader.load_normalized_dataset("dataset4.csv")
test_dataset4 = loader.load_normalized_dataset("test_dataset4.csv")

dataset5 = loader.load_normalized_dataset("dataset5.csv")
test_dataset5 = loader.load_normalized_dataset("test_dataset5.csv")


def separate_x_y(data_frame: pd.DataFrame) -> tuple:
    """Separate dataset to X* and Y."""
    num_columns = len(data_frame.columns)
    x = data_frame.iloc[:, : num_columns - 1]
    y = data_frame.iloc[:, num_columns - 1:]

    return x, y


def insert_bias(data_frame: pd.DataFrame) -> None:
    data_frame.insert(0, "bias", 1, True)


x1, y1 = separate_x_y(dataset1)
insert_bias(x1)
x1 = x1.to_numpy()
y1 = y1.to_numpy()

test_x1, test_y1 = separate_x_y(test_dataset1)
insert_bias(test_x1)
test_x1 = test_x1.to_numpy()
test_y1 = test_y1.to_numpy()

x2, y2 = separate_x_y(dataset2)
insert_bias(x2)
x2 = x2.to_numpy()
y2 = y2.to_numpy()

test_x2, test_y2 = separate_x_y(test_dataset2)
insert_bias(test_x2)
test_x2 = test_x2.to_numpy()
test_y2 = test_y2.to_numpy()

x3, y3 = separate_x_y(dataset3)
insert_bias(x3)
x3 = x3.to_numpy()
y3 = y3.to_numpy()

test_x3, test_y3 = separate_x_y(test_dataset3)
insert_bias(test_x3)
test_x3 = test_x3.to_numpy()
test_y3 = test_y3.to_numpy()

x4, y4 = separate_x_y(dataset4)
insert_bias(x4)
x4 = x4.to_numpy()
y4 = y4.to_numpy()

test_x4, test_y4 = separate_x_y(test_dataset4)
insert_bias(test_x4)
test_x4 = test_x4.to_numpy()
test_y4 = test_y4.to_numpy()

x5, y5 = separate_x_y(dataset5)
insert_bias(x5)
x5 = x5.to_numpy()
y5 = y5.to_numpy()

test_x5, test_y5 = separate_x_y(test_dataset5)
insert_bias(test_x5)
test_x5 = test_x5.to_numpy()
test_y5 = test_y5.to_numpy()


def gradient_descent(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gradient descent."""
    previous = x.dot(w)
    _, col = x.shape
    return (2 / col) * x.T.dot(previous - y)


# Step function
init_step = 0.0001


def get_step(num_step) -> float:
    return init_step / num_step


# Gradient Decscsdcds
num_columns1 = len(dataset1.columns)
num_columns2 = len(dataset2.columns)
num_columns3 = len(dataset3.columns)
num_columns4 = len(dataset4.columns)
num_columns5 = len(dataset5.columns)

weights1 = np.array([[random.randint(-10, 10) * 0.001] for _ in range(num_columns1)])
weights2 = np.array([[random.randint(-10, 10) * 0.001] for _ in range(num_columns2)])
weights3 = np.array([[random.randint(-10, 10) * 0.001] for _ in range(num_columns3)])
weights4 = np.array([[random.randint(-10, 10) * 0.001] for _ in range(num_columns4)])
weights5 = np.array([[random.randint(-10, 10) * 0.001] for _ in range(num_columns4)])


def teach(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    num_step = 1
    epsilon = 0.0001

    new_weights = weights - (get_step(num_step) * gradient_descent(x, y, weights))
    norm = la.norm(weights - new_weights)
    flag = True

    while flag:
        print(num_step, norm)
        if norm < epsilon:
            flag = False
        new_weights = weights - (get_step(num_step) * gradient_descent(x, y, weights))
        norm = la.norm(weights - new_weights)
        weights = new_weights
        num_step = num_step + 1

    return new_weights


ans1 = teach(x1, y1, weights1)
# pd.DataFrame(ans1).to_csv('ans1.csv', header=False, index=False)
ans2 = teach(x2, y2, weights2)
# pd.DataFrame(ans2).to_csv('ans2.csv', header=False, index=False)
ans3 = teach(x3, y3, weights3)
# pd.DataFrame(ans3).to_csv('ans3.csv', header=False, index=False)
ans4 = teach(x4, y4, weights4)
# pd.DataFrame(ans4).to_csv('ans4.csv', header=False, index=False)
ans5 = teach(x5, y5, weights5)


# pd.DataFrame(ans5).to_csv('ans5.csv', header=False, index=False)


def print_results() -> None:
    test1_mse = mean_squared_error(test_y1, test_x1.dot(ans1), squared=False)
    test1_r2 = r2_score(test_y1, test_x1.dot(ans1))
    print("FOLD 1 RMSE ON TEST:", test1_mse)
    print("FOLD 1 R^2 ON TEST:", test1_r2)

    train1_mse = mean_squared_error(y1, x1.dot(ans1), squared=False)
    train1_r2 = r2_score(y1, x1.dot(ans1))
    print("FOLD 1 RMSE ON TRAIN:", train1_mse)
    print("FOLD 1 R^2 ON TRAIN:", train1_r2)

    test2_mse = mean_squared_error(test_y2, test_x2.dot(ans2), squared=False)
    test2_r2 = r2_score(test_y2, test_x2.dot(ans2))
    print("FOLD 2 RMSE ON TEST:", test2_mse)
    print("FOLD 2 R^2 ON TEST:", test2_r2)

    train2_mse = mean_squared_error(y2, x2.dot(ans2), squared=False)
    train2_r2 = r2_score(y2, x2.dot(ans2))
    print("FOLD 2 RMSE ON TRAIN:", train2_mse)
    print("FOLD 2 R^2 ON TRAIN:", train2_r2)

    test3_mse = mean_squared_error(test_y3, test_x3.dot(ans3), squared=False)
    test3_r2 = r2_score(test_y3, test_x3.dot(ans3))
    print("FOLD 3 RMSE ON TEST:", test3_mse)
    print("FOLD 3 R^2 ON TEST:", test3_r2)

    train3_mse = mean_squared_error(y3, x3.dot(ans3), squared=False)
    train3_r2 = r2_score(y3, x3.dot(ans3))
    print("FOLD 3 RMSE ON TRAIN:", train3_mse)
    print("FOLD 3 R^2 ON TRAIN:", train3_r2)

    test4_mse = mean_squared_error(test_y4, test_x4.dot(ans4), squared=False)
    test4_r2 = r2_score(test_y4, test_x4.dot(ans4))
    print("FOLD 4 RMSE ON TEST:", test4_mse)
    print("FOLD 4 R^2 ON TEST:", test4_r2)

    train4_mse = mean_squared_error(y4, x4.dot(ans4), squared=False)
    train4_r2 = r2_score(y4, x4.dot(ans4))
    print("FOLD 4 RMSE ON TRAIN:", train4_mse)
    print("FOLD 4 R^2 ON TRAIN:", train4_r2)

    test5_mse = mean_squared_error(test_y5, test_x5.dot(ans5), squared=False)
    test5_r2 = r2_score(test_y5, test_x5.dot(ans5))
    print("FOLD 5 RMSE ON TEST:", test5_mse)
    print("FOLD 5 R^2 ON TEST:", test5_r2)

    train5_mse = mean_squared_error(y5, x5.dot(ans5), squared=False)
    train5_r2 = r2_score(y5, x5.dot(ans5))
    print("FOLD 5 RMSE ON TRAIN:", train5_mse)
    print("FOLD 5 R^2 ON TRAIN:", train5_r2)

    test_rmse_array = np.array([test1_mse, test2_mse, test3_mse, test4_mse, test5_mse])
    print(
        "RMSE ON TEST MEAN:",
        np.array([test1_mse, test2_mse, test3_mse, test4_mse, test5_mse]).mean(),
    )
    print(
        "RMSE ON TEST STD:",
        np.array([test1_mse, test2_mse, test3_mse, test4_mse, test5_mse]).std(),
    )

    test_r2_array = np.array([test1_r2, test2_r2, test3_r2, test4_r2, test5_r2])
    print(
        "R2 ON TEST MEAN:",
        np.array([test1_r2, test2_r2, test3_r2, test4_r2, test5_r2]).mean(),
    )
    print(
        "R2 ON TEST STD:",
        np.array([test1_r2, test2_r2, test3_r2, test4_r2, test5_r2]).std(),
    )

    train_rmse_array = np.array(
        [train1_mse, train2_mse, train3_mse, train4_mse, train5_mse]
    )
    print(
        "RMSE ON TRAIN MEAN:",
        np.array([train1_mse, train2_mse, train3_mse, train4_mse, train5_mse]).mean(),
    )
    print(
        "RMSE ON TRAIN STD:",
        np.array([train1_mse, train2_mse, train3_mse, train4_mse, train5_mse]).std(),
    )

    train_r2_array = np.array([train1_r2, train2_r2, train3_r2, train4_r2, train5_r2])
    print(
        "R2 ON TRAIN MEAN:",
        np.array([train1_r2, train2_r2, train3_r2, train4_r2, train5_r2]).mean(),
    )
    print(
        "R2 ON TRAIN STD:",
        np.array([train1_r2, train2_r2, train3_r2, train4_r2, train5_r2]).std(),
    )


print_results()
