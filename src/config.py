import os

data_dir = "C:\\Users\\prans\\Python files\\HR Analytics"

def get_path(path_string):
    return os.path.join(data_dir, path_string)


train_set = get_path('input\\train_set.csv')
test_set = get_path('input\\test_set.csv')
train_X = get_path('input\\train_X.csv')
train_y = get_path('input\\train_y.csv')
test_df = get_path("input\\test_df.csv")
train_folds = get_path("input\\train_folds.csv")
model_path = get_path("models")
index = "employee_id"
params_path = get_path('parameters.json')