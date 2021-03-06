import scipy.io as sio
import pandas as pd
import os


def mat2csv():
    """ 将当前目录下的data目录下的 .mat 文件转换成多个 .csv文件 :return: """
    curr_path = os.path.dirname(__file__)
    mat_data_path = os.path.join(curr_path, "data")
    csv_data_path = os.path.join(curr_path, "csv")
    if not os.path.exists(csv_data_path):
        os.makedirs(csv_data_path)
    if not os.path.exists(mat_data_path):
        os.makedirs(mat_data_path)
    file_list = os.listdir(mat_data_path)
    mat_list = [file_name for file_name in file_list if file_name.endswith(".mat")]
    print("find mat file : ", mat_list)

    for mat_file in mat_list:
        file_path = os.path.join(mat_data_path, mat_file)
        mat_data = sio.loadmat(file_path)
        version = str(mat_data.get("__version__", "1.0")).replace(".", "_")
        for key in iter(mat_data.keys()):

            if not str(key).startswith("__"):
                print(key)
                # print(mat_data.keys())
                data = mat_data[key][:]
                dfdata = pd.DataFrame(data)

                csv_name = "_".join([mat_file.split(".")[0], key, '.csv'])
                csv_path = os.path.join(csv_data_path, csv_name)
                dfdata.to_csv(csv_path)


if __name__ == "__main__":
    mat2csv()
