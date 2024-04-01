# 加载数据

import os
import pandas as pd

categories = {"卸": 1, "-": 0, "装": 2}


def convert_to_int(data):
    return categories[data]


CREATE_STATS_CSV = False
# fixing_sates = True


def load_state(read_cache: bool = True):
    if read_cache and not CREATE_STATS_CSV and os.path.exists("./data/states.csv"):
        return pd.read_csv("./data/states.csv", parse_dates=["start", "end"], delimiter=',')
    dfstate = pd.read_csv("./data/states.txt")
    dfstate["start"] = dfstate.apply(lambda row: row["date"] + " " + row["start_time"], axis=1)
    dfstate["end"] = dfstate.apply(lambda row: row["date"] + " " + row["end_time"], axis=1)
    dfstate["label"] = dfstate["state"].str.strip().apply(convert_to_int)

    # 转为时间类型
    dfstate[["start", "end"]] = dfstate[["start", "end"]].apply(pd.to_datetime)
    # dfstate["statev"] = dfstate.apply(label_to_number, axis=1)
    # dfstate['label'] = dfstate.apply(number_to_label,axis=1)
    dfstate.to_csv("./data/states.csv")
    # print(dfstate.head(), dfstate.shape)
    return dfstate


# 定义读取数据函数
def fix_data(inputDF: pd.DataFrame, read_cache: bool = True) -> pd.DataFrame:
    _dfstate = load_state(read_cache)
    # 创建一个新的空列用于存储结果
    inputDF.insert(0, "label", categories["-"])
    # 对df1中的每一行遍历，并查找df2中符合条件的记录
    for index, row in inputDF.iterrows():

        condition = (_dfstate["start"] <= row["时间"]) & (row["时间"] <= _dfstate["end"])
        match = _dfstate[condition]
        if not match.empty:
            # 如果找到了匹配项，则将df2的'state'赋值给df1的新列
            inputDF.at[index, "label"] = match["label"].values[0]
        else:
            # 如果没有找到匹配项，则保持原样（这里已经初始化为-）
            pass

    return inputDF


creat_data_csv = False
# fixing_data = True


def load_fixed_data(file_path: str, read_cache: bool = True) -> pd.DataFrame:

    csv_path = file_path.replace('.xlsx', '.csv')

    if read_cache and not creat_data_csv and os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["时间", "轨迹时间"])

    _data = fix_data(pd.read_excel(file_path, engine="openpyxl", parse_dates=["时间", "轨迹时间"]), read_cache)
    _data.to_csv(csv_path)
    return _data


# DATA_17 = None

# create_global_vars(globals(), 'data')
