# 筛选数据，
import numpy as np
import pandas as pd


# 筛选数据，
def expandRows(idx_list_list, windowsize=30) -> np.ndarray:
    """
    索引列表扩展 正负 windowsize大小，然后去重
    """
    my_array = np.arange(-windowsize, windowsize + 1)
    new_array = idx_list_list.copy()
    for item in my_array:
        new_array += [element + item for element in idx_list_list]
    ret = np.sort(list(dict.fromkeys(new_array)))

    return ret


def expand_dataframe(all_dataframe: pd.DataFrame, windowsize=30):
    """
    索引列表扩展 正负 windowsize大小，然后去重
    """
    # print(origindata['label'] != 0)
    indices_list = all_dataframe.index[all_dataframe['label'] != 0].tolist()  # + _origindata.index[_origindata['label'] == '卸'].tolist()
    indices_list = expandRows(indices_list)

    # 或者使用.iloc基于位置索引（如果是整数索引）
    return all_dataframe.iloc[indices_list]
