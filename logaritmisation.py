import pandas as pd
from math import log10

path = 'data/cotrain/all_data_final_real.csv'
log_path = 'data/cotrain/all_data_final_real_log.csv'


def logarithm():
    data = pd.read_csv(path, index_col=0)

    ignore_list = ['updatedAt', 'bad', 'authorChannelId', 'canRate', 'textDisplay', 'reactions', 'id', 'parentId',
                   'viewerRating', 'hateLabel', 'authorProfileImageUrl',
                   'authorChannelUrl', 'good', 'neutral', 'publishedAt', 'videoId', 'authorDisplayName',
                   'textDisplayProcessed']

    for key in data:
        if key not in ignore_list:
            temp = []
            d_min = data[key].min()

            for x in data[key]:
                if d_min <= 1:
                    val = log10(x - d_min + 1)
                else:
                    val = log10(x)
                temp.append(val)

            data[key] = pd.DataFrame(temp)

    data.to_csv(log_path)
