import pandas as pd

path = 'data/cotrain/all_data_final_real_boxcox.csv'
norm_path = 'data/cotrain/all_data_final_real_boxcox_norm.csv'


def normalize():
    data = pd.read_csv(path, index_col=0)

    ignore_list = ['updatedAt', 'bad', 'authorChannelId', 'canRate', 'textDisplay', 'reactions', 'id', 'parentId',
                   'viewerRating', 'hateLabel', 'authorProfileImageUrl',
                   'authorChannelUrl', 'good', 'neutral', 'publishedAt', 'videoId', 'authorDisplayName',
                   'textDisplayProcessed']

    for key in data:
        if key not in ignore_list:
            temp = []
            d_min = data[key].min()
            d_max = data[key].max()

            for x in data[key]:
                if (d_max - d_min) != 0:
                    val = (x - d_min) / (d_max - d_min)
                    temp.append(val)
                else:
                    temp.append(0)

            data[key] = pd.DataFrame(temp)

    data.to_csv(norm_path)
