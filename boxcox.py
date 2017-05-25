import pandas as pd
from scipy.stats import boxcox

path = 'data/cotrain/all_data_final_real.csv'
log_path = 'data/cotrain/all_data_final_real_boxcox.csv'


def do_boxcox():
    data = pd.read_csv(path, index_col=0)

    ignore_list = ['updatedAt', 'bad', 'authorChannelId', 'canRate', 'textDisplay', 'reactions', 'id', 'parentId',
                   'viewerRating', 'hateLabel', 'authorProfileImageUrl',
                   'authorChannelUrl', 'good', 'neutral', 'publishedAt', 'videoId', 'authorDisplayName',
                   'textDisplayProcessed',
                   'after', 'averageNegativeCoefficient', 'averageNeutralCoefficient', 'averagePositiveCoefficient',
                   'before', 'diffBadWordsCount', 'diffBadWordsRatio', 'diffCapitalLetterRatio',
                   'diffCapitalWordRatio', 'diffEmoticonCount', 'diffEmoticonToWordRatio', 'diffFirstPronounsCount',
                   'diffFirstPronounsToWordRatio', 'diffHatewordsCount', 'diffHatewordsRatio', 'diffInsultsCount',
                   'diffInsultsRatio', 'diffLikeCount', 'diffNegativeCoefficient', 'diffNeutralCoefficient',
                   'diffPositiveCoefficient', 'diffProfanityWindow2', 'diffProfanityWindow3', 'diffProfanityWindow4',
                   'diffProfanityWindow5', 'diffPunctuationRatio', 'diffSecondPronounsCount',
                   'diffSecondPronounsToWordRatio', 'diffTextDisplayLength', 'diffTextDisplayProcessedLength',
                   'diffTextDisplayProcessedWordsCount', 'diffTextDisplayWordsCount', 'negativeCoefficient',
                   'neutralCoefficient', 'positiveCoefficient', 'sentimentLabel', 'userAverageNegativeCoefficient',
                   'userAverageNeutralCoefficient', 'userAveragePositiveCoefficient', 'userNumberOfComments',
                   'userNumberOfContent', 'userNumberOfReplies']

    for key in data:
        if key not in ignore_list:
            temp = []
            d_min = data[key].min()

            for x in data[key]:
                if d_min <= 1:
                    val = x - d_min + 1
                else:
                    val = x
                temp.append(val)

            data[key] = pd.DataFrame(boxcox(temp)[0])

    data.to_csv(log_path)
