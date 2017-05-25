import re
import pandas as pd
from random import choice, shuffle

norm_path = 'data/all_labelled_data.csv'


# noinspection PyTypeChecker
def run(k):
    # load data
    data_file = pd.read_csv(norm_path, index_col=0)
    data = []
    for index, row in data_file.iterrows():
        data.append(row)

    # init chunks
    data_chunks = [[] for _ in range(k)]
    data_len = len(data)
    chunk_len = int(data_len / k)

    # create random list of index numbers
    indices = []
    indices.extend(range(0, data_len))
    shuffle(indices)

    chunk_num = 0

    # fill k chunks with same amount of randomly selected data rows
    for idx, index in enumerate(indices):
        data_chunks[chunk_num].append(data[index])

        if idx != 0 and idx % chunk_len == 0 and chunk_num < k - 1:
            chunk_num += 1

    # init sets
    train_sets_pre = [[] for _ in range(k)]
    train_sets = [[] for _ in range(k)]
    test_sets = [[] for _ in range(k)]

    # fill sets
    for i in range(k):
        for j in range(k):
            if j != i:
                train_sets_pre[i] += data_chunks[j]
            else:
                test_sets[i] += data_chunks[j]

    # oversample train sets
    for idx, train_set_pre in enumerate(train_sets_pre):
        data_0 = []
        data_1 = []
        data_temp = []

        for item in train_set_pre:
            if item['hateLabel'] == 0:
                data_0.append(item)
            elif item['hateLabel'] == 1:
                data_1.append(item)

        data_0_len = len(data_0)
        data_1_len = len(data_1)
        range_abs_diff = range(abs(len(data_0) - len(data_1)))

        if data_0_len > data_1_len:
            for _ in range_abs_diff:
                data_temp.append(choice(data_1))
        else:
            for _ in range_abs_diff:
                data_temp.append(choice(data_0))

        train_sets[idx] = data_0 + data_1 + data_temp
        shuffle(train_sets[idx])

    # header for csv file

    header = ',after,authorChannelId,authorChannelUrl,authorDisplayName,authorProfileImageUrl,averageBadWordsCount,' \
             'averageBadWordsRatio,averageCapitalLetterRatio,averageCapitalWordRatio,averageEmoticonCount,' \
             'averageEmoticonToWordRatio,averageFirstPronounsCount,averageFirstPronounsToWordRatio,' \
             'averageHatewordsCount,averageHatewordsRatio,averageInsultsCount,averageInsultsRatio,' \
             'averageLikeCount,averageNegativeCoefficient,averageNeutralCoefficient,averagePositiveCoefficient,' \
             'averageProfanityWindow2,averageProfanityWindow3,averageProfanityWindow4,averageProfanityWindow5,' \
             'averagePunctuationRatio,averageSecondPronounsCount,averageSecondPronounsToWordRatio,' \
             'averageTextDisplayLength,averageTextDisplayProcessedLength,averageTextDisplayProcessedWordsCount,' \
             'averageTextDisplayWordsCount,bad,badWordsCount,badWordsRatio,before,canRate,capitalLetterRatio,' \
             'capitalWordRatio,emoticonCount,emoticonToWordRatio,firstPronounsCount,firstPronounsToWordRatio,good,' \
             'hateLabel,hatewordsCount,hatewordsRatio,id,insultsCount,insultsRatio,isComment,likeCount,' \
             'negativeCoefficient,neutral,neutralCoefficient,parentId,positiveCoefficient,profanityWindow2,' \
             'profanityWindow3,profanityWindow4,profanityWindow5,publishedAt,punctuationRatio,reactions,' \
             'secondPronounsCount,secondPronounsToWordRatio,sentimentLabel,textDisplay,textDisplayLength,' \
             'textDisplayProcessed,textDisplayProcessedLength,textDisplayProcessedWordsCount,textDisplayWordsCount,' \
             'totalReplyCount,updatedAt,userAverageBadWordsCount,userAverageBadWordsRatio,' \
             'userAverageCapitalLetterRatio,userAverageCapitalWordRatio,userAverageEmoticonCount,' \
             'userAverageEmoticonToWordRatio,userAverageFirstPronounsCount,userAverageFirstPronounsToWordRatio,' \
             'userAverageHatewordsCount,userAverageHatewordsRatio,userAverageInsultsCount,' \
             'userAverageInsultsRatio,userAverageLikeCount,userAverageNegativeCoefficient,' \
             'userAverageNeutralCoefficient,userAveragePositiveCoefficient,userAverageProfanityWindow2,' \
             'userAverageProfanityWindow3,userAverageProfanityWindow4,userAverageProfanityWindow5,' \
             'userAveragePunctuationRatio,userAverageSecondPronounsCount,userAverageSecondPronounsToWordRatio,' \
             'userAverageTextDisplayLength,userAverageTextDisplayProcessedLength,' \
             'userAverageTextDisplayProcessedWordsCount,userAverageTextDisplayWordsCount,userNumberOfComments,' \
             'userNumberOfContent,userNumberOfReplies,videoId,viewerRating,diffBadWordsCount,diffBadWordsRatio,' \
             'diffCapitalLetterRatio,diffCapitalWordRatio,diffEmoticonCount,diffEmoticonToWordRatio,' \
             'diffFirstPronounsCount,diffFirstPronounsToWordRatio,diffHatewordsCount,diffHatewordsRatio,' \
             'diffInsultsCount,diffInsultsRatio,diffLikeCount,diffNegativeCoefficient,diffNeutralCoefficient,' \
             'diffPositiveCoefficient,diffProfanityWindow2,diffProfanityWindow3,diffProfanityWindow4,' \
             'diffProfanityWindow5,diffPunctuationRatio,diffSecondPronounsCount,diffSecondPronounsToWordRatio,' \
             'diffTextDisplayLength,diffTextDisplayProcessedLength,diffTextDisplayProcessedWordsCount,' \
             'diffTextDisplayWordsCount,readabilityScore,diversityScore,threadAnger,threadDisgust,threadFear,' \
             'threadJoy,threadSadness,threadAnalytical,threadConfident,threadTentative,threadOpennessBig5,' \
             'threadConscientiousnessBig5,threadExtraversionBig5,threadAgreeablenessBig5,threadEmotionalRangeBig5,' \
             'anger,disgust,fear,joy,sadness,analytical,confident,tentative,opennessBig5,conscientiousnessBig5,' \
             'extraversionBig5,agreeablenessBig5,emotionalRangeBig5\n'

    # fill csv files
    fill_csv(k, header, train_sets, "data/norm_sets/train/train_data")
    fill_csv(k, header, test_sets, "data/norm_sets/test/test_data")


def fill_csv(k, header, sets, path):
    for i in range(k):
        with open('%s%d.csv' % (path, i), 'w') as ud:
            ud.write(header)

            for idx, item in enumerate(sets[i]):
                ud.write(
                    str(idx) + ',' +
                    str(item['after']) + ',' +
                    str('"' + item['authorChannelId']) + '",' +
                    str('"' + item['authorChannelUrl']) + '",' +
                    str('"' + item['authorDisplayName']) + '",' +
                    str('"' + item['authorProfileImageUrl']) + '",' +
                    str(item['averageBadWordsCount']) + ',' +
                    str(item['averageBadWordsRatio']) + ',' +
                    str(item['averageCapitalLetterRatio']) + ',' +
                    str(item['averageCapitalWordRatio']) + ',' +
                    str(item['averageEmoticonCount']) + ',' +
                    str(item['averageEmoticonToWordRatio']) + ',' +
                    str(item['averageFirstPronounsCount']) + ',' +
                    str(item['averageFirstPronounsToWordRatio']) + ',' +
                    str(item['averageHatewordsCount']) + ',' +
                    str(item['averageHatewordsRatio']) + ',' +
                    str(item['averageInsultsCount']) + ',' +
                    str(item['averageInsultsRatio']) + ',' +
                    str(item['averageLikeCount']) + ',' +
                    str(item['averageNegativeCoefficient']) + ',' +
                    str(item['averageNeutralCoefficient']) + ',' +
                    str(item['averagePositiveCoefficient']) + ',' +
                    str(item['averageProfanityWindow2']) + ',' +
                    str(item['averageProfanityWindow3']) + ',' +
                    str(item['averageProfanityWindow4']) + ',' +
                    str(item['averageProfanityWindow5']) + ',' +
                    str(item['averagePunctuationRatio']) + ',' +
                    str(item['averageSecondPronounsCount']) + ',' +
                    str(item['averageSecondPronounsToWordRatio']) + ',' +
                    str(item['averageTextDisplayLength']) + ',' +
                    str(item['averageTextDisplayProcessedLength']) + ',' +
                    str(item['averageTextDisplayProcessedWordsCount']) + ',' +
                    str(item['averageTextDisplayWordsCount']) + ',' +
                    str(item['bad']) + ',' +
                    str(item['badWordsCount']) + ',' +
                    str(item['badWordsRatio']) + ',' +
                    str(item['before']) + ',' +
                    str(item['canRate']) + ',' +
                    str(item['capitalLetterRatio']) + ',' +
                    str(item['capitalWordRatio']) + ',' +
                    str(item['emoticonCount']) + ',' +
                    str(item['emoticonToWordRatio']) + ',' +
                    str(item['firstPronounsCount']) + ',' +
                    str(item['firstPronounsToWordRatio']) + ',' +
                    str(item['good']) + ',' +
                    str(item['hateLabel']) + ',' +
                    str(item['hatewordsCount']) + ',' +
                    str(item['hatewordsRatio']) + ',' +
                    str('"' + item['id']) + '",' +
                    str(item['insultsCount']) + ',' +
                    str(item['insultsRatio']) + ',' +
                    str(item['isComment']) + ',' +
                    str(item['likeCount']) + ',' +
                    str(item['negativeCoefficient']) + ',' +
                    str(item['neutral']) + ',' +
                    str(item['neutralCoefficient']) + ',' +
                    str('"' + item['parentId']) + '",' +
                    str(item['positiveCoefficient']) + ',' +
                    str(item['profanityWindow2']) + ',' +
                    str(item['profanityWindow3']) + ',' +
                    str(item['profanityWindow4']) + ',' +
                    str(item['profanityWindow5']) + ',' +
                    str(item['publishedAt']) + ',' +
                    str(item['punctuationRatio']) + ',' +
                    str(item['reactions']) + ',' +
                    str(item['secondPronounsCount']) + ',' +
                    str(item['secondPronounsToWordRatio']) + ',' +
                    str(item['sentimentLabel']) + ',' +
                    str('"' + re.sub('"', '\'', item['textDisplay'])) + '",' +
                    str(item['textDisplayLength']) + ',' +
                    str('"' + re.sub('"', '\'', item['textDisplayProcessed'])) + '",' +
                    str(item['textDisplayProcessedLength']) + ',' +
                    str(item['textDisplayProcessedWordsCount']) + ',' +
                    str(item['textDisplayWordsCount']) + ',' +
                    str(item['totalReplyCount']) + ',' +
                    str(item['updatedAt']) + ',' +
                    str(item['userAverageBadWordsCount']) + ',' +
                    str(item['userAverageBadWordsRatio']) + ',' +
                    str(item['userAverageCapitalLetterRatio']) + ',' +
                    str(item['userAverageCapitalWordRatio']) + ',' +
                    str(item['userAverageEmoticonCount']) + ',' +
                    str(item['userAverageEmoticonToWordRatio']) + ',' +
                    str(item['userAverageFirstPronounsCount']) + ',' +
                    str(item['userAverageFirstPronounsToWordRatio']) + ',' +
                    str(item['userAverageHatewordsCount']) + ',' +
                    str(item['userAverageHatewordsRatio']) + ',' +
                    str(item['userAverageInsultsCount']) + ',' +
                    str(item['userAverageInsultsRatio']) + ',' +
                    str(item['userAverageLikeCount']) + ',' +
                    str(item['userAverageNegativeCoefficient']) + ',' +
                    str(item['userAverageNeutralCoefficient']) + ',' +
                    str(item['userAveragePositiveCoefficient']) + ',' +
                    str(item['userAverageProfanityWindow2']) + ',' +
                    str(item['userAverageProfanityWindow3']) + ',' +
                    str(item['userAverageProfanityWindow4']) + ',' +
                    str(item['userAverageProfanityWindow5']) + ',' +
                    str(item['userAveragePunctuationRatio']) + ',' +
                    str(item['userAverageSecondPronounsCount']) + ',' +
                    str(item['userAverageSecondPronounsToWordRatio']) + ',' +
                    str(item['userAverageTextDisplayLength']) + ',' +
                    str(item['userAverageTextDisplayProcessedLength']) + ',' +
                    str(item['userAverageTextDisplayProcessedWordsCount']) + ',' +
                    str(item['userAverageTextDisplayWordsCount']) + ',' +
                    str(item['userNumberOfComments']) + ',' +
                    str(item['userNumberOfContent']) + ',' +
                    str(item['userNumberOfReplies']) + ',' +
                    str('"' + item['videoId']) + '",' +
                    str(item['viewerRating']) + ',' +
                    str(item['diffBadWordsCount']) + ',' +
                    str(item['diffBadWordsRatio']) + ',' +
                    str(item['diffCapitalLetterRatio']) + ',' +
                    str(item['diffCapitalWordRatio']) + ',' +
                    str(item['diffEmoticonCount']) + ',' +
                    str(item['diffEmoticonToWordRatio']) + ',' +
                    str(item['diffFirstPronounsCount']) + ',' +
                    str(item['diffFirstPronounsToWordRatio']) + ',' +
                    str(item['diffHatewordsCount']) + ',' +
                    str(item['diffHatewordsRatio']) + ',' +
                    str(item['diffInsultsCount']) + ',' +
                    str(item['diffInsultsRatio']) + ',' +
                    str(item['diffLikeCount']) + ',' +
                    str(item['diffNegativeCoefficient']) + ',' +
                    str(item['diffNeutralCoefficient']) + ',' +
                    str(item['diffPositiveCoefficient']) + ',' +
                    str(item['diffProfanityWindow2']) + ',' +
                    str(item['diffProfanityWindow3']) + ',' +
                    str(item['diffProfanityWindow4']) + ',' +
                    str(item['diffProfanityWindow5']) + ',' +
                    str(item['diffPunctuationRatio']) + ',' +
                    str(item['diffSecondPronounsCount']) + ',' +
                    str(item['diffSecondPronounsToWordRatio']) + ',' +
                    str(item['diffTextDisplayLength']) + ',' +
                    str(item['diffTextDisplayProcessedLength']) + ',' +
                    str(item['diffTextDisplayProcessedWordsCount']) + ',' +
                    str(item['diffTextDisplayWordsCount']) + ',' +
                    str(item['readabilityScore']) + ',' +
                    str(item['diversityScore']) + ',' +
                    str(item['threadAnger']) + ',' +
                    str(item['threadDisgust']) + ',' +
                    str(item['threadFear']) + ',' +
                    str(item['threadJoy']) + ',' +
                    str(item['threadSadness']) + ',' +
                    str(item['threadAnalytical']) + ',' +
                    str(item['threadConfident']) + ',' +
                    str(item['threadTentative']) + ',' +
                    str(item['threadOpennessBig5']) + ',' +
                    str(item['threadConscientiousnessBig5']) + ',' +
                    str(item['threadExtraversionBig5']) + ',' +
                    str(item['threadAgreeablenessBig5']) + ',' +
                    str(item['threadEmotionalRangeBig5']) + ',' +
                    str(item['anger']) + ',' +
                    str(item['disgust']) + ',' +
                    str(item['fear']) + ',' +
                    str(item['joy']) + ',' +
                    str(item['sadness']) + ',' +
                    str(item['analytical']) + ',' +
                    str(item['confident']) + ',' +
                    str(item['tentative']) + ',' +
                    str(item['opennessBig5']) + ',' +
                    str(item['conscientiousnessBig5']) + ',' +
                    str(item['extraversionBig5']) + ',' +
                    str(item['agreeablenessBig5']) + ',' +
                    str(item['emotionalRangeBig5']) + '\n'
                )
