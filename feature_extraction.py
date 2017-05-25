import re
import string
import pandas as pd
import ast
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from urllib import parse, request
from textstat.textstat import textstat
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

path = 'data/cotrain/all_data_final_real.csv'
bad_words_path = 'data/bad_words_list.txt'
emoticons_path = 'data/emoticons.txt'
insults_path = 'data/insults.txt'
hatewords_path = 'data/hatewords.txt'


def remove_feature(feature):
    data = pd.read_csv(path, index_col=0)
    data.drop(feature, 1, inplace=True)
    data.to_csv(path)


def extract_bad_words():
    data = pd.read_csv(path, index_col=0)

    with open(bad_words_path, 'r') as bad_words_file:
        bad_words = []

        for word in bad_words_file.readlines():
            word = re.sub('\n', '', word)
            bad_words.append(word)

    exclude = set(string.punctuation)

    exp = '(%s)' % '|'.join(bad_words)

    comments = data['textDisplayProcessed']
    word_count = data['textDisplayProcessedWordsCount']

    bw = []
    bwr = []

    for idx, comment in enumerate(comments):
        line = ''.join(char for char in comment if char not in exclude)
        l = len(re.findall(exp, line.casefold()))
        bw.append(l)
        bwr.append(l / word_count[idx])

    data['badWordsCount'] = pd.DataFrame(bw)
    data['badWordsRatio'] = pd.DataFrame(bwr)

    data.to_csv(path)

    print('extract_bad_words DONE')


def extract_cap_letters():
    data = pd.read_csv(path, index_col=0)

    comments = data['textDisplay']

    cl = []

    for comment in comments:
        cap = len(re.findall('[A-Z]', comment))
        letters = len(re.findall('[A-Za-z]', comment))

        if letters == 0:
            cl.append(0)
        else:
            cl.append(cap / letters)

    data['capitalLetterRatio'] = pd.DataFrame(cl)

    data.to_csv(path)

    print('extract_cap_letters DONE')


def extract_cap_words():
    data = pd.read_csv(path, index_col=0)

    comments = data['textDisplay']
    word_count = data['textDisplayWordsCount']

    cwr = []

    exclude = set(string.punctuation)

    for idx, comment in enumerate(comments):
        line = ''.join(char for char in comment if char not in exclude)
        cap = len(re.findall('\\b[A-Z]+\\b', line))
        cwr.append(cap / word_count[idx])

    data['capitalWordRatio'] = pd.DataFrame(cwr)

    data.to_csv(path)

    print('extract_cap_words DONE')


def extract_comment_length():
    data = pd.read_csv(path, index_col=0)

    comments = data['textDisplay']
    comments_processed = data['textDisplayProcessed']

    ln = []
    ln_p = []

    for comment in comments:
        ln.append(len(comment))

    for comment in comments_processed:
        ln_p.append(len(comment))

    data['textDisplayLength'] = pd.DataFrame(ln)
    data['textDisplayProcessedLength'] = pd.DataFrame(ln_p)

    data.to_csv(path)

    print('extract_comment_length DONE')


def extract_emoticons():
    data = pd.read_csv(path, index_col=0)

    emoticons = []

    with open(emoticons_path, 'r') as emoticons_file:
        for emoticon in emoticons_file.readlines():
            emoticon = re.sub('\n', '', emoticon)
            emoticons.append(re.escape(emoticon.casefold()))

    exp = '(%s)' % '|'.join(emoticons)

    comments = data['textDisplay']
    word_count = data['textDisplayWordsCount']

    em = []
    emr = []

    for idx, comment in enumerate(comments):
        em.append(len(re.findall(exp, comment.casefold())))
        emr.append(len(re.findall(exp, comment.casefold())) / word_count[idx])

    data['emoticonCount'] = pd.DataFrame(em)
    data['emoticonToWordRatio'] = pd.DataFrame(emr)

    data.to_csv(path)

    print('extract_emoticons DONE')


def extract_insults():
    data = pd.read_csv(path, index_col=0)

    insults = []

    with open(insults_path, 'r') as insults_file:
        for word in insults_file.readlines():
            word = re.sub('\n', '', word)
            insults.append(word.casefold())

    exclude = set(string.punctuation)

    exp = '(%s)' % '|'.join(insults)

    comments = data['textDisplayProcessed']
    word_count = data['textDisplayProcessedWordsCount']

    iw = []
    iwr = []

    for idx, comment in enumerate(comments):
        line = ''.join(char for char in comment if char not in exclude)
        l = len(re.findall(exp, line.casefold()))
        iw.append(l)
        iwr.append(l / word_count[idx])

    data['insultsCount'] = pd.DataFrame(iw)
    data['insultsRatio'] = pd.DataFrame(iwr)

    data.to_csv(path)

    print('extract_insults DONE')


def extract_hatewords():
    data = pd.read_csv(path, index_col=0)

    insults = []

    with open(hatewords_path, 'r') as insults_file:
        for word in insults_file.readlines():
            word = re.sub('\n', '', word)
            insults.append(word.casefold())

    exclude = set(string.punctuation)

    exp = '(%s)' % '|'.join(insults)

    comments = data['textDisplayProcessed']
    word_count = data['textDisplayProcessedWordsCount']

    iw = []
    iwr = []

    for idx, comment in enumerate(comments):
        line = ''.join(char for char in comment if char not in exclude)
        l = len(re.findall(exp, line.casefold()))
        iw.append(l)
        iwr.append(l / word_count[idx])

    data['hatewordsCount'] = pd.DataFrame(iw)
    data['hatewordsRatio'] = pd.DataFrame(iwr)

    data.to_csv(path)

    print('extract_hatewords DONE')


def extract_profanity_windows():
    data = pd.read_csv(path, index_col=0)

    pronouns = ['you', 'your', 'yours']
    bad_words = []

    with open(bad_words_path, 'r') as bad_words_file:
        for word in bad_words_file.readlines():
            word = re.sub('\n', '', word)
            bad_words.append(word)

    exclude = set(string.punctuation)

    comments = data['textDisplay']

    window_2 = []
    window_3 = []
    window_4 = []
    window_5 = []

    for comment in comments:
        line = ''.join(char for char in comment if char not in exclude)
        line_list = line.split()

        windows = {
            1: 0,
            2: 0,
            3: 0,
            4: 0
        }

        counter = 0
        word_found = False

        for word in line_list:
            if word_found:
                counter += 1

                if word in bad_words:
                    windows[counter] = 1
                    word_found = False
                    counter = 0

            if counter >= 4:
                word_found = False
                counter = 0

            if word in pronouns:
                word_found = True
                counter = 0

        window_2.append(windows[1])
        window_3.append(windows[2])
        window_4.append(windows[3])
        window_5.append(windows[4])

    data['profanityWindow2'] = pd.DataFrame(window_2)
    data['profanityWindow3'] = pd.DataFrame(window_3)
    data['profanityWindow4'] = pd.DataFrame(window_4)
    data['profanityWindow5'] = pd.DataFrame(window_5)

    data.to_csv(path)

    print('extract_profanity_windows DONE')


def extract_pronouns_count():
    data = pd.read_csv(path, index_col=0)

    pronouns_first = ['i', 'me', 'we', 'us', 'my', 'mine', 'our', 'ours']
    pronouns_second = ['you', 'your', 'yours']

    for idx, pronoun in enumerate(pronouns_first):
        pronouns_first[idx] = "\\b" + pronoun + "\\b"

    for idx, pronoun in enumerate(pronouns_second):
        pronouns_second[idx] = "\\b" + pronoun + "\\b"

    exclude = set(string.punctuation)

    exp_first = '(%s)' % '|'.join(pronouns_first)
    exp_second = '(%s)' % '|'.join(pronouns_second)

    comments = data['textDisplay']
    word_count = data['textDisplayWordsCount']

    pnr_f = []
    pn_f = []
    pnr_s = []
    pn_s = []

    for idx, comment in enumerate(comments):
        line = ''.join(char for char in comment if char not in exclude)
        pn_f.append(len(re.findall(exp_first, line.casefold())))
        pn_s.append(len(re.findall(exp_second, line.casefold())))
        pnr_f.append(len(re.findall(exp_first, line.casefold())) / word_count[idx])
        pnr_s.append(len(re.findall(exp_second, line.casefold())) / word_count[idx])

    data['firstPronounsCount'] = pd.DataFrame(pn_f)
    data['secondPronounsCount'] = pd.DataFrame(pn_s)
    data['firstPronounsToWordRatio'] = pd.DataFrame(pnr_f)
    data['secondPronounsToWordRatio'] = pd.DataFrame(pnr_s)

    data.to_csv(path)

    print('extract_pronouns_count DONE')


def extract_punctuation():
    data = pd.read_csv(path, index_col=0)

    comments = data['textDisplay']

    pr = []

    punctuation = '[' + string.punctuation + ']'

    for comment in comments:
        pr.append(len(re.findall(punctuation, comment)) / len(comment))

    data['punctuationRatio'] = pd.DataFrame(pr)

    data.to_csv(path)

    print('extract_punctuation DONE')


def remove_stopwords():
    lemmatizer = WordNetLemmatizer()

    data = pd.read_csv(path, index_col=0)

    comments = data['textDisplay']

    new_comments = []

    for idx, comment in enumerate(comments):
        line = [word for word in comment.split() if word.casefold() not in (stopwords.words('english'))]
        line = ' '.join([lemmatizer.lemmatize(word) for word in line])
        new_comments.append(line)

        if idx % 10 == 0:
            print(idx)

    data['textDisplayProcessed'] = pd.DataFrame(new_comments)

    data.to_csv(path)

    print('remove_stopwords DONE')


def extract_sentiment():
    data = pd.read_csv(path, index_col=0)

    comments = data['textDisplay']

    neg = []
    neutral = []
    pos = []
    label = []  # neg = 0, neutral = 1, pos = 2

    res = []

    for idx, comment in enumerate(comments, 1):
        my_data = parse.urlencode({"language": "english", "text": comment})
        request_headers = {
            "X-Mashape-Key": "DskPmxMLhxmsh9daKprt5IgtYFB0p1KtZnwjsnnkSn82RLmw9U",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        req = request.Request("https://japerk-text-processing.p.mashape.com/sentiment/",
                              data=str.encode(my_data),
                              headers=request_headers)

        # u = request.urlopen("http://text-processing.com/api/sentiment/", data=str.encode(my_data))
        u = request.urlopen(req)

        result = ast.literal_eval((u.read()).decode('ascii'))
        res.append(result)

        neg.append(result['probability']['neg'])
        neutral.append(result['probability']['neutral'])
        pos.append(result['probability']['pos'])

        if result['label'] == 'neg':
            label.append(0)
        elif result['label'] == 'neutral':
            label.append(1)
        elif result['label'] == 'pos':
            label.append(2)
        else:
            label.append(-1)

        print(idx)

        if idx % 100 == 0:
            with open('data/cotrain/temp/' + str(idx) + '.txt', 'w') as temp_file:
                for r in res:
                    temp_file.write(str(r))

    data['negativeCoefficient'] = pd.DataFrame(neg)
    data['neutralCoefficient'] = pd.DataFrame(neutral)
    data['positiveCoefficient'] = pd.DataFrame(pos)
    data['sentimentLabel'] = pd.DataFrame(label)

    data.to_csv(path)

    print('extract_sentiment DONE')


def extract_words_count():
    data = pd.read_csv(path, index_col=0)

    comments = data['textDisplay']
    comments_processed = data['textDisplayProcessed']

    count = []
    count_processed = []

    for comment in zip(comments, comments_processed):
        count.append(len(list(comment)[0].split()))
        count_processed.append(len(list(comment)[1].split()))

    data['textDisplayWordsCount'] = pd.DataFrame(count)
    data['textDisplayProcessedWordsCount'] = pd.DataFrame(count_processed)

    data.to_csv(path)

    print('extract_words_count DONE')


def determine_nature():
    data = pd.read_csv(path, index_col=0)

    parent_ids = data['parentId']

    is_comment = []

    for idx, parent_id in enumerate(parent_ids):
        if parent_id == '0':
            is_comment.append(1)
        else:
            is_comment.append(0)

    data['isComment'] = pd.DataFrame(is_comment)

    data.to_csv(path)

    print('determine_nature DONE')


def extract_avg_features():
    data = pd.read_csv(path, index_col=0)

    averageBadWordsCount = data['averageBadWordsCount']
    averageBadWordsRatio = data['averageBadWordsRatio']
    averageCapitalLetterRatio = data['averageCapitalLetterRatio']
    averageCapitalWordRatio = data['averageCapitalWordRatio']
    averageEmoticonCount = data['averageEmoticonCount']
    averageEmoticonToWordRatio = data['averageEmoticonToWordRatio']
    averageFirstPronounsCount = data['averageFirstPronounsCount']
    averageFirstPronounsToWordRatio = data['averageFirstPronounsToWordRatio']
    averageHatewordsCount = data['averageHatewordsCount']
    averageHatewordsRatio = data['averageHatewordsRatio']
    averageInsultsCount = data['averageInsultsCount']
    averageInsultsRatio = data['averageInsultsRatio']
    averageLikeCount = data['averageLikeCount']
    averageNegativeCoefficient = data['averageNegativeCoefficient']
    averageNeutralCoefficient = data['averageNeutralCoefficient']
    averagePositiveCoefficient = data['averagePositiveCoefficient']
    averageProfanityWindow2 = data['averageProfanityWindow2']
    averageProfanityWindow3 = data['averageProfanityWindow3']
    averageProfanityWindow4 = data['averageProfanityWindow4']
    averageProfanityWindow5 = data['averageProfanityWindow5']
    averagePunctuationRatio = data['averagePunctuationRatio']
    averageSecondPronounsCount = data['averageSecondPronounsCount']
    averageSecondPronounsToWordRatio = data['averageSecondPronounsToWordRatio']
    averageTextDisplayLength = data['averageTextDisplayLength']
    averageTextDisplayProcessedLength = data['averageTextDisplayProcessedLength']
    averageTextDisplayProcessedWordsCount = data['averageTextDisplayProcessedWordsCount']
    averageTextDisplayWordsCount = data['averageTextDisplayWordsCount']
    badWordsCount = data['badWordsCount']
    badWordsRatio = data['badWordsRatio']
    capitalLetterRatio = data['capitalLetterRatio']
    capitalWordRatio = data['capitalWordRatio']
    emoticonCount = data['emoticonCount']
    emoticonToWordRatio = data['emoticonToWordRatio']
    firstPronounsCount = data['firstPronounsCount']
    firstPronounsToWordRatio = data['firstPronounsToWordRatio']
    hatewordsCount = data['hatewordsCount']
    hatewordsRatio = data['hatewordsRatio']
    insultsCount = data['insultsCount']
    insultsRatio = data['insultsRatio']
    likeCount = data['likeCount']
    negativeCoefficient = data['negativeCoefficient']
    neutralCoefficient = data['neutralCoefficient']
    positiveCoefficient = data['positiveCoefficient']
    profanityWindow2 = data['profanityWindow2']
    profanityWindow3 = data['profanityWindow3']
    profanityWindow4 = data['profanityWindow4']
    profanityWindow5 = data['profanityWindow5']
    punctuationRatio = data['punctuationRatio']
    secondPronounsCount = data['secondPronounsCount']
    secondPronounsToWordRatio = data['secondPronounsToWordRatio']
    textDisplayLength = data['textDisplayLength']
    textDisplayProcessedLength = data['textDisplayProcessedLength']
    textDisplayProcessedWordsCount = data['textDisplayProcessedWordsCount']
    textDisplayWordsCount = data['textDisplayWordsCount']
    diffBadWordsCount = []
    diffBadWordsRatio = []
    diffCapitalLetterRatio = []
    diffCapitalWordRatio = []
    diffEmoticonCount = []
    diffEmoticonToWordRatio = []
    diffFirstPronounsCount = []
    diffFirstPronounsToWordRatio = []
    diffHatewordsCount = []
    diffHatewordsRatio = []
    diffInsultsCount = []
    diffInsultsRatio = []
    diffLikeCount = []
    diffNegativeCoefficient = []
    diffNeutralCoefficient = []
    diffPositiveCoefficient = []
    diffProfanityWindow2 = []
    diffProfanityWindow3 = []
    diffProfanityWindow4 = []
    diffProfanityWindow5 = []
    diffPunctuationRatio = []
    diffSecondPronounsCount = []
    diffSecondPronounsToWordRatio = []
    diffTextDisplayLength = []
    diffTextDisplayProcessedLength = []
    diffTextDisplayProcessedWordsCount = []
    diffTextDisplayWordsCount = []

    for i in range(len(averageBadWordsCount)):
        diffBadWordsCount.append(badWordsCount[i] - averageBadWordsCount[i])
        diffBadWordsRatio.append(badWordsRatio[i] - averageBadWordsRatio[i])
        diffCapitalLetterRatio.append(capitalLetterRatio[i] - averageCapitalLetterRatio[i])
        diffCapitalWordRatio.append(capitalWordRatio[i] - averageCapitalWordRatio[i])
        diffEmoticonCount.append(emoticonCount[i] - averageEmoticonCount[i])
        diffEmoticonToWordRatio.append(emoticonToWordRatio[i] - averageEmoticonToWordRatio[i])
        diffFirstPronounsCount.append(firstPronounsCount[i] - averageFirstPronounsCount[i])
        diffFirstPronounsToWordRatio.append(firstPronounsToWordRatio[i] - averageFirstPronounsToWordRatio[i])
        diffHatewordsCount.append(hatewordsCount[i] - averageHatewordsCount[i])
        diffHatewordsRatio.append(hatewordsRatio[i] - averageHatewordsRatio[i])
        diffInsultsCount.append(insultsCount[i] - averageInsultsCount[i])
        diffInsultsRatio.append(insultsRatio[i] - averageInsultsRatio[i])
        diffLikeCount.append(likeCount[i] - averageLikeCount[i])
        diffNegativeCoefficient.append(negativeCoefficient[i] - averageNegativeCoefficient[i])
        diffNeutralCoefficient.append(neutralCoefficient[i] - averageNeutralCoefficient[i])
        diffPositiveCoefficient.append(positiveCoefficient[i] - averagePositiveCoefficient[i])
        diffProfanityWindow2.append(profanityWindow2[i] - averageProfanityWindow2[i])
        diffProfanityWindow3.append(profanityWindow3[i] - averageProfanityWindow3[i])
        diffProfanityWindow4.append(profanityWindow4[i] - averageProfanityWindow4[i])
        diffProfanityWindow5.append(profanityWindow5[i] - averageProfanityWindow5[i])
        diffPunctuationRatio.append(punctuationRatio[i] - averagePunctuationRatio[i])
        diffSecondPronounsCount.append(secondPronounsCount[i] - averageSecondPronounsCount[i])
        diffSecondPronounsToWordRatio.append(secondPronounsToWordRatio[i] - averageSecondPronounsToWordRatio[i])
        diffTextDisplayLength.append(textDisplayLength[i] - averageTextDisplayLength[i])
        diffTextDisplayProcessedLength.append(textDisplayProcessedLength[i] - averageTextDisplayProcessedLength[i])
        diffTextDisplayProcessedWordsCount.append(textDisplayProcessedWordsCount[i] - averageTextDisplayProcessedWordsCount[i])
        diffTextDisplayWordsCount.append(textDisplayWordsCount[i] - averageTextDisplayWordsCount[i])

    data['diffBadWordsCount'] = pd.DataFrame(diffBadWordsCount)
    data['diffBadWordsRatio'] = pd.DataFrame(diffBadWordsRatio)
    data['diffCapitalLetterRatio'] = pd.DataFrame(diffCapitalLetterRatio)
    data['diffCapitalWordRatio'] = pd.DataFrame(diffCapitalWordRatio)
    data['diffEmoticonCount'] = pd.DataFrame(diffEmoticonCount)
    data['diffEmoticonToWordRatio'] = pd.DataFrame(diffEmoticonToWordRatio)
    data['diffFirstPronounsCount'] = pd.DataFrame(diffFirstPronounsCount)
    data['diffFirstPronounsToWordRatio'] = pd.DataFrame(diffFirstPronounsToWordRatio)
    data['diffHatewordsCount'] = pd.DataFrame(diffHatewordsCount)
    data['diffHatewordsRatio'] = pd.DataFrame(diffHatewordsRatio)
    data['diffInsultsCount'] = pd.DataFrame(diffInsultsCount)
    data['diffInsultsRatio'] = pd.DataFrame(diffInsultsRatio)
    data['diffLikeCount'] = pd.DataFrame(diffLikeCount)
    data['diffNegativeCoefficient'] = pd.DataFrame(diffNegativeCoefficient)
    data['diffNeutralCoefficient'] = pd.DataFrame(diffNeutralCoefficient)
    data['diffPositiveCoefficient'] = pd.DataFrame(diffPositiveCoefficient)
    data['diffProfanityWindow2'] = pd.DataFrame(diffProfanityWindow2)
    data['diffProfanityWindow3'] = pd.DataFrame(diffProfanityWindow3)
    data['diffProfanityWindow4'] = pd.DataFrame(diffProfanityWindow4)
    data['diffProfanityWindow5'] = pd.DataFrame(diffProfanityWindow5)
    data['diffPunctuationRatio'] = pd.DataFrame(diffPunctuationRatio)
    data['diffSecondPronounsCount'] = pd.DataFrame(diffSecondPronounsCount)
    data['diffSecondPronounsToWordRatio'] = pd.DataFrame(diffSecondPronounsToWordRatio)
    data['diffTextDisplayLength'] = pd.DataFrame(diffTextDisplayLength)
    data['diffTextDisplayProcessedLength'] = pd.DataFrame(diffTextDisplayProcessedLength)
    data['diffTextDisplayProcessedWordsCount'] = pd.DataFrame(diffTextDisplayProcessedWordsCount)
    data['diffTextDisplayWordsCount'] = pd.DataFrame(diffTextDisplayWordsCount)

    data.to_csv(path)

    print('extract_avg_features DONE')


def extract_readability_score():
    data = pd.read_csv(path, index_col=0)

    comments = data['textDisplay']

    readability = []

    for comment in comments:
        readability.append(textstat.flesch_kincaid_grade(comment))

    data['readabilityScore'] = pd.DataFrame(readability)

    data.to_csv(path)
    print('extract_readability_score DONE')


def extract_diversity_score():
    data = pd.read_csv(path, index_col=0)

    comments = data['textDisplay']

    diversity = []

    for comment in comments:
        diversity.append(len(comment)/len(set(comment)))

    data['diversityScore'] = pd.DataFrame(diversity)

    data.to_csv(path)
    print('extract_diversity_score DONE')


def longtail():
    data = pd.read_csv(path, index_col=0)

    ignore_list = ['updatedAt', 'bad', 'authorChannelId', 'canRate', 'textDisplay', 'reactions', 'id', 'parentId',
                   'viewerRating', 'hateLabel', 'authorProfileImageUrl',
                   'authorChannelUrl', 'good', 'neutral', 'publishedAt', 'videoId', 'authorDisplayName',
                   'textDisplayProcessed']

    for key in data:
        if key not in ignore_list:
            feature = data[key]

            plt.hist(feature, bins='auto')
            plt.title(key)
            plt.savefig('data/img/hist/%s.png' % key)
            plt.close()

            print(key)


def correlation():
    data = pd.read_csv('data/cotrain/all_data_final_real_norm.csv', index_col=0)

    ignore_list = ['updatedAt', 'bad', 'authorChannelId', 'canRate', 'textDisplay', 'reactions', 'id', 'parentId',
                   'viewerRating', 'hateLabel', 'authorProfileImageUrl',
                   'authorChannelUrl', 'good', 'neutral', 'publishedAt', 'videoId', 'authorDisplayName',
                   'textDisplayProcessed']

    with open('data/correlations.csv', 'a') as csv_file:
        for key in data:
            flag = False

            for key2 in data:
                if flag and key is not key2 and key not in ignore_list and key2 not in ignore_list:
                    feature = data[key]
                    feature2 = data[key2]

                    corr, p = pearsonr(feature, feature2)

                    csv_file.write('%s,%s,%s,%s\n' % (key, key2, corr, p))

                if key is key2:
                    flag = True
