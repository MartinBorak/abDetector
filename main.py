from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
import feature_extraction as ef
import normalisation
import logaritmisation
import boxcox
import make_train_test as mtt
import scikit_train_test as stt


def main():
    # normalize()
    # divide_data(10)
    # play()
    # test()
    # logarithm()

    # for u in [1000]:
    #     for p in [5, 6, 7, 8, 9, 10]:
    #         for n in [1, 2, 3, 4]:
    #             for k_i in [10, 20, 30, 50, 75]:
    #                 if not (p == 8 and n == 2):
    #                     print(k_i)
    #                     cotrain_play(u, p, n, k_i)

    # cotrain_play(1000, 8, 2, 1)

    # do_boxcox()
    play()

    print('Done')


def cotrain_play(u, p, n, k_i):
    labels_1 = [
        # 'after',
        'badWordsCount',
        'badWordsRatio',
        # 'before',
        'capitalLetterRatio',
        'capitalWordRatio',
        'diversityScore',
        # 'emoticonCount',
        # 'emoticonToWordRatio',
        'firstPronounsCount',
        # 'firstPronounsToWordRatio',
        'hatewordsCount',
        'hatewordsRatio',
        'insultsCount',
        'insultsRatio',
        # 'isComment',
        # 'negativeCoefficient',
        'neutralCoefficient',
        # 'positiveCoefficient',
        # 'profanityWindow2',
        # 'profanityWindow3',
        # 'profanityWindow4',
        # 'profanityWindow5',
        'punctuationRatio',
        'readabilityScore',
        'secondPronounsCount',
        'secondPronounsToWordRatio',
        'sentimentLabel',
        'textDisplayLength',
        'textDisplayProcessedLength',
        'textDisplayProcessedWordsCount',
        'textDisplayWordsCount',
        # 'diffBadWordsCount',
        # 'diffBadWordsRatio',
        # 'diffCapitalLetterRatio',
        # 'diffCapitalWordRatio',
        # 'diffEmoticonCount',
        # 'diffEmoticonToWordRatio',
        # 'diffFirstPronounsCount',
        # 'diffFirstPronounsToWordRatio',
        # 'diffHatewordsCount',
        # 'diffHatewordsRatio',
        # 'diffInsultsCount',
        # 'diffInsultsRatio',
        # 'diffNegativeCoefficient',
        # 'diffNeutralCoefficient',
        # 'diffPositiveCoefficient',
        # 'diffProfanityWindow2',
        # 'diffProfanityWindow3',
        # 'diffProfanityWindow4',
        # 'diffProfanityWindow5',
        # 'diffPunctuationRatio',
        # 'diffSecondPronounsCount',
        # 'diffSecondPronounsToWordRatio',
        # 'diffTextDisplayLength',
        # 'diffTextDisplayProcessedLength',
        # 'diffTextDisplayProcessedWordsCount',
        # 'diffTextDisplayWordsCount',
        # 'threadAnger',
        # 'threadDisgust',
        # 'threadFear',
        # 'threadJoy',
        # 'threadSadness',
        # 'threadAnalytical',
        # 'threadConfident',
        # 'threadTentative',
        # 'threadOpennessBig5',
        # 'threadConscientiousnessBig5',
        # 'threadExtraversionBig5',
        # 'threadAgreeablenessBig5',
        # 'threadEmotionalRangeBig5',
        'anger',
        'disgust',
        'fear',
        'joy',
        'sadness',
        'analytical',
        'confident',
        'tentative',
        # 'opennessBig5',
        # 'conscientiousnessBig5',
        # 'extraversionBig5',
        # 'agreeablenessBig5',
        # 'emotionalRangeBig5'
    ]

    labels_2 = [
        'likeCount',
        'totalReplyCount',
        'diffLikeCount',
        'userAverageBadWordsCount',
        'userAverageBadWordsRatio',
        'userAverageCapitalLetterRatio',
        'userAverageCapitalWordRatio',
        'userAverageEmoticonCount',
        'userAverageEmoticonToWordRatio',
        'userAverageFirstPronounsCount',
        'userAverageFirstPronounsToWordRatio',
        'userAverageHatewordsCount',
        'userAverageHatewordsRatio',
        'userAverageInsultsCount',
        'userAverageInsultsRatio',
        'userAverageLikeCount',
        'userAverageNegativeCoefficient',
        'userAverageNeutralCoefficient',
        'userAveragePositiveCoefficient',
        'userAverageProfanityWindow2',
        # 'userAverageProfanityWindow3',
        # 'userAverageProfanityWindow4',
        # 'userAverageProfanityWindow5',
        'userAveragePunctuationRatio',
        'userAverageSecondPronounsCount',
        'userAverageSecondPronounsToWordRatio',
        'userAverageTextDisplayLength',
        'userAverageTextDisplayProcessedLength',
        'userAverageTextDisplayProcessedWordsCount',
        'userAverageTextDisplayWordsCount',
        'userNumberOfComments',
        'userNumberOfContent',
        'userNumberOfReplies'
    ]

    name_1 = 'ExtraTreesClassifier'
    name_2 = 'AdaBoostClassifier - ExtraTreesClassifier'

    classifier_1 = ExtraTreesClassifier(n_estimators=200,
                                        n_jobs=-1,
                                        max_features='sqrt',
                                        min_samples_leaf=1,
                                        max_depth=5)

    classifier_2 = AdaBoostClassifier(base_estimator=ExtraTreesClassifier(n_estimators=200,
                                                                          n_jobs=-1,
                                                                          max_features='sqrt',
                                                                          min_samples_leaf=1,
                                                                          max_depth=5),
                                      n_estimators=200)

    stt.cotrain(labels_1, labels_2, name_1, name_2, classifier_1, classifier_2, iterations=1, u=u, p=p, n=n, k_i=k_i)


def extract_features():
    # text features
    ef.extract_sentiment()
    ef.remove_stopwords()
    ef.extract_comment_length()
    ef.extract_words_count()
    ef.extract_cap_letters()
    ef.extract_cap_words()
    ef.extract_bad_words()
    ef.extract_emoticons()
    ef.extract_insults()
    ef.extract_profanity_windows()
    ef.extract_pronouns_count()
    ef.extract_punctuation()
    ef.extract_hatewords()
    ef.extract_readability_score()
    ef.extract_diversity_score()

    # other features
    ef.determine_nature()


def normalize():
    normalisation.normalize()


def logarithm():
    logaritmisation.logarithm()


def do_boxcox():
    boxcox.do_boxcox()


def divide_data(k):
    mtt.run(k)


def play():
    labels = [
        # 'after', #H
        'badWordsCount',
        'badWordsRatio',
        # 'before', #H
        'capitalLetterRatio',
        'capitalWordRatio',
        'diversityScore',
        'emoticonCount',
        'emoticonToWordRatio',
        'firstPronounsCount',
        'firstPronounsToWordRatio',
        'hatewordsCount',
        'hatewordsRatio',
        'insultsCount',
        'insultsRatio',
        # 'isComment', #H
        'negativeCoefficient',
        'neutralCoefficient',
        'positiveCoefficient',
        # 'profanityWindow2',
        # 'profanityWindow3',
        # 'profanityWindow4',
        # 'profanityWindow5',
        'punctuationRatio',
        'readabilityScore',
        'secondPronounsCount',
        'secondPronounsToWordRatio',
        'sentimentLabel',
        'textDisplayLength',
        'textDisplayProcessedLength',
        'textDisplayProcessedWordsCount',
        'textDisplayWordsCount',
        'totalReplyCount',
        # 'diffBadWordsCount', #H
        # 'diffBadWordsRatio', #H
        # 'diffCapitalLetterRatio', #H
        # 'diffCapitalWordRatio', #H
        # 'diffEmoticonCount', #H
        # 'diffEmoticonToWordRatio', #H
        # 'diffFirstPronounsCount', #H
        # 'diffFirstPronounsToWordRatio', #H
        # 'diffHatewordsCount', #H
        # 'diffHatewordsRatio', #H
        # 'diffInsultsCount', #H
        # 'diffInsultsRatio', #H
        # 'diffNegativeCoefficient', #H
        # 'diffNeutralCoefficient', #H
        # 'diffPositiveCoefficient', #H
        # 'diffProfanityWindow2', #H
        # 'diffProfanityWindow3', #H
        # 'diffProfanityWindow4', #H
        # 'diffProfanityWindow5', #H
        # 'diffPunctuationRatio', #H
        # 'diffSecondPronounsCount', #H
        # 'diffSecondPronounsToWordRatio', #H
        # 'diffTextDisplayLength', #H
        # 'diffTextDisplayProcessedLength', #H
        # 'diffTextDisplayProcessedWordsCount', #H
        # 'diffTextDisplayWordsCount', #H
        # 'threadAnger', #H
        # 'threadDisgust', #H
        # 'threadFear', #H
        # 'threadJoy', #H
        # 'threadSadness', #H
        # 'threadAnalytical', #H
        # 'threadConfident', #H
        # 'threadTentative', #H
        # 'threadOpennessBig5', #H
        # 'threadConscientiousnessBig5', #H
        # 'threadExtraversionBig5', #H
        # 'threadAgreeablenessBig5', #H
        # 'threadEmotionalRangeBig5', #H
        'anger',
        'disgust',
        'fear',
        'joy',
        'sadness',
        'analytical',
        'confident',
        'tentative',
        'opennessBig5',
        'conscientiousnessBig5',
        'extraversionBig5',
        'agreeablenessBig5',
        'emotionalRangeBig5',
        'likeCount',
        # 'diffLikeCount', #H
        'userAverageBadWordsCount',
        'userAverageBadWordsRatio',
        'userAverageCapitalLetterRatio',
        'userAverageCapitalWordRatio',
        'userAverageEmoticonCount',
        'userAverageEmoticonToWordRatio',
        'userAverageFirstPronounsCount',
        'userAverageFirstPronounsToWordRatio',
        'userAverageHatewordsCount',
        'userAverageHatewordsRatio',
        'userAverageInsultsCount',
        'userAverageInsultsRatio',
        'userAverageLikeCount',
        'userAverageNegativeCoefficient',
        'userAverageNeutralCoefficient',
        'userAveragePositiveCoefficient',
        'userAverageProfanityWindow2',
        # 'userAverageProfanityWindow3',
        # 'userAverageProfanityWindow4',
        # 'userAverageProfanityWindow5',
        'userAveragePunctuationRatio',
        'userAverageSecondPronounsCount',
        'userAverageSecondPronounsToWordRatio',
        'userAverageTextDisplayLength',
        'userAverageTextDisplayProcessedLength',
        'userAverageTextDisplayProcessedWordsCount',
        'userAverageTextDisplayWordsCount',
        'userNumberOfComments',
        'userNumberOfContent',
        'userNumberOfReplies'
    ]

    stt.test_all_classifiers(
        k=10,
        labels=labels)

    # stt.get_feature_importance(
    #     k=10,
    #     labels=labels,
    #     name='etc',
    #     classifier=ExtraTreesClassifier(n_estimators=200,
    #                                     n_jobs=-1,
    #                                     max_features='sqrt',
    #                                     min_samples_leaf=1,
    #                                     max_depth=5))

if __name__ == "__main__":
    main()

