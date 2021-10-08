import nltk
import numpy

nltk.download('stopwords')
nltk.download('punkt')

N = 100  # Number of words to consider
cluster_threshold = 5  # Distance between words to consider
top_sentences = 5  # number of sentences for a top-n-summary

# Extend stopwords
stopwords = nltk.corpus.stopwords.words('english') + [
    '.',
    ',',
    '!',
    "''",
    '``',
    '--',
    '\'s',
    '?',
    ')',
    '(',
    ':',
    '\'',
    '\'re',
    '"',
    '-',
    '}',
    '{',
    u'-',
    '>',
    '<',
    '...',
    '’']


def score_sentences(sentences: str, important_words: list) -> list:
    """
        Score a sentence by its words

        :param sentences: The sentences to score
        :param important_words: The words to score
        :return: A list of tuples containing the index of the sentence and its score

    """

    scores = []
    sentence_id_x = 0

    for s in [nltk.tokenize.word_tokenize(s) for s in sentences]:
        word_id_x = []

        for w in important_words:
            try:
                word_id_x.append(s.index(w))
            except ValueError:
                pass

        word_id_x.sort()

        if len(word_id_x) == 0:
            continue

        clusters = []
        cluster = [word_id_x[0]]
        i = 1

        while i < len(word_id_x):
            if word_id_x[i] - word_id_x[i-1] < cluster_threshold:
                cluster.append(word_id_x[i])
            else:
                clusters.append(cluster[:])
                cluster = [word_id_x[i]]
            i += 1
        clusters.append(cluster)

        # Cluster scores
        max_cluster_score = 0

        for cl in clusters:
            significant_words_in_cluster = len(cl)
            total_words_in_cluster = cl[-1] - cl[0] + 1
            score = 1.0 * significant_words_in_cluster**2 / total_words_in_cluster

            if score > max_cluster_score:
                max_cluster_score = score

        scores.append(((sentence_id_x, max_cluster_score)))
        sentence_id_x += 1

    return scores


def summarize(txt: str) -> dict:
    """
        Summarize a text

        :param txt: The text to summarize
        :return: A dictionary containing the summary and the top sentences
    """

    sentences = [s for s in nltk.tokenize.sent_tokenize(txt)]
    normalized_sentences = [s.lower() for s in sentences]

    words = [w.lower() for sentence in normalized_sentences for w in
             nltk.tokenize.word_tokenize(sentence)]

    fdist = nltk.FreqDist(words)

    # Remove stopwords from fdist
    for sw in stopwords:
        del fdist[sw]

    top_n_words = [w[0] for w in fdist.most_common(N)]

    scored_sentences = score_sentences(normalized_sentences, top_n_words)

    # Summarization Approach 1:
    # Filter out nonsignificant sentences by using the average score plus a
    # fraction of the std dev as a filter

    avg = numpy.mean([s[1] for s in scored_sentences])
    std = numpy.std([s[1] for s in scored_sentences])
    mean_scored = [(sent_idx, score) for (sent_idx, score) in scored_sentences
                   if score > avg + 0.5 * std]

    # Summarization Approach 2:
    # Another approach would be to return only the top N ranked sentences

    top_n_scored = sorted(
        scored_sentences, key=lambda s: s[1])[-top_sentences:]
    top_n_scored = sorted(top_n_scored, key=lambda s: s[0])

    # Decorate the post object with summaries

    return dict(top_n_summary=[sentences[idx] for (idx, score) in top_n_scored],
                mean_scored_summary=[sentences[idx] for (idx, score) in mean_scored])
