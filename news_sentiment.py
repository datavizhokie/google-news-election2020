from GoogleNews import GoogleNews
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
from datetime import datetime, date
import sys


news_search_string  = '2020 election'
pages               = 4

def gen_cal_dates(start_date, end_date):

    delta = end_date - start_date
    datetime_list  = pd.date_range(end = end_date, periods = delta.days+1).to_pydatetime().tolist()
    
    return datetime_list
    

def googlenews_extract(date_range, num_pages, search_text):

    ''' Use googlenews package to extract top X stories per day based on search string '''
    
    df_days = []
    
    #TODO: add multi-threading

    for date in date_range:
        
        result = []
        googlenews = GoogleNews(start=date, end=date)
        googlenews.search(search_text)
        print("Search Date = ", date)
        
        for i in range(0, num_pages):

            print('Executing GoogleNews call #', i+1)

            googlenews.getpage(i)
            result_next = googlenews.result()
            print("Total records returned: ", len(result_next))
            
            df = pd.DataFrame(result_next)   
            df['date_calendar'] = date
        
        df_days.append(df) 
        appended_data = pd.concat(df_days)

    # Drop duplicate titles
    appended_data = appended_data.drop_duplicates(subset=['title'])

    # Append to master news df
    df_news = appended_data.reset_index(drop=True).drop(['date'], axis=1)
      
    return df_news


def tokenize_headlines_with_sentiment(df):
    
    headlines = df.title.tolist()

    all_bigrams = []

    headlines_string = (' '.join(filter(None, headlines))).lower()
    tokens = word_tokenize(headlines_string)

    # Remove single letter tokens
    tokens_sans_singles = [i for i in tokens if len(i) > 1]

    # Remove stop words
    stopwords = nltk.corpus.stopwords.words('english')
    new_words=("s'","'s","election", "2020", "n't", "wo")
    for i in new_words:
        stopwords.append(i)

    tokens_sans_stop = [t for t in tokens_sans_singles if t not in stopwords]

    # Get bigrams and frequencies
    bi_grams = list(ngrams(tokens_sans_stop, 2)) 
    counter = Counter(bi_grams)

    # Convert counter dictionary to dataframe
    counter_df = pd.DataFrame.from_dict(counter, orient='index').reset_index().rename(columns={"index": "bigram", 0: "freq"})
    counter_df_sort = counter_df.sort_values(by=['freq'],ignore_index=True, ascending=False)


    # Create concatenated bigram string for sentiment scoring
    counter_df_sort['word1'], counter_df_sort['word2'] = counter_df_sort.bigram.str
    counter_df_sort['bigram_joined'] = counter_df_sort.word1 + " " + counter_df_sort.word2
    counter_df_sort=counter_df_sort.drop(['word1','word2'], axis=1)

    # get sentiment for bigrams
    analyzer = SentimentIntensityAnalyzer()
    bigrams_scores = counter_df_sort['bigram_joined'].apply(analyzer.polarity_scores).tolist()
    df_bigrams_scores = pd.DataFrame(bigrams_scores).drop(['neg','neu','pos'], axis=1).rename(columns={"compound": "sentiment_compound"})
    bigrams_freq_and_scores = counter_df_sort.join(df_bigrams_scores, rsuffix='_right')


    print(f"There are {len(bigrams_freq_and_scores)} extracted bigrams in across all headlines")

    return bigrams_freq_and_scores


def headline_sentiment_scores(df, field):

    analyzer = SentimentIntensityAnalyzer()
    scores = df[field].apply(analyzer.polarity_scores).tolist()
    df_scores = pd.DataFrame(scores)
    df_scored = df.join(df_scores, rsuffix='_right')

    return df_scored


def main():

    datetime_list = gen_cal_dates(date(2020, 6, 1), date.today())
    #datetime_list = gen_cal_dates(date(2020, 10, 1), date.today())

    stringdate_list = []
    for i in range(len(datetime_list)):
        format_date = datetime.strftime(datetime_list[i], "%m/%d/%Y")
        stringdate_list.append(format_date)
        
    min_date = stringdate_list[0] 
    max_date = stringdate_list[-1]
    min_date = min_date.replace("/", "-")
    max_date = max_date.replace("/", "-")

    print(f"There are {len(datetime_list)} days in the generated list from {min_date} to {max_date}")

    # Call API and specify number of pages to extract
    df_news = googlenews_extract(stringdate_list, pages, news_search_string)

    # Subset to having a description (valid news stories)
    df_news_subset = df_news[df_news.desc != ""].reset_index(drop=True)

    daily_cnts = df_news_subset.groupby('date_calendar')['title'].count().reset_index().rename(columns={'title':'cnt_stories'})

    daily_avg_cnt = daily_cnts['cnt_stories'].mean()

    print(f"There are {len(df_news_subset)} valid stories for search string '{news_search_string}' in the generated dataset (across {pages} pages each day)")
    print(f"There are {daily_avg_cnt} average stories per date")

    # bigram sentiment
    bigrams_freq_and_scores = tokenize_headlines_with_sentiment(df_news_subset)

    # headline sentiment
    df_news_subset_scored = headline_sentiment_scores(df_news_subset, 'title')
    df_news_subset_scored2 = headline_sentiment_scores(df_news_subset_scored, 'desc')
    df_news_subset_scored2.rename(columns={'compound': 'compound_title', 'compound_right': 'compound_desc'}, inplace=True)
    
    # write results
    df_news_subset_scored2.to_csv(f"'Election 2020' News for {min_date} through {max_date} with Sentiment Scores.csv", index=False)
    bigrams_freq_and_scores.to_csv(f"'Election 2020' News Bigrams for {min_date} through {max_date} with Sentiment Scores.csv", index=False)


if __name__== "__main__" :
    main()