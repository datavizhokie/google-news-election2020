from GoogleNews import GoogleNews
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from datetime import datetime, date

def gen_cal_dates(start_date, end_date):

    delta = end_date - start_date
    datetime_list  = pd.date_range(end = end_date, periods = delta.days+1).to_pydatetime().tolist()
    
    return datetime_list
    
#datetime_list = gen_cal_dates(date(2020, 1, 1), date.today())
datetime_list = gen_cal_dates(date(2020, 9, 30), date.today())

print(f"There are {len(datetime_list)} days in the generated list")

stringdate_list = []
for i in range(len(datetime_list)):
    format_date = datetime.strftime(datetime_list[i], "%m/%d/%Y")
    stringdate_list.append(format_date)
    
min_date = stringdate_list[0] 
max_date = stringdate_list[-1]
min_date = min_date.replace("/", "-")
max_date = max_date.replace("/", "-")


def googlenews_extract(date_range, num_pages, search_text):

    ''' Use googlenews package to extract top 30 stories per day based on search string '''
    
    df_days = []
    
    # loop through date range to ensure equal sample size from each day
    #TODO: if we want to pull multiple years of data, perhaps add multi-threading...not necessary for < ~20 calls

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

    df_news = appended_data.reset_index(drop=True).drop(['date'], axis=1)
      
    return df_news

df_news = googlenews_extract(stringdate_list, 2, '2020 election')