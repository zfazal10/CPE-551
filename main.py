from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier

def standardize_date(date):
    #print(date)
    parts = date.split(' ')
    month = parts[0]
    day = parts[1][:-1]
    year = parts[2][2:]
    return (month + '-' + day + '-' + year)

finviz_url = 'https://finviz.com/quote.ashx?t='
yahoo_fin_url = ['https://finance.yahoo.com/quote/', '/history?p=']
tickers = ['AMZN', 'GOOG', 'FB', 'AAPL', 'TSLA', 'MSFT', 'JPM', 'V', 'NVDA', 'JNJ', 'DIS', 'NFLX', 'WMT', 'BRK-B', 'CMCSA', 'PEP', 'HD', 'MA', 'UNH', 'XOM']
#tickers = ['AMZN']

news_tables = {}
price_tables = {}

ticker = 'GOOG'
url = yahoo_fin_url[0] + ticker + yahoo_fin_url[1] + ticker
req = Request(url=url)
response = urlopen(req)
html = BeautifulSoup(response, features='html.parser')
vader = SentimentIntensityAnalyzer()
model_data_features = []
model_data_target = []

for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')
    #price_tables[ticker] = price_table

    parsed_data = []

    for row in news_table.findAll('tr'):

        title = row.a.text
        date_data = row.td.text.split(' ')

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, title])
        #print([ticker, date, time, title])
#print(parsed_data[0])

    yahoo_url = yahoo_fin_url[0] + ticker + yahoo_fin_url[1] + ticker
    response2 = urlopen(yahoo_url).read()
    yahoo_html = BeautifulSoup(response2, 'html')
    html_str = str(yahoo_html)
    yahoo_html = html_str.split("<table")[1]
    table_html = yahoo_html.split("table>")[0]
    table_no_header_html = table_html.split("<tbody")[1]
    body_html = table_no_header_html.split("</tbody>")[0]
    rows_html = body_html.split("</tr>")
    flag = 0
    data_per_date = {}
    for row_html in rows_html:
        if row_html != "":
            cells_html = row_html.split("</span>")
            row = []
            for cell_html in cells_html:
                num = cell_html.count('>')
                cell = cell_html.split('>')[num]
                if ' ' in cell:
                    cell = standardize_date(cell)
                if cell != '':
                    row.append(cell)
            if row[1] != "Dividend":
                last_date = parsed_data[-1][1]
                for article in parsed_data:
                    if row[0] == article[1]:
                        flag = 0
                        break
                    else:
                        flag = 1
                if flag == 1:
                    # print(row)
                    break
                data_per_date[row[0]] = [row[1], row[2], row[3], row[4], row[6], 0] #default sentiment: 0


    articles_per_date = {}#TODO: we are in loop, dont need company name, how does that change other things?
    close_day_news = ""
    for article_row in parsed_data:
        key = article_row[1]

        if key not in data_per_date:
            #not real trading day
            close_day_news = close_day_news + ". " + article_row[3]
        elif key not in articles_per_date:    #does not exist yet
            articles_per_date[key] = [close_day_news + ". " + article_row[3]]
            close_day_news = ""
        else:                               #already exists
            articles_per_date[key][0] = articles_per_date[key][0] + ". " + close_day_news + ". " + article_row[3]
            close_day_news = ""
    for headlines in articles_per_date:
        #print(articles_per_date[headlines])
        sentiment_score = vader.polarity_scores(articles_per_date[headlines][0])['compound']
        if headlines not in data_per_date:
            print("error")
        else:
            data_per_date[headlines][5] = sentiment_score
    print(ticker)
    tmr_close = 0.0
    for data in data_per_date:
        if tmr_close != 0.0:
            data_per_date[data].append(tmr_close)
        tmr_close = data_per_date[data][3]

        if len(data_per_date[data]) == 7:
            new_data = []
            for number in data_per_date[data]:
                if isinstance(number, str):
                    number = number.replace(',', '')
                new_data.append(number)
            data_per_date[data] = new_data
            open = float(data_per_date[data][0])
            high = float(data_per_date[data][1])
            low = float(data_per_date[data][2])
            close = float(data_per_date[data][3])
            sent = float(data_per_date[data][5])
            target = float(data_per_date[data][6])

            high = high/open        #acaounting for stocks of varying prices
            low = low/open
            close = close/open
            target = target/open

            model_data_features.append([high, low, close, sent])
            model_data_target.append(target)

    print(model_data_features)
    print(len(model_data_features))

#factor in "volume" of articles?

x = np.array(model_data_features)
y = np.array(model_data_target)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(model_data_target)
print(acc)

bi_model_data_target = []
for target in model_data_target:
    if target > 1:
        bi_model_data_target.append(1)
    else:
        bi_model_data_target.append(0)

y = np.array(bi_model_data_target)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
classes = [[0, 1]]
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
print("KNN:")
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)

for x in range(len(x_test)):
    print("Predicted: ", predicted[x], "Data: ", x_test[x], "Actual: ", y_test[x])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)