#!/usr/bin/env python
import datetime
import sys
import os
file_location = os.path.abspath(__file__) # Get current file abspath
root_directory = os.path.dirname(file_location) # Get root dir
sys.path.append(os.path.join(root_directory))
from utils.TrendReq import *
from binance.spot import Spot as Client
from utils.secrets import *
# Set up client connector
# RSA keys set up
[api_key, api_secret] = get_api_key()
client = Client(api_key, api_secret)
pytrend = TrendReq()
# Set up params
PAIR = "BTCUSDT"  # Pair Binance API query
INTERVAL = "1h"  # Interval Binance API query
INTERVAL_MS = int(60 * 60 * 60)
BINANCE_HEADER = ["Open Time", "Open Price", "High Price", "Low Price", "Close Price", "Volume",
                  "Close Time", "Quote Asset Volume", "Number of Trades", "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"]
KW_LIST = ["Bitcoin"]  # KW used by Google trend API


class Date:
    """
    Store a date, transform into ms timestamp
    """

    def __init__(self, year, month, day, hour, min=None):
        self.y = year
        self.m = month
        self.d = day
        self.h = hour
        self.min = 0 if min is None else min

    def get_timestamp(self):
        return int(datetime.datetime(self.y, self.m, self.d, self.h, self.min).timestamp() * 1000)

    def print(self):
        print("{}/{:02d}/{:02d} - {:02d}:{:02d}".format(self.y, self.m, self.d, self.h, self.min))


def get_klines(start, end) -> pd.DataFrame:
    return pd.DataFrame(client.klines(PAIR, INTERVAL, limit=1000, startTime=start, endTime=end),
                        columns=BINANCE_HEADER)


def _check_data(data_frame) -> pd.DataFrame:
    """
    Check and fix open time data
    """

    for i in range(len(data_frame.index) - 1):
        o1 = data_frame["Open Time"][i] + int(INTERVAL_MS / 1e3)
        o2 = data_frame["Open Time"][i + 1]

        if o1 != o2:  # Check if i and i+1 data are consistent
            s = data_frame.copy().xs(i)
            for inter in range(o1, o2, int(INTERVAL_MS / 1e3)):
                s["Open Time"] = inter
                for c in ["Volume", "Quote Asset Volume", "Number of Trades", "Taker Buy Base Asset Volume",
                          "Taker Buy Quote Asset Volume", "Close Time"]:
                    if c in data_frame.columns and c != "Close Time":
                        s[c] = 0
                    elif c in data_frame.columns and c == "Close Time":
                        s[c] = inter - 1 + int(INTERVAL_MS / 1e3)
                data_frame = pd.concat(
                    [data_frame[:i], pd.DataFrame(dict(s), columns=data_frame.columns, index=[len(data_frame.columns)]),
                     data_frame[i:]])
    return data_frame.sort_values(by=["Open Time"]).reset_index(drop=True)


def get_data_klines(start, end) -> pd.DataFrame:
    """
    Get k_lines calls to fetch Binance Klines data from start to end date
    """
    begin_date = start.get_timestamp()
    end_date = end.get_timestamp()
    data_frame = pd.DataFrame([], columns=BINANCE_HEADER)

    for time in range(begin_date, end_date, INTERVAL_MS):
        if time + INTERVAL_MS <= end_date:
            data_frame = pd.concat([data_frame, get_klines(time, time + INTERVAL_MS)], ignore_index=True)
        else:
            data_frame = pd.concat([data_frame, get_klines(time, end_date)], ignore_index=True)
        return _check_data(data_frame.drop_duplicates(subset=["Open Time"]).reset_index(drop=True))


def get_data_trend(start, end) -> pd.DataFrame:
    """"
    Fetch Google Search trend data
    """

    data_frame = pytrend.get_historical_interest(KW_LIST, start.y, start.m, start.d, start.h, end.y,
                                                 end.m, end.d, end.h)
    data_frame["Open Time"] = [int(float(df.strftime('%s.%f')) * 1000) for df in data_frame.index]
    return _check_data(data_frame.drop_duplicates(subset=["Open Time"]).reset_index(drop=True))


def test_time_dfs(df1, df2):
    """
    Checking the correlation between each "Open Time" data from two dfs.
    Print the row where the error happen if there is one
    """

    size = len(df2.index) if len(df2.index) < len(df1.index) else len(df1.index)
    for i in range(size):
        o1 = int(df1["Open Time"][i])
        o2 = int(df2["Open Time"][i])
        if o1 != o2:
            print(i, o1, o2)
            break


def formatting(dfs, cols) -> pd.DataFrame:
    """
    Formatting the final Data Frame. Dfs is a list with one or multiple DataFrame to be concatenated/
    Cols is a list of the columns required in the final DataFrame
    """

    final_df = pd.DataFrame
    for df in dfs:
        for col in df.columns:
            if col in cols and col not in final_df.columns:
                final_df[col] = df[col].copy()

    return final_df


def export_xlsx(excel_dir, file_xlsx):
    # Get dataset dir
    file_xlsx.to_excel(excel_dir, index=False)
    print(f'Exported to {excel_dir}')
