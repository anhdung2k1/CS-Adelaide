import os
import sys

file_location = os.path.abspath(__file__)  # Get current file abspath
root_directory = os.path.dirname(file_location)  # Get root dir
sys.path.append(os.path.join(root_directory, '..'))
from tools.scraping_data.scraping_bitcoin_RNN import *

# Params
_start = Date(2022, 8, 17, 0)
_end = Date(2023, 11, 14, 0)


def scraping_data_export(start, end, df) -> pd.DataFrame:
    begin_date = start.get_timestamp()
    end_date = end.get_timestamp()
    print(begin_date, end_date)
    request_num = len([k for k in range(begin_date, end_date, INTERVAL_MS)])
    print(f'Request Data: {request_num}')
    print(f'Export Data with row x cols: {df.shape}')
    dataset_file_dir = os.path.join(root_directory, '..', '..', 'dataset', 'BTCUSDT.xlsx')
    export_xlsx(dataset_file_dir, df)


def main():
    df1 = get_data_klines(_start, _end)
    # df2 = get_data_trend(_start, _end)
    scraping_data_export(_start, _end, df1)


if __name__ == "__main__":
    main()
