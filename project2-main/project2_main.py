""" scaffold for project2


"""
# IMPORTANT: You should not import any other modules. This means that the
# only import statements in this module should be the ones below. In
# particular, this means that you cannot import modules inside functions.

import os

import numpy as np
import pandas as pd
from pandas.core.interchange.from_dataframe import protocol_df_chunk_to_pandas

import tk_utils
from lectures.solutions.lec_pd_numpy import df_nan

#WE WILL DELETE THIS ONCE DONE
ROOTDIR = os.path.dirname(__file__)
DATDIR = os.path.join(ROOTDIR, 'data')
pth_prc_dat = os.path.join(DATDIR, 'prc0.dat')
pth_ret_dat = os.path.join(DATDIR, 'ret0.dat')

# ----------------------------------------------------------------------------
#   Aux functions
# ----------------------------------------------------------------------------
def read_dat(pth: str) -> pd.DataFrame:
    """ Create a data frame with the raw content of a .dat file.

    This function loads data from a `.dat` file into a data frame. It does not
    parse or clean the data, nor does it assign specific data types. All
    entries in the resulting data frame are stored as `str` instances, and all
    columns have an object `dtype`. This function can be used to load any
    `.dat` file.

    Parameters
    ----------
    pth: str
        The location of a .dat file.

    Returns
    -------
    frame:
        A data frame. The dtype of each column is 'object' and the type of
        each element is `str`
    """
    # IMPORTANT: Please do not modify this function
    return pd.read_csv(pth, dtype=str).astype(str)



def str_to_float(value: str) -> float:
    """ This function attempts to convert a string into a float. It returns a
    float if the conversion is successful and None otherwise.

    Parameters
    ----------
    value: str
        A string representing a float. Quotes and spaces will be ignored.

    Returns
    -------
    float or None
        A float representing the string or None

    """
    # IMPORTANT: Please do not modify this function
    out = value.replace('"', '').replace("'", '').strip()
    try:
        out = pd.to_numeric(out)
    except:
        return None
    return float(out)


def fmt_col_name(label: str) -> str:
    """ Formats a column name according to the rules specified in the "Project
    Description" slide

    Parameters
    ----------
    label: str
        The original column label. See the "Project description" slide for
        more information

    Returns
    -------
    str:
        The formatted column label.

    Examples
    --------

    - `fmt_col_name(' Close') -> 'close'

    - `fmt_col_name('Adj    Close') -> 'adj_close'

    """
    # <COMPLETE_THIS_PART>
    columnlabel = label.strip()
    columnlabel = '_'.join(columnlabel.replace('_', ' ').split())
    columnlabel = columnlabel.lower()
    return columnlabel


#TEST FUNCTION
# label = "Adj   Close "
# formatted_label = fmt_col_name(label)
# print(f"Formatted label: '{formatted_label}'")


def fmt_ticker(value: str) -> str:
    """ Formats a ticker value according to the rules specified in the "Project
    Description" slide

    Parameters
    ----------
    value: str
        The original ticker value

    Returns
    -------
    str:
        The formatted ticker value.

    """
    # Remove leading/trailing spaces and quotes
    formatted = value.strip().replace('"', '').replace("'", '')

    # Convert to uppercase
    formatted = formatted.upper()

    # Remove additional inner spaces
    formatted = ''.join(formatted.split())

    return formatted

def read_prc_dat(pth: str):
    """ This function produces a data frame with volume and return from a single
    `<PRICE_DAT>` file.

    This function should clean the original data in `<PRICE_DAT>` as described
    in the "Project description" slide.

    Returns should be computed using adjusted closing prices.


    Parameters
    ----------
    pth: str
        The location of a <PRICE_DAT> file. This file includes price and
        volume information for different tickers and dates. See the project
        description for more information on these files.


    Returns
    -------
    frame:
        A dataframe with formatted column names (in any order):

         Column     dtype
         ------     -----
         date       datetime64[ns]
         ticker     object
         return     float64
         volume     float64

    Notes
    -----

    Assume that there are no gaps in the time series of adjusted closing
    prices for each ticker.


    """
    # <COMPLETE_THIS_PART>
    def process_data(pth):
        df = read_dat(pth)

        df.columns = [fmt_col_name(col) for col in df.columns]

        for col in df.columns:
            if col not in ['ticker', 'date']:
                df[col] = df[col].apply(lambda x: str_to_float(x) if not isinstance(x, str) else x)
            elif col == 'date':
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

        df.replace([-99, -99.9], np.nan, inplace=True)

        if 'adj_close' in df.columns:
            df.loc[df['adj_close'] < 0, 'adj_close'] = np.nan
            df['return'] = df.groupby('ticker')['adj_close'].pct_change()

        df = df.sort_values(by=['ticker', 'date'])
        df = df[['date', 'ticker', 'return', 'volume']]
        df['ticker'] = df['ticker'].apply(fmt_ticker)
        df.reset_index(drop=True, inplace=True)

        return df


def read_ret_dat(pth: str) -> pd.DataFrame:
    """ This function produces a data frame with volume and returns from a single
    `<RET_DAT>` file.


    This function should clean the original data in `<RET_DAT>` as described
    in the "Project description" slide.

    Parameters
    ----------
    pth: str
        The location of a <RET_DAT> file. This file includes returns and
        volume information for different tickers and dates. See the project
        description for more information on these files.


    Returns
    -------
    frame:
        A dataframe with columns (in any order):

          Column        dtype
          ------        -----
          date          datetime64[ns]
          ticker        object
          return        float64
          volume        float64

    Notes
    -----
    This .dat file also includes market returns. Market returns are
    represented by a special ticker called 'MKT'

    """
    # <COMPLETE_THIS_PART>
    raw = pd.read_csv(pth, dtype=str).astype(str)
    raw.columns = [fmt_col_name(label) for label in raw.columns]
    raw['ticker'] = raw['ticker'].apply(fmt_ticker)

    raw['return'] = raw['return'].astype(str).str.replace('"', '').str.replace('O', '0')
    raw['volume'] = raw['volume'].astype(str).str.replace('"', '').str.replace('O', '0')

    raw['return'] = pd.to_numeric(raw['return'], errors='coerce')
    raw['volume'] = pd.to_numeric(raw['volume'], errors='coerce')

    raw = raw.astype({
        'date': 'datetime64[ns]',
        'ticker': 'object',
        'return': 'float64',
        'volume': 'float64',
    })
    cleaned_raw = raw[['date', 'ticker', 'return', 'volume']]
    cleaned_raw.rename(columns={'return': 'MKT'}, inplace=True)
    cleaned_raw = pd.DataFrame(data=cleaned_raw).set_index('date')
    return cleaned_raw



def mk_ret_df(
        pth_prc_dat: str,
        pth_ret_dat: str,
        tickers: list[str],
        ):
    """ Combine information from two sources to produce a data frame 
    with stock and market returns according to the following rules:

    - Returns should be computed using information from <PRICE_DAT>, if
      possible. If a ticker is not found in the <PRICE_DAT> file, then returns
      should be obtained from the <RET_DAT> file.

    - Market returns should always be obtained from the <RET_DAT> file.

    - Only dates with available market returns should be part of the index.


    Parameters
    ----------
    pth_prc_dat: str
        Location of the <PRICE_DAT> file with price and volume information.
        This is the same parameter as the one described in `read_prc_dat`

    pth_ret_dat: str
        Location of the <RET_DAT> file with returns and volume information.
        This is the same parameter as the one described in `read_ret_dat`

    tickers: list
        A list of (possibly unformatted) tickers to be included in the output
        data frame. 

    Returns
    -------
    frame:
        A data frame with a DatetimeIndex and the following columns (in any
        order):

        Column      dtype 
        ------      -----
        <tic0>      float64
        <tic1>      float64
        ...
        <ticN>      float64
        <mkt>       float64


        Where `<tic0>`, ..., `<ticN>` are formatted column labels with tickers
        in the list `tickers`, and `<mkt>` is the formatted column label
        representing market returns.

        Should only include dates with available market returns.


    """
    # <COMPLETE_THIS_PART>
    df_prc = pd.read_csv(pth_prc_dat, dtype=str).astype(str)
    df_ret = pd.read_csv(pth_ret_dat, dtype=str).astype(str)

    # Clean and format column names
    df_prc.columns = df_prc.columns.str.strip().str.replace(r'\s+|_+', '_', regex=True).str.lower()
    df_ret.columns = df_ret.columns.str.strip().str.replace(r'\s+|_+', '_', regex=True).str.lower()

    # Convert specified columns to numeric
    numeric_columns_prc = ['close', 'adj_close', 'high', 'low', 'open', 'volume']
    numeric_columns_ret = ['return', 'volume']

    for col in numeric_columns_prc:
        if col in df_prc.columns:
            df_prc[col] = df_prc[col].apply(str_to_float)
    for col in numeric_columns_ret:
        if col in df_ret.columns:
            df_ret[col] = df_ret[col].apply(str_to_float)

    # Clean tickers
    df_prc['ticker'] = df_prc['ticker'].apply(lambda x: x.replace('"', '').replace("'", '').strip().upper())
    df_ret['ticker'] = df_ret['ticker'].apply(lambda x: x.replace('"', '').replace("'", '').strip().upper())

    # Filter for specific tickers, ensuring that the market ticker 'MKT' is always included
    tickers = [tkr.upper() for tkr in tickers]
    df_prc = df_prc[df_prc['ticker'].isin(tickers)]
    df_ret = df_ret[df_ret['ticker'].isin(tickers + ['MKT'])]

    # Group by date and ticker, keeping the last available entry
    df_prc = df_prc.groupby(['date', 'ticker']).last().reset_index()
    df_ret = df_ret.groupby(['date', 'ticker']).last().reset_index()

    # Separate market returns and stock returns
    market_returns = df_ret[df_ret['ticker'] == 'MKT'][['date', 'return']].rename(columns={'return': 'mkt'})
    stock_prices = df_prc.pivot(index='date', columns='ticker', values='adj_close')
    stock_returns = df_ret.pivot(index='date', columns='ticker', values='return')

    # Ensure all requested tickers are present in stock_prices and stock_returns by adding any missing columns with NaN
    for ticker in tickers:
        if ticker not in stock_prices.columns:
            stock_prices[ticker] = np.nan
        if ticker not in stock_returns.columns:
            stock_returns[ticker] = np.nan

    # Calculate computed returns from stock_prices
    computed_returns = stock_prices.pct_change(fill_method=None)

    # Update computed returns with existing return data from df_ret
    computed_returns.update(stock_returns)

    # Forward-fill any missing values in computed returns and market returns
    computed_returns.ffill(inplace=True)
    market_returns.set_index('date', inplace=True)
    market_returns.ffill(inplace=True)

    # Merge market and computed returns with an outer join to retain all dates
    final_df = pd.merge(market_returns, computed_returns, left_index=True, right_index=True, how='outer')

    # Clean up final columns and index formatting
    final_df.columns = [col.lower() if col != 'mkt' else 'mkt' for col in final_df.columns]
    final_df.index = pd.to_datetime(final_df.index)

    return final_df

# TEST FUNCTION (should be deleted after we finish, or put it in a new module)
def _test_fmt_ticker():
    test_values = [" AAPL ", "'msft'", '"GOOGL"', " tsla ", "  amzn  ", "' ibm '", "'MKT'", '"o    p e n"']
    print("Testing `fmt_ticker` with sample inputs:")

    for value in test_values:
        formatted_value = fmt_ticker(value)
        print(f"Original: '{value}' -> Formatted: '{formatted_value}'")

def _test_read_ret_dat():
    # Load the data
    test_df = read_ret_dat(pth_ret_dat)
    test_tickers = ["BAC", "CSCO", "PYPL"]
    filtered_df = test_df[test_df['ticker'].isin(test_tickers)]
    print("Output the 'read_ret_dat' with test_tickers:")
    print(filtered_df.head(10).to_string())

def _test_mk_ret_df():
    test_tickers = ["AAPL", "AAL", "BAC"]
    df = mk_ret_df(pth_prc_dat, pth_ret_dat, test_tickers)
    print("Output of `mk_ret_df` for test tickers:")
    print(df.head(10).to_string())


# This function is used to test the output of fmt_ticker, and apply the output to mk_ret_df to check the workflow
# i.e. uppercase tickers(fmt) -> lowercase(ret_df) -> process(ret_df)
def _test_fmt_mk_ret_df():
    raw_tickers = ["Aa Pl","A   al","BAc"]
    formatted_tickers = [fmt_ticker(ticker) for ticker in raw_tickers]
    print("Formatted tickers:", formatted_tickers)
    df = mk_ret_df(pth_prc_dat, pth_ret_dat, formatted_tickers)
    print("Output of `mk_ret_df` for formatted tickers:")
    print(df.head(10).to_string())

if __name__ == "__main__":
    _test_fmt_ticker()
    _test_read_ret_dat()
    _test_mk_ret_df()
    _test_fmt_mk_ret_df()
    pass

