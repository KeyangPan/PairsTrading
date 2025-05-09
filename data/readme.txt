1. 由get_data_tickers中的代码获得nasdaq/nysq所有上市股票的代码
   保存至.txt文件方便WRDS读取

2. 使用cornell账号登入Warton Research Data Services(WRDS)
   https://johnson.library.cornell.edu/database/wharton-research-data-services-wrds/
   手动下载Center for Research in Security Prices (CRSP)股票数据
   
3. get_stock_tickers文件夹中，代码用于提取股票tickers并存为txt文件，tickers.txt可以用于从WRDS下载股票历史价格数据
