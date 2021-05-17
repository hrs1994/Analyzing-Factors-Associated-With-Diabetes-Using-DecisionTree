import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv('diabetes.csv')

profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
profile.to_file("your_report.html")