import dataretrieval
from dataretrieval import nwis

df, meta = nwis.get_iv(sites='01013500', parameterCd=['00010'], start='2020-01-01', end='2020-01-05')
print(df.index.tz)  # should say UTC
print(df.head())