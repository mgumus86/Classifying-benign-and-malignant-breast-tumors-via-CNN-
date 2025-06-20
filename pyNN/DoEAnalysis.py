#%% DoE analysis for each chunk
import pandas as pd
from statsmodels.formula.api import ols

data = pd.read_csv('data_log - Copy.csv')
print(data.columns)

# %% Analyze Chunk 5

chunk5 = data.iloc[16:]

# Build a linear model for this whole thing
chunk5_model = ols("ValAcc ~ weightDecay + learningRate + activationFunction + weightDecay*learningRate + weightDecay*activationFunction + learningRate*activationFunction + weightDecay*learningRate*activationFunction", data=chunk5)
results = chunk5_model.fit()
results.summary()
# %%
