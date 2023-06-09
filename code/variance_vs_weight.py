from pathlib import Path
import pandas as pd
from scipy.stats import zscore
import statsmodels.api as sm

root = Path(__file__).parent.parent.absolute()

df_weight = pd.read_csv(root / "results" / "phoneme_weights_pho.csv")
# standardize
df_weight = (df_weight - df_weight.mean()) / df_weight.std()
X = df_weight[["svar", "count", "tvar"]]
y = df_weight["weight"]


X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
est.summary()

md = smf.mixedlm("weight ~ amplitude_variance", df, groups=df["subject_id"])
mdf = md.fit()
print(mdf.summary())
