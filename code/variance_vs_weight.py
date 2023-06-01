from pathlib import Path
import pandas as pd
import statsmodels.api as sm

root = Path(__file__).parent.parent.absolute()

df_weight = pd.read_csv(root / "results" / "weight_per_phoneme.csv")
df_variance = pd.read_csv(root / "results" / "variance_per_phoneme.csv")

weight = df_weight.weight.to_list()
tmp_var = df_variance.temporal_variance.to_list() * (df_weight.subject_id.max() + 1)
amp_var = df_variance.amplitude_variance.to_list() * (df_weight.subject_id.max() + 1)
count = df_variance["count"].to_list() * (df_weight.subject_id.max() + 1)

data = np.stack([weight, tmp_var, amp_var, count]).T
data = (data - data.mean(axis=0)) / data.std(axis=0)

df = pd.DataFrame(
    data=data, columns=["weight", "temporal_variance", "amplitude_variance", "count"]
)
X = df[["temporal_variance", "count", "amplitude_variance"]]
y = df["weight"]

X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
est.summary()
