import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/raw/kc_house_data_NaN.csv")
df.corr(numeric_only=True)["price"].sort_values(ascending=False)
def corr(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("images/feature_analysis/correlation_heatmap.png", dpi = 300, bbox_inches='tight')
    plt.show()
    plt.close()