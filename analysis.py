import pandas as pd

df = pd.read_csv("./eval.csv")
for col in ["category_name", "brand_name"]:
    df2 = df[[col,
              "err"]].groupby(col).agg(avg_err=pd.NamedAgg(column="err",
                                                           aggfunc="mean"),
                                       med_err=pd.NamedAgg(column="err",
                                                           aggfunc="median"),
                                       std_err=pd.NamedAgg(column="err",
                                                           aggfunc="std"))
    top_max = []
    top_min = []
    for i in range(10):
        z = df2.loc[df2["avg_err"].idxmax()]
        top_max.append(z.name)
        df2 = df2.drop(df2["avg_err"].idxmax())

    for i in range(10):
        z = df2.loc[df2["avg_err"].idxmin()]
        top_min.append(z.name)
        df2 = df2.drop(df2["avg_err"].idxmin())

    df3 = pd.read_csv("./train.csv")

    max_analysis = df3[df3[col].isin(top_max)].groupby(col).agg(
        count=pd.NamedAgg(column="price", aggfunc="size"),
        med_price=pd.NamedAgg(column="price", aggfunc="median"),
        std_price=pd.NamedAgg(column="price", aggfunc="std"))

    max_analysis = max_analysis.dropna()
    max_analysis[col] = max_analysis.index.tolist()
    print(max_analysis)
    fig = max_analysis[[col, "std_price"]].plot(kind="bar").get_figure()
    fig.savefig("{}_max.png".format(col))

    min_analysis = df3[df3[col].isin(top_min)].groupby(col).agg(
        avg_err=pd.NamedAgg(column="price", aggfunc="size"),
        med_price=pd.NamedAgg(column="price", aggfunc="median"),
        std_price=pd.NamedAgg(column="price", aggfunc="std"))
    min_analysis = min_analysis.dropna()
    min_analysis[col] = min_analysis.index.tolist()
    print(min_analysis)
    fig = min_analysis[[col, "std_price"]].plot(kind="bar").get_figure()
    fig.savefig("{}_min.png".format(col))
