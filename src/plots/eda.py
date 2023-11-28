from src.data_load import *
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from nltk.corpus import stopwords

plt.ion()
sns.set_context("paper")


def remove_stopwords(text):
    words = text.split()
    clean_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(clean_words)


def color_func_fake(
    word, font_size, position, orientation, random_state=None, **kwargs
):
    return "rgb(255, 0, 0)"  # Red color for fake news


def color_func_real(
    word, font_size, position, orientation, random_state=None, **kwargs
):
    return "rgb(137, 207, 240)"  # blue color for real news


stop_words = set(stopwords.words("english"))

if __name__ == "__main__":
    nela = NELADataset(path=f"../../data/nela")
    nela_df = nela.df
    tfg = TFGDataset(path=f"../../data/tfg")
    tfg_df = tfg.df
    isot = ISOTDataset(path=f"../../data/isot")
    isot_df = isot.df
    kaggle = KaggleFakeNewsDataset(path=f"../../data/kaggle_fake_news")
    kaggle_df = kaggle.df
    liar = LIARDataset(path=f"../../data/liar")
    liar_df = liar.df
    ticnn = TICNNDataset(path=f"../../data/ti_cnn")
    ticnn_df = ticnn.df
    datasets = [nela_df, liar_df, isot_df, kaggle_df, tfg_df, ticnn_df]

    for index, df in enumerate(datasets, 1):
        print(f"Dataset {index}:")
        print(df["label"].value_counts())

    dataset_names = ["NELA", "LIAR", "ISOT", "KAGGLE", "TFG", "TICNN"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 15))

    colors = ["#e74c3c", "#3498db"]  # Blue for 'Fake', Red for 'Real'

    for i, (ax, df, name) in enumerate(zip(axes.ravel(), datasets, dataset_names)):
        counts = df["label"].value_counts().sort_index()
        bars = counts.plot(kind="bar", color=colors, ax=ax, alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Fake", "Real"], rotation=0, fontsize=14)
        ax.set_title(name, fontweight="bold", fontsize=16)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_frame_on(False)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_ylim([0, 70000])
        ax.set_yticks(list(range(0, 70001, 5000)))
        ax.yaxis.grid(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(1.5)
        for spine in ["right", "top"]:
            ax.spines[spine].set_visible(False)

        if i % 3 != 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(list(range(0, 70001, 5000)), fontsize=12)

        for bar, value in zip(ax.patches, counts):
            height = bar.get_height()
            ax.annotate(
                f"{int(value)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("../../figs/class_dists.pdf", dpi=600)

    plt.figure(figsize=(15, 10))

    for idx, (name, df) in enumerate(zip(dataset_names, datasets)):
        fake_articles = df[df["label"] == 0]["content"].str.split().str.len()
        real_articles = df[df["label"] == 1]["content"].str.split().str.len()

        plt.subplot(2, 3, idx + 1)
        plt.hist(
            fake_articles,
            bins=np.arange(0, 500, 10),
            alpha=0.5,
            color="red",
            label="Fake News",
        )
        plt.hist(
            real_articles,
            bins=np.arange(0, 500, 10),
            alpha=0.5,
            color="blue",
            label="Real News",
        )
        plt.title(name + " Article Length Distribution")
        plt.xlabel("Length of Article (in words)")
        plt.ylabel("Number of Articles")
        plt.legend()

    plt.tight_layout()
    plt.savefig("../../figs/article_lengths.pdf", dpi=600)

    for df in datasets:
        df["content"] = df["content"].apply(remove_stopwords)

    vectorizer = TfidfVectorizer(max_df=0.85, stop_words="english", max_features=100)

    plt.figure(figsize=(20, 10))

    for i, df in enumerate(datasets):
        tfidf_matrix = vectorizer.fit_transform(df["content"])
        feature_names = vectorizer.get_feature_names_out()
        dense = tfidf_matrix.todense()
        denselist = dense.tolist()
        df_tfidf = pd.DataFrame(denselist, columns=feature_names).reset_index(drop=True)
        mask_fake = df["label"].reset_index(drop=True) == 0
        mask_real = df["label"].reset_index(drop=True) == 1
        fake_news_top_words = df_tfidf[mask_fake].sum().nlargest(20)
        real_news_top_words = df_tfidf[mask_real].sum().nlargest(20)

        wordcloud_fake = WordCloud(
            background_color="white", color_func=color_func_fake
        ).generate_from_frequencies(fake_news_top_words)
        wordcloud_real = WordCloud(
            background_color="white", color_func=color_func_real
        ).generate_from_frequencies(real_news_top_words)

        plt.subplot(3, 4, 2 * i + 1)
        plt.imshow(wordcloud_fake, interpolation="bilinear")
        plt.title(f"Fake News ({dataset_names[i]})")
        plt.xlabel("Frequency")
        plt.ylabel("Top Words")
        plt.axis("on")

        plt.subplot(3, 4, 2 * i + 2)
        plt.imshow(wordcloud_real, interpolation="bilinear")
        plt.title(f"Real News ({dataset_names[i]})")
        plt.xlabel("Frequency")
        plt.ylabel("Top Words")
        plt.axis("on")

    plt.tight_layout()
    plt.show()
    plt.savefig("../../figs/tfidf_combined.pdf", dpi=600)
