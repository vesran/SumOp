try:
    from sumop.params import PATH_TO_OUTPUT_DATA, ASPECTS
except ModuleNotFoundError:
    from params import PATH_TO_OUTPUT_DATA, ASPECTS

import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pandas as pd


def main():
    df = pd.read_csv(PATH_TO_OUTPUT_DATA)
    plot_sentiment_for_each_aspects(df)
    plot_aspects_distribution(df)



def plot_sentiment_for_each_aspects(df):
    for aspect in ASPECTS:
        sample = df.query(f'{aspect}==1')
        fig = px.pie(sample,
                     values=aspect,
                     names='sentiment',
                     color='sentiment',
                     title=f'Sentiments for {aspect}',
                     color_discrete_map={'POSITIVE':'#32CD32', 'NEGATIVE':'#d32f2f'})
        st.plotly_chart(fig)


def plot_aspects_distribution(df):
    asps = {a: df[a].sum() for a in ASPECTS}
    df_out = pd.DataFrame({
        "aspect": asps.keys(),
        "count": asps.values()
    })

    df_out = (df_out
                .sort_values('count', ascending=True)
            )

    fig = px.bar(df_out,
                 x='aspect',
                 y='count',
                 title='Number of opinions per aspect')
    st.plotly_chart(fig)


def plot_overall_sentiment(df):
    counts = df['sentiment'].value_counts().reset_index()

    fig, ax = plt.subplots(1, 1)
    counts.plot(kind='pie', y='sentiment', labels=counts['index'], ax=ax, autopct='%1.2f%%')
    st.pyplot(fig)


if __name__ == '__main__':
    main()
