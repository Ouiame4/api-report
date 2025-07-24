# api.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

app = FastAPI(title="API Analyse Veille Médiatique")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

desired_order = ['strongly positive', 'positive', 'neutral', 'negative', 'strongly negative']
palette_custom = ["#81C3D7", "#219ebc", "#D9DCD6", "#2F6690", "#16425B"]

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@app.post("/analyser")
async def analyser_csv(
    file: UploadFile = File(...),
    selected_authors: Optional[List[str]] = Form(None),
    min_year: Optional[int] = Form(None),
    max_year: Optional[int] = Form(None),
    granularity: Optional[str] = Form("Par mois")
):
    df = pd.read_csv(file.file)

    df['articleCreatedDate'] = pd.to_datetime(df['articleCreatedDate'], errors='coerce')
    df['sentimentHumanReadable'] = df['sentimentHumanReadable'].astype(str).str.strip().str.lower()
    df['Year'] = df['articleCreatedDate'].dt.year

    if min_year is None:
        min_year = int(df['Year'].min())
    if max_year is None:
        max_year = int(df['Year'].max())
    if selected_authors is None:
        selected_authors = df['authorName'].dropna().unique().tolist()

    df_filtered = df[
        (df['Year'] >= min_year) &
        (df['Year'] <= max_year) &
        (df['authorName'].isin(selected_authors))
    ]

    kpis = {
        "total_mentions": int(df_filtered.shape[0]),
        "positive": int(df_filtered[df_filtered['sentimentHumanReadable'] == 'positive'].shape[0]),
        "negative": int(df_filtered[df_filtered['sentimentHumanReadable'] == 'negative'].shape[0]),
        "neutral": int(df_filtered[df_filtered['sentimentHumanReadable'] == 'neutral'].shape[0]),
    }

    # Graph 1
    if granularity == "Par jour":
        df_filtered['Period'] = df_filtered['articleCreatedDate'].dt.date
    elif granularity == "Par semaine":
        df_filtered['Period'] = df_filtered['articleCreatedDate'].dt.to_period('W')
    elif granularity == "Par mois":
        df_filtered['Period'] = df_filtered['articleCreatedDate'].dt.to_period('M')
    elif granularity == "Par année":
        df_filtered['Period'] = df_filtered['articleCreatedDate'].dt.to_period('Y')
    else:
        df_filtered['Period'] = df_filtered['articleCreatedDate'].dt.to_period('M')

    mentions_over_time = df_filtered['Period'].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(mentions_over_time.index.astype(str), mentions_over_time.values, marker='o', linestyle='-', color="#2F6690")
    ax1.set_title(f"Évolution des mentions ({granularity.lower()})")
    ax1.set_xlabel("Période")
    ax1.set_ylabel("Mentions")
    plt.xticks(rotation=45)
    evolution_mentions_b64 = fig_to_base64(fig1)
    plt.close(fig1)

    # Graph 2
    sentiment_counts_raw = df_filtered['sentimentHumanReadable'].value_counts()
    sentiment_counts = pd.Series([sentiment_counts_raw.get(s, 0) for s in desired_order], index=desired_order)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=palette_custom, ax=ax2)
    ax2.set_ylabel("Nombre d'articles")
    ax2.set_xlabel("Sentiment")
    ax2.set_title("Répartition globale des sentiments")
    sentiments_global_b64 = fig_to_base64(fig2)
    plt.close(fig2)

    # Graph 3
    author_sentiment = df_filtered.groupby(['authorName', 'sentimentHumanReadable']).size().unstack(fill_value=0)
    author_sentiment['Total'] = author_sentiment.sum(axis=1)
    top_authors_sentiment = author_sentiment.sort_values(by='Total', ascending=False).head(10)
    top_authors_sentiment = top_authors_sentiment.drop(columns='Total')
    existing_sentiments = [s for s in desired_order if s in top_authors_sentiment.columns]
    top_authors_sentiment = top_authors_sentiment[existing_sentiments].iloc[::-1]
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    top_authors_sentiment.plot(kind='barh', stacked=True, ax=ax3,
        color=[palette_custom[desired_order.index(s)] for s in existing_sentiments])
    ax3.set_xlabel("Nombre d'articles")
    ax3.set_ylabel("Auteur / Source")
    ax3.set_title("Répartition des sentiments par auteur")
    sentiments_auteurs_b64 = fig_to_base64(fig3)
    plt.close(fig3)

    # Rapport HTML
    html_report = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Rapport de Veille Médiatique</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            h1 {{ color: #2F6690; }}
            .kpi {{ font-size: 18px; margin-bottom: 10px; }}
            .image {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>📊 Rapport de Veille Médiatique</h1>
        <p><strong>Période :</strong> {min_year} - {max_year}</p>
        <p><strong>Auteurs sélectionnés :</strong> {', '.join(selected_authors)}</p>
        <div class="kpi">
            <h2>🔢 Indicateurs Clés</h2>
            <ul>
                <li><strong>Mentions totales :</strong> {kpis['total_mentions']}</li>
                <li><strong>Positives :</strong> {kpis['positive']}</li>
                <li><strong>Négatives :</strong> {kpis['negative']}</li>
                <li><strong>Neutres :</strong> {kpis['neutral']}</li>
            </ul>
        </div>
        <div class="image">
            <h2>📈 Évolution des mentions</h2>
            <img src="data:image/png;base64,{evolution_mentions_b64}" width="700"/>
        </div>
        <div class="image">
            <h2>📊 Répartition globale des sentiments</h2>
            <img src="data:image/png;base64,{sentiments_global_b64}" width="600"/>
        </div>
        <div class="image">
            <h2>📊 Répartition des sentiments par auteur</h2>
            <img src="data:image/png;base64,{sentiments_auteurs_b64}" width="700"/>
        </div>
        <p style="margin-top: 40px; font-size: 12px; color: #999;">Généré automatiquement avec FastAPI</p>
    </body>
    </html>
    """

    return {
        "kpis": kpis,
        "html_report": html_report
    } 

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
