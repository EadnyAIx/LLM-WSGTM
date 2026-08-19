import itertools
from typing import Callable, List, Union

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy as sch
from sklearn.metrics.pairwise import euclidean_distances


def select_topics(topic_model, top_n=None, topic_indices=None):
    weights = topic_model.get_topic_weights()
    if top_n is None and topic_indices is None:
        top_n = 5
    if top_n is not None:
        if top_n <= 0 or topic_indices is not None:
            raise ValueError("top_n must be positive and topic_indices must be omitted")
        topic_indices = np.argsort(weights)[:-(top_n + 1):-1]
    return topic_indices


def visualize_topics(topic_model, top_n=None, topic_indices=None, label_words=5, width=250, height=250, topic_labels=None):
    topic_indices = select_topics(topic_model, top_n, topic_indices)
    top_words = topic_model.top_words
    beta = topic_model.get_beta()
    titles = [f"Topic {i}" for i in topic_indices] if topic_labels is None else [topic_labels[i] for i in topic_indices]
    columns = 4
    rows = int(np.ceil(len(topic_indices) / columns))
    palette = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])
    figure = make_subplots(rows=rows, cols=columns, horizontal_spacing=0.1, subplot_titles=titles)
    row = 1
    column = 1
    for topic_id in topic_indices:
        words = top_words[topic_id].split()[:label_words][::-1]
        scores = np.sort(beta[topic_id])[:-(label_words + 1):-1][::-1]
        figure.add_trace(go.Bar(x=scores, y=words, orientation="h", marker_color=next(palette)), row=row, col=column)
        if column == columns:
            column = 1
            row += 1
        else:
            column += 1
    figure.update_layout(template="plotly_white", showlegend=False, title={"text": "Topic-Word Distributions", "x": 0.5, "xanchor": "center"}, width=width * 4, height=height * rows if rows > 1 else height * 1.3)
    figure.update_xaxes(showgrid=True)
    figure.update_yaxes(showgrid=True)
    return figure


def visualize_activity(topic_model, topic_activity: np.ndarray, time_slices: Union[np.ndarray, List], top_n=None, topic_indices=None, label_words=5, title="Topics Activity over Time", width=1000, height=600, topic_labels=None):
    topic_indices = select_topics(topic_model, top_n, topic_indices)
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]
    figure = go.Figure()
    if topic_labels is None:
        topic_labels = [f"{i}_{'_'.join(words.split()[:label_words])}" for i, words in enumerate(topic_model.top_words)]
    labels = np.unique(time_slices).tolist()
    for i, topic_id in enumerate(topic_indices):
        figure.add_trace(go.Scatter(x=labels, y=topic_activity[topic_id].tolist(), mode="lines", marker_color=colors[i % len(colors)], name=topic_labels[topic_id], hovertext=topic_labels[topic_id]))
    figure.update_layout(yaxis_title="Topic Weight", title={"text": title, "x": 0.4, "xanchor": "center"}, template="simple_white", width=width, height=height)
    return figure


def visualize_topic_weights(topic_model, top_n=50, topic_indices=None, label_words=5, title="Topic Weights", width=1000, height=1000, sort=True, topic_labels=None):
    weights = topic_model.get_topic_weights()
    topic_indices = select_topics(topic_model, top_n, topic_indices)
    labels = []
    values = []
    for topic_id in topic_indices:
        words = topic_model.top_words[topic_id]
        labels.append(topic_labels[topic_id] if topic_labels is not None else f"{topic_id}_{'_'.join(words.split()[:label_words])}")
        values.append(weights[topic_id])
    if sort:
        order = np.argsort(values)
        labels = np.asarray(labels)[order].tolist()
        values = np.asarray(values)[order].tolist()
    figure = go.Figure(go.Bar(x=values, y=labels, orientation="h"))
    figure.update_layout(xaxis_title="Weight", title={"text": title, "x": 0.5, "xanchor": "center"}, template="simple_white", width=width, height=height)
    return figure


def visualize_hierarchy(topic_model, orientation="left", width=1000, height=1000, linkage_function: Callable=None, distance_function: Callable=None, label_words=5, color_threshold=None, topic_labels=None):
    embeddings = topic_model.topic_embeddings
    distance_function = distance_function or euclidean_distances
    linkage_function = linkage_function or (lambda x: sch.linkage(x, "ward", optimal_ordering=True))
    if topic_labels is None:
        topic_labels = [f"{i}_{'_'.join(words.split()[:label_words])}" for i, words in enumerate(topic_model.top_words)]
    figure = ff.create_dendrogram(embeddings, orientation=orientation, labels=topic_labels, distfun=distance_function, linkagefun=linkage_function, color_threshold=color_threshold)
    figure.update_layout({"width": width, "height": height})
    return figure
