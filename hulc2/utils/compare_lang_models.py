import logging

import hydra
import numpy as np
import pandas
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers.util import pytorch_cos_sim
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

from hulc2.models.encoders.clip_lang_encoder import LangClip
from hulc2.models.encoders.language_network import SBert

logger = logging.getLogger(__name__)


def compute_score(model_name, cfg, compute_tsne=False):
    # lang_model = SBert(model_name)
    lang_model = LangClip(model_name=model_name)
    train_labels = []
    train_tasks = []
    train_embeddings = []
    instructions = []
    for i, (task, train_instructions) in enumerate(cfg.train_instructions.items()):
        for instruction in train_instructions:
            train_labels.append(i)
            train_tasks.append(task)
            train_embeddings.append(lang_model(instruction).cpu().numpy().squeeze().astype(np.float32))
            instructions.append(instruction)

    emb = np.array(train_embeddings)
    if compute_tsne:
        print("compute TSNE")
        tsne_emb = TSNE(n_components=2, random_state=40, perplexity=30.0).fit_transform(emb)

        print(tsne_emb.shape)
        data = pandas.DataFrame(
            {
                "task_id": train_labels,
                "task": train_tasks,
                "x": tsne_emb[:, 0].flatten(),
                "y": tsne_emb[:, 1].flatten(),
                "instruction": instructions,
            }
        )
        fig = go.Figure()
        task_scatter = px.scatter(
            data,
            x="x",
            y="y",
            hover_data={"instruction": True},
            color="task",
            color_discrete_sequence=px.colors.qualitative.Alphabet,
            labels={"color": "Tasks"},
        )
        for scatter in task_scatter.data:
            fig.add_trace(scatter)
        fig.show()
    performance_score = 0
    val_embeddings = []
    for i, (val_task, val_instructions) in enumerate(cfg.val_instructions.items()):
        val_instruction = val_instructions[0]
        val_emb = lang_model(val_instruction).cpu().numpy().squeeze().astype(np.float32)
        val_embeddings.append(val_emb)
        logger.debug("")
        logger.debug(f"{val_task} | {val_instruction}")
        results = []
        for task, instruction, train_emb in zip(train_tasks, train_instructions, train_embeddings):
            score = pytorch_cos_sim(train_emb, val_emb).item()
            results.append([task, instruction, score])
            if task == val_task:
                performance_score += score
            else:
                performance_score -= score
        for res in sorted(results, reverse=True, key=lambda x: x[2])[:10]:
            logger.debug(res)

        logger.debug("")
    logger.debug(f"{model_name} performance: {performance_score}")

    sr = classify(train_embeddings, train_labels, val_embeddings, list(range(len(val_embeddings))))

    return sr


def classify(train_embeddings, train_labels, val_embeddings, val_labels):
    clf = RandomForestClassifier()
    # clf = KMeans(n_clusters=100)
    X_train = np.array(train_embeddings)
    y_train = np.array(train_labels)
    p = np.random.permutation(len(y_train))
    X_train = X_train[p]
    y_train = y_train[p]

    X_test = np.array((val_embeddings))
    y_test = np.array(val_labels)
    clf.fit(X_train, y_train)

    p = np.random.permutation(len(y_test))
    X_test = X_test[p]
    y_test = y_test[p]

    y_test_hat = clf.predict(X_test)
    # sr = metrics.rand_score(y_test, y_test_hat)
    sr = np.mean(y_test_hat == y_test)
    return sr


@hydra.main(config_path="../../conf", config_name="lang_ann.yaml")
def main(cfg):
    # models = ["all-mpnet-base-v2",
    #           "all-mpnet-base-v1",
    #           "multi-qa-mpnet-base-dot-v1",
    #           "multi-qa-mpnet-base-cos-v1",
    #           "all-roberta-large-v1",
    #           "all-distilroberta-v1",
    #           "all-MiniLM-L12-v1",
    #           "all-MiniLM-L12-v2",
    #           "multi-qa-distilbert-dot-v1",
    #           "multi-qa-distilbert-cos-v1",
    #           "all-MiniLM-L6-v2",
    #           "multi-qa-MiniLM-L6-cos-v1",
    #           "all-MiniLM-L6-v1",
    #           "paraphrase-mpnet-base-v2",
    #           "msmarco-bert-base-dot-v5",
    #           "multi-qa-MiniLM-L6-dot-v1",
    #           "msmarco-distilbert-base-tas-b",
    #           "msmarco-distilbert-dot-v5",
    #           "paraphrase-distilroberta-base-v2",
    #           "paraphrase-MiniLM-L12-v2",
    #           "multi-qa-MiniLM-L6-dot-v1",
    #           "msmarco-distilbert-base-tas-b",
    #           "msmarco-distilbert-dot-v5",
    #           "paraphrase-distilroberta-base-v2",
    #           "paraphrase-MiniLM-L12-v2",
    #           "paraphrase-multilingual-mpnet-base-v2",
    #           "paraphrase-TinyBERT-L6-v2",
    #           "paraphrase-MiniLM-L6-v2",
    #           "paraphrase-albert-small-v2",
    #           "paraphrase-multilingual-MiniLM-L12-v2",
    #           "paraphrase-MiniLM-L3-v2",
    #           "distiluse-base-multilingual-cased-v1",
    #           "distiluse-base-multilingual-cased-v2",
    #           "average_word_embeddings_komninos",
    #           "average_word_embeddings_glove.6B.300d"]
    # models = ["msmarco-bert-base-dot-v5"]
    models = ["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"]
    scores = []
    for model in models:
        try:
            scores.append((model, compute_score(model, cfg)))
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")

    scores = sorted(scores, key=lambda x: x[1])
    for model, score in scores:
        print(f"{model}: {score}")


if __name__ == "__main__":
    main()
