"""Embeddings (OpenAI) + clustering (HDBSCAN/KMeans) and GPT descriptions.

Two strategies:
  - without k → HDBSCAN: discovers the number of clusters by density, flags outliers.
  - with k → KMeans with that k (explicit override).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean

import numpy as np
from openai import OpenAI
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from .client import Work

# Outlier label (HDBSCAN assigns -1 to points that do not form a cluster).
NOISE_LABEL = -1


def embed_works(
    works: list[Work], api_key: str, model: str, batch_size: int = 256
) -> np.ndarray:
    """Return an (n_works, dim) matrix with each work's embedding."""
    from .ssl_setup import ensure_system_trust

    ensure_system_trust()
    client = OpenAI(api_key=api_key)
    texts = [w.text_for_embedding or w.title for w in works]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        resp = client.embeddings.create(model=model, input=chunk)
        vectors.extend(d.embedding for d in resp.data)
    return np.array(vectors, dtype=np.float32)


@dataclass
class Cluster:
    label: int
    works: list[Work]
    top_topics: list[tuple[str, int]]
    top_fields: list[tuple[str, int]]
    top_domains: list[tuple[str, int]]
    top_keywords: list[tuple[str, int]]
    total_citations: int
    mean_citations: float
    mean_fwci: float | None
    n_top_10_percent: int  # number of works in the top-10% of their field/year
    representatives: list[Work]  # closest to the centroid
    is_noise: bool = False  # outlier cluster (HDBSCAN label -1)
    description: str | None = None  # GPT-generated description (optional)

    @property
    def size(self) -> int:
        return len(self.works)


def _representatives(
    works: list[Work], embeddings: np.ndarray, centroid: np.ndarray, n: int = 3
) -> list[Work]:
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    order = np.argsort(dists)[:n]
    return [works[i] for i in order]


# Number of components after reducing, before HDBSCAN. Density is unreliable in 1536
# dims (HDBSCAN over-flags outliers); reducing to a few dims fixes it. Verified across
# several datasets: ~5 dims minimizes false outliers in a stable way.
REDUCE_COMPONENTS = 5


# Below this N, UMAP is unstable (too few points to learn the manifold); 'auto' uses
# PCA. At or above it, 'auto' uses UMAP, which captures structure better.
UMAP_MIN_SAMPLES = 50


def _reduce(embeddings: np.ndarray, method: str, n_components: int) -> np.ndarray:
    """Reduce dimensionality before HDBSCAN. method: 'auto'|'umap'|'pca'|'none'.

    - 'auto' (default): PCA if N<UMAP_MIN_SAMPLES, UMAP otherwise (robust on small
      datasets, better structure on large ones).
    - 'umap': preserves local structure better (BERTopic standard) but needs enough
      points.
    - 'pca': linear, stable, no heavy dependencies.
    - 'none': clusters on the full embeddings (not recommended in 1536 dims).
    """
    n = len(embeddings)
    comps = min(n_components, n - 1, embeddings.shape[1])
    if method == "auto":
        method = "umap" if n >= UMAP_MIN_SAMPLES else "pca"
    if method == "none":
        return embeddings
    if method == "pca":
        return PCA(n_components=comps, random_state=42).fit_transform(embeddings)
    if method == "umap":
        import warnings

        import umap  # deferred: pulls in numba and is slow to import

        # n_neighbors scaled with N (15 is too large on small datasets).
        n_neighbors = min(15, max(2, n // 3))
        reducer = umap.UMAP(
            n_components=comps,
            n_neighbors=n_neighbors,
            metric="cosine",
            random_state=42,  # reproducible (disables parallelism: warning expected)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return reducer.fit_transform(embeddings)
    raise ValueError(f"Unknown reduction method: {method}")


def _assign_labels(
    embeddings: np.ndarray,
    k: int | None,
    min_cluster_size: int,
    reduce: str = "auto",
    n_components: int = REDUCE_COMPONENTS,
) -> np.ndarray:
    """fixed k → KMeans; no k → HDBSCAN (discovers cluster count by density).

    For HDBSCAN we first reduce dimensionality (`reduce`) and L2-normalize, so that
    euclidean distance is equivalent to cosine.
    """
    if k is not None:
        k = max(1, min(k, len(embeddings)))
        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(embeddings)

    reduced = _reduce(embeddings, reduce, n_components)
    unit = normalize(reduced)
    size = max(2, min(min_cluster_size, len(embeddings)))
    return HDBSCAN(
        min_cluster_size=size, min_samples=1, metric="euclidean", copy=True
    ).fit_predict(unit)


def cluster_works(
    works: list[Work],
    embeddings: np.ndarray,
    k: int | None = None,
    min_cluster_size: int = 2,
    reduce: str = "auto",
) -> tuple[list[Cluster], int]:
    """Group the works and summarize each cluster's impact.

    Returns (clusters, n_clusters). With HDBSCAN an outlier cluster (label -1) may
    appear; it is placed last and is not counted in n_clusters.
    """
    if len(works) < 2:
        raise ValueError("At least 2 articles are required to cluster.")

    labels = _assign_labels(embeddings, k, min_cluster_size, reduce=reduce)

    clusters: list[Cluster] = []
    for lbl in sorted(set(labels)):
        idxs = [i for i, x in enumerate(labels) if x == lbl]
        members = [works[i] for i in idxs]
        member_emb = embeddings[idxs]
        centroid = member_emb.mean(axis=0)
        cites = [w.cited_by_count for w in members]
        fwcis = [w.fwci for w in members if w.fwci is not None]
        topic_counts = Counter(w.topic for w in members if w.topic)
        field_counts = Counter(w.field for w in members if w.field)
        domain_counts = Counter(w.domain for w in members if w.domain)
        keyword_counts = Counter(kw for w in members for kw in w.keywords)

        clusters.append(
            Cluster(
                label=int(lbl),
                works=members,
                top_topics=topic_counts.most_common(3),
                top_fields=field_counts.most_common(3),
                top_domains=domain_counts.most_common(2),
                top_keywords=keyword_counts.most_common(5),
                total_citations=sum(cites),
                mean_citations=mean(cites) if cites else 0.0,
                mean_fwci=mean(fwcis) if fwcis else None,
                n_top_10_percent=sum(1 for w in members if w.is_top_10_percent),
                representatives=_representatives(members, member_emb, centroid),
                is_noise=(lbl == NOISE_LABEL),
            )
        )

    n_clusters = sum(1 for c in clusters if not c.is_noise)
    # Sort by impact (total citations) desc; outliers always last.
    clusters.sort(key=lambda c: (c.is_noise, -c.total_citations))
    return clusters, n_clusters


def _cluster_prompt(cluster: Cluster, max_articles: int, abstract_chars: int) -> str:
    lines: list[str] = []
    for w in cluster.works[:max_articles]:
        abstract = (w.abstract or "").strip().replace("\n", " ")
        if abstract:
            abstract = abstract[:abstract_chars]
        lines.append(f"- «{w.title}»\n  {abstract or '(no abstract)'}")
    return (
        "You are a scientific-literature analyst. Given the following articles "
        "(title and abstract) that belong to the same thematic cluster, write a "
        "2-3 sentence description in English capturing the common theme, the "
        "methodological approach, and what ties them together. Do not list the "
        "articles one by one; synthesize. Be concrete.\n\nArticles:\n"
        + "\n".join(lines)
    )


def describe_clusters(
    clusters: list[Cluster],
    api_key: str,
    model: str,
    max_articles: int = 8,
    abstract_chars: int = 700,
) -> None:
    """Fill `cluster.description` with a GPT-generated synthesis (in place).

    One call per cluster, over the abstracts of its articles.
    """
    client = OpenAI(api_key=api_key)
    for cluster in clusters:
        prompt = _cluster_prompt(cluster, max_articles, abstract_chars)
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=400,
        )
        cluster.description = (resp.output_text or "").strip() or None
