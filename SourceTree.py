"""
Citation Tree Explorer — Backend
Pluggable TreeSource interface: swap in any data source (citations, npm deps, etc.)
API: Semantic Scholar (free, no key required)
"""

import re
import time
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Rate limiting ──────────────────────────────────────────────────────────
LAST_REQUEST: dict[str, float] = {}
MIN_INTERVAL = 0.4


def _throttle(source_name: str):
    now = time.time()
    gap = now - LAST_REQUEST.get(source_name, 0)
    if gap < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - gap)
    LAST_REQUEST[source_name] = time.time()


# ── Data model ─────────────────────────────────────────────────────────────

@dataclass
class Paper:
    doi: str
    title: str = ""
    authors: list = field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    citation_count: int = 0
    reference_count: int = 0
    url: str = ""
    source: str = ""

    def to_dict(self):
        return asdict(self)


# ── Abstract TreeSource interface ──────────────────────────────────────────

class TreeSource(ABC):
    """
    Subclass this to add a new domain (npm deps, law citations, etc.).
    Only resolve() and get_references() are required.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def resolve(self, identifier: str) -> Optional[Paper]: ...

    @abstractmethod
    def get_references(self, paper: Paper) -> list[Paper]: ...


# ── Semantic Scholar ───────────────────────────────────────────────────────

SS_BASE = "https://api.semanticscholar.org/graph/v1"
SS_PAPER_FIELDS = "title,authors,year,abstract,citationCount,referenceCount,externalIds,url"
SS_REF_FIELDS   = "title,authors,year,externalIds,citationCount,referenceCount,url"
SS_HEADERS = {
    "User-Agent": "CitationTreeExplorer/1.0 (academic research tool)",
    "Accept": "application/json",
}


def _ss_paper_from_json(d: dict) -> Optional[Paper]:
    if not d:
        return None
    ext = d.get("externalIds") or {}
    doi = (ext.get("DOI") or "").lower()
    if not doi:
        doi = (d.get("paperId") or "").lower()
    if not doi:
        return None
    return Paper(
        doi=doi,
        title=d.get("title") or "",
        authors=[a.get("name", "") for a in (d.get("authors") or [])[:6]],
        year=d.get("year"),
        abstract=(d.get("abstract") or "")[:500],
        citation_count=d.get("citationCount") or 0,
        reference_count=d.get("referenceCount") or 0,
        url=d.get("url") or (f"https://doi.org/{doi}" if doi.startswith("10.") else ""),
        source="semantic_scholar",
    )


class SemanticScholarSource(TreeSource):
    name = "semantic_scholar"

    def _get(self, path: str, params: dict = None) -> Optional[dict]:
        _throttle(self.name)
        try:
            r = requests.get(SS_BASE + path, params=params,
                             headers=SS_HEADERS, timeout=12)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                log.warning("SS rate limited — sleeping 3s")
                time.sleep(3)
            else:
                log.warning("SS %s -> %s", path, r.status_code)
        except Exception as e:
            log.error("SS error: %s", e)
        return None

    def resolve(self, identifier: str) -> Optional[Paper]:
        for prefix in [f"/paper/DOI:{identifier}", f"/paper/{identifier}"]:
            data = self._get(prefix, {"fields": SS_PAPER_FIELDS})
            p = _ss_paper_from_json(data or {})
            if p:
                return p
        # Fallback: keyword search
        data = self._get("/paper/search", {
            "query": identifier, "fields": SS_PAPER_FIELDS, "limit": 1
        })
        for item in (data or {}).get("data", []):
            p = _ss_paper_from_json(item)
            if p:
                return p
        return None

    def get_references(self, paper: Paper) -> list[Paper]:
        doi = paper.doi
        for prefix in [f"DOI:{doi}", doi]:
            data = self._get(f"/paper/{prefix}/references",
                             {"fields": SS_REF_FIELDS, "limit": 100})
            if data and data.get("data"):
                results = []
                for ref in data["data"]:
                    cited = ref.get("citedPaper") or {}
                    p = _ss_paper_from_json(cited)
                    if p:
                        results.append(p)
                return results
        return []


# ── Tree builder ───────────────────────────────────────────────────────────

def _clean_doi(raw: str) -> str:
    raw = raw.strip()
    m = re.search(r"10\.\d{4,}[^\s\"'<>]+", raw)
    return m.group(0).rstrip(".,)") if m else raw


def build_tree(identifier: str, source: TreeSource,
               max_depth: int = 2, max_refs_per_node: int = 15) -> dict:
    doi = _clean_doi(identifier)
    root_paper = source.resolve(doi)
    if not root_paper:
        return {"error": f"Could not resolve '{identifier}'. Try a bare DOI like 10.1038/s41586-020-2649-2"}

    freq: dict[str, int] = defaultdict(int)
    registry: dict[str, Paper] = {}
    visited: set[str] = set()
    min_depths: dict[str, int] = {} 

    def recurse(paper: Paper, depth: int) -> dict:
        freq[paper.doi] += 1
        registry[paper.doi] = paper
        
        if paper.doi not in min_depths or depth < min_depths[paper.doi]:
            min_depths[paper.doi] = depth

        node = {
            "id": paper.doi,
            "paper": paper.to_dict(),
            "children": [],
            "depth": depth,
        }
        if depth >= max_depth or paper.doi in visited:
            return node
        visited.add(paper.doi)
        for ref in source.get_references(paper)[:max_refs_per_node]:
            node["children"].append(recurse(ref, depth + 1))
        return node

    tree = recurse(root_paper, 0)

    max_freq = max(freq.values(), default=1)
    max_cit  = max((p.citation_count for p in registry.values()), default=1) or 1

    ranked = []
    DECAY_FACTOR = 0.9  # 1.0 = no decay, 0.9 = 10% penalty per depth level

    for doi_key, paper in registry.items():
        # Basic Scores (0.0 to 1.0)
        f_score = freq[doi_key] / max_freq
        c_score = min(paper.citation_count, max_cit) / max_cit
        
        # Calculate Depth Penalty
        # Root (depth 0) = 0.9^0 = 1.0 (No penalty)
        # Depth 1 = 0.9^1 = 0.9
        # Depth 2 = 0.9^2 = 0.81
        depth = min_depths.get(doi_key, 0)
        penalty = DECAY_FACTOR ** depth

        # Combine with weights
        base_score = (f_score * 0.5) + (c_score * 0.5)
        
        # Apply penalty and add identity bonus for the root
        final_score = (base_score * penalty)

        ranked.append({
            **paper.to_dict(),
            "frequency": freq[doi_key],
            "importance_score": round(final_score, 4),
            "found_at_depth": depth # Useful for debugging in the UI
        })
    ranked.sort(key=lambda x: x["importance_score"], reverse=True)

    return {
        "tree": tree,
        "ranked": ranked,
        "stats": {
            "total_unique": len(registry),
            "max_depth": max_depth,
            "root_doi": root_paper.doi,
            "root_title": root_paper.title,
        },
    }


SOURCE = SemanticScholarSource()


# ── Flask routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/resolve", methods=["POST"])
def api_resolve():
    body = request.get_json(force=True)
    identifier = (body.get("identifier") or "").strip()
    if not identifier:
        return jsonify({"error": "No identifier provided"}), 400
    paper = SOURCE.resolve(_clean_doi(identifier))
    if not paper:
        return jsonify({"error": f"Could not find: {identifier}"}), 404
    return jsonify(paper.to_dict())


@app.route("/api/tree", methods=["POST"])
def api_tree():
    body       = request.get_json(force=True)
    identifier = (body.get("identifier") or "").strip()
    max_depth  = min(int(body.get("max_depth", 2)), 4)
    max_refs   = min(int(body.get("max_refs_per_node", 15)), 30)
    if not identifier:
        return jsonify({"error": "No identifier provided"}), 400
    result = build_tree(identifier, SOURCE, max_depth=max_depth, max_refs_per_node=max_refs)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
