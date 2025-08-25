
from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

DATA_PATH = "mushrooms.csv"
data = pd.read_csv(DATA_PATH)

SELECTED_FEATURES = ["odor", "cap-color", "gill-color", "spore-print-color"]
X = data[SELECTED_FEATURES]
y = data["class"]

def entropy(y: pd.Series) -> float:
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))

def information_gain(X: pd.DataFrame, y: pd.Series, feature: str) -> float:
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_entropy = 0.0
    total = counts.sum()
    for v, c in zip(values, counts):
        sub_y = y[X[feature] == v]
        weighted_entropy += (c / total) * entropy(sub_y)
    return entropy(y) - weighted_entropy

def id3(X: pd.DataFrame, y: pd.Series, features: list[str]) -> Any:
    if len(np.unique(y)) == 1:
        return y.iloc[0]
    if len(features) == 0:
        return y.mode()[0]
    gains = [information_gain(X, y, f) for f in features]
    best_feature = features[int(np.argmax(gains))]
    tree: Dict[str, Dict[str, Any]] = {best_feature: {}}
    for v in np.unique(X[best_feature]):
        sub_X = X[X[best_feature] == v]
        sub_y = y[X[best_feature] == v]
        remaining = [f for f in features if f != best_feature]
        if len(sub_y) == 0:
            tree[best_feature][v] = y.mode()[0]
        else:
            tree[best_feature][v] = id3(sub_X, sub_y, remaining)
    return tree

def predict_tree(tree: Any, sample: Dict[str, str]) -> str:
    if not isinstance(tree, dict):
        return str(tree)
    feature = next(iter(tree))
    value = sample.get(feature)
    if value is None:
        return "unknown"
    if value not in tree[feature]:
        return "unknown"
    return predict_tree(tree[feature][value], sample)

COLOR_MAP = {
    "k": "đen (black)","n": "nâu (brown)","b": "vàng nhạt (buff)","c": "quế (cinnamon)",
    "g": "xám (gray)","r": "xanh lục (green)","p": "hồng (pink)","o": "cam (orange)",
    "u": "tím (purple)","e": "đỏ (red)","w": "trắng (white)","y": "vàng (yellow)","l":"xanh dương (blue)"
}
ODOR_MAP = {
    "a":"mùi hạnh nhân (almond)","l":"mùi hồi (anise)","c":"mùi creosote",
    "y":"mùi tanh cá (fishy)","f":"mùi hôi (foul)","m":"mốc (musty)",
    "n":"không mùi (none)","p":"hắc (pungent)","s":"cay nồng (spicy)"
}
FEATURE_NAME_VI = {
    "odor":"Mùi nấm",
    "cap-color":"Màu mũ nấm",
    "gill-color":"Màu phiến nấm",
    "spore-print-color":"Màu bào tử (spore print)"
}

def to_vi_options(feature: str, codes: List[str]) -> List[Tuple[str, str]]:
    opts = []
    for code in sorted(set(map(str, codes))):
        if feature == "odor":
            label = ODOR_MAP.get(code, code)
        else:
            label = COLOR_MAP.get(code, code)
        opts.append((code, label))
    return opts

FEATURE_OPTIONS: Dict[str, list[tuple[str, str]]] = {
    f: to_vi_options(f, list(X[f].unique())) for f in SELECTED_FEATURES
}

TREE = id3(X, y, SELECTED_FEATURES)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", feature_options=FEATURE_OPTIONS, feature_name_vi=FEATURE_NAME_VI)

@app.post("/predict")
def predict():
    data_in = request.json or {}
    sample = {col: str(data_in.get(col)) if data_in.get(col) else None for col in SELECTED_FEATURES}
    label = predict_tree(TREE, sample)
    return jsonify({"label": label})

@app.get("/tree")
def tree():
    return jsonify(TREE)

if __name__ == "__main__":
    print("✅ Flask app chạy tại: http://127.0.0.1:5000")
    app.run(debug=True)
