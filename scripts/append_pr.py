import json
from pathlib import Path

nb_path = Path('notebooks/02_Model_Selection.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Combined Precision-Recall Curve\n",
        "Here we plot the Precision-Recall curves for all evaluated models on a single graph to directly compare their ability to capture failures (Recall) against their false alarm rate (Precision)."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "plt.figure(figsize=(10, 8))\n",
        "\n",
        "for name, results in model_results.items():\n",
        "    y_prob = results['y_prob']\n",
        "    pr_vals, rc_vals, _ = precision_recall_curve(y_test, y_prob)\n",
        "    pr_auc = auc(rc_vals, pr_vals)\n",
        "    plt.plot(rc_vals, pr_vals, lw=2, label=f'{name} (AUC = {pr_auc:.3f})')\n",
        "\n",
        "plt.xlabel('Recall')\n",
        "plt.ylabel('Precision')\n",
        "plt.title('Combined Precision-Recall Curve Comparison')\n",
        "plt.legend(loc='lower left')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
}

nb['cells'].extend([md_cell, code_cell])

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Appended cells to the notebook.")
