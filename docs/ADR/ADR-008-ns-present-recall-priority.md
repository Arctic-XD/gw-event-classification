# ADR-008 — NS-Present Recall Priority

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-008 |
| **Title** | Evaluation Priority: NS-Present Recall |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

In binary classification between BBH and NS-present events, we must decide how to balance different types of errors:

- **False Positive (FP)**: Predict NS-present when actually BBH
  - Consequence: Unnecessary electromagnetic follow-up observations
  
- **False Negative (FN)**: Predict BBH when actually NS-present
  - Consequence: Miss opportunity for multi-messenger observation

Multi-messenger astronomy—combining gravitational waves with electromagnetic (EM) observations—is one of the most exciting frontiers in astrophysics. The first BNS detection (GW170817) was accompanied by observations across the entire EM spectrum, from gamma rays to radio waves. This "kilonova" produced transformative science including:

- First direct proof that BNS mergers produce heavy elements (gold, platinum)
- Independent measurement of the Hubble constant
- Constraints on neutron star equation of state
- Confirmation of GW-EM association

**The time-critical nature of EM follow-up is crucial**: electromagnetic counterparts fade rapidly (hours to days). A delayed classification that misses the NS-present nature of an event means losing unique scientific opportunities forever.

---

## 2. Problem Statement

Standard ML evaluation treats all errors equally. However, for GW classification supporting multi-messenger astronomy:
- Missing an NS-present event has higher scientific cost than false alarms
- How should we weight different error types in model evaluation?
- What metrics best capture operational value of the classifier?

**Key Question**: Should we prioritize NS-present recall over overall accuracy, and if so, how do we operationalize this priority?

---

## 3. Decision Drivers

1. **Scientific Asymmetry**: Missing an NS-present event is "irreversible"—the science is lost. A false alarm just triggers unproductive observations.

2. **Class Imbalance**: ~85 BBH vs ~5 NS-present in O1-O3. Standard accuracy would be dominated by BBH performance.

3. **Operational Context**: Rapid classification is meant to trigger EM follow-up. False negatives defeat this purpose.

4. **Real-World Precedent**: LVK alert systems already prioritize "HasNS" probability for alert dissemination.

5. **Resource Constraints**: Telescope time is limited but not prohibitively expensive. Some false alarms are acceptable.

6. **Evaluation Standards**: Science fair judges will want to see domain-appropriate metrics, not just accuracy.

---

## 4. Considered Options

### Option A: Prioritize NS-Present Recall

**Description**: Optimize for high NS-present recall (sensitivity), accepting some increase in false positives.

**Target Metrics**:
- NS-present recall ≥85%
- Precision ≥70% (acceptable false alarm rate)
- Report recall prominently alongside accuracy

**Pros**:
- Directly addresses operational goal (don't miss NS events)
- Aligns with multi-messenger science needs
- Appropriate for asymmetric error costs
- Consistent with LVK alert philosophy

**Cons**:
- May trigger more unnecessary follow-ups
- Overall accuracy may decrease
- Requires threshold tuning

### Option B: Balanced F1 Score Optimization

**Description**: Optimize for F1 score (harmonic mean of precision and recall), treating both error types equally.

**Pros**:
- Standard, well-understood metric
- Balanced approach
- Easy to compare with other work

**Cons**:
- Treats FP and FN equally (not appropriate here)
- May under-weight rare class performance
- Doesn't reflect operational costs

### Option C: Optimize Overall Accuracy

**Description**: Maximize correct classifications regardless of class.

**Pros**:
- Simple, intuitive metric
- Easy to communicate

**Cons**:
- Dominated by majority class (BBH)
- Could achieve 95% by predicting all BBH
- Completely inappropriate for imbalanced data
- Ignores asymmetric error costs

### Option D: Custom Cost-Sensitive Optimization

**Description**: Define explicit cost matrix reflecting real-world error costs, optimize expected cost.

**Pros**:
- Most principled approach
- Directly encodes domain knowledge
- Theoretically optimal

**Cons**:
- Difficult to quantify actual costs
- More complex to implement
- Harder to communicate

---

## 5. Decision Outcome

**Chosen Option**: Option A — Prioritize NS-Present Recall

**Rationale**:

The scientific context makes the error cost asymmetry clear:

1. **Irreversibility of Missed NS Events**: 
   - GW170817-like events may occur once per year or less
   - Each missed event loses unique multi-messenger data
   - EM counterparts fade within days—no second chances
   - Scientific discovery is permanently lost

2. **Tolerability of False Alarms**:
   - Telescope follow-up costs time but not irreversible science
   - O(10) unnecessary pointings per true event is acceptable
   - Systems already handle high alert rates
   - Better to investigate and find nothing than miss something

3. **Class Imbalance Reality**:
   - With 5 NS-present events in training, even 1 missed = 20% recall loss
   - Accuracy of 94% is achievable by always predicting BBH
   - NS-present recall is the meaningful measure of success

4. **Operational Alignment**:
   - LVK already publishes "HasNS" probability in public alerts
   - Astronomers make follow-up decisions based on NS probability
   - Our classifier should optimize for the same priority

**Threshold Tuning Strategy**:
Instead of default 0.5 probability threshold, lower threshold to favor NS-present predictions:
- `P(NS-present) > 0.3` → predict NS-present
- This increases recall at cost of precision

---

## 6. Consequences

### 6.1 Positive Consequences

- **Mission-appropriate optimization**: Classifier serves its intended purpose
- **Scientific relevance**: Evaluation matches real operational needs
- **Domain credibility**: Shows understanding of multi-messenger context
- **Practical utility**: Results translate to operational value
- **Rare class focus**: Ensures NS-present events aren't ignored

### 6.2 Negative Consequences

- **Lower precision**: More BBH events incorrectly flagged as NS-present
- **Accuracy may suffer**: Overall accuracy not optimal
- **Threshold complexity**: Must tune classification threshold
- **Synthetic data pressure**: Must ensure synthetic NS-present events are representative

### 6.3 Neutral Consequences

- Must report multiple metrics (recall, precision, accuracy, F1)
- Confusion matrix becomes essential visualization
- Trade-off curves (precision-recall) become important

---

## 7. Validation

**Success Criteria**:
- NS-present recall ≥85% on O1-O3 cross-validation
- NS-present recall ≥75% on O4a test set
- Precision ≥60% (not too many false alarms)
- Clear documentation of threshold choice and rationale

**Review Date**: Week 4 (Phase 1) and Week 8 (O4a evaluation)

**Reversal Trigger**:
- Precision drops below 50% (too many false alarms)
- Domain expert advises different priority
- O4a reveals systematic NS-present failures

---

## 8. Implementation Notes

### Primary Metrics to Report

| Metric | Definition | Target | Priority |
|--------|------------|--------|----------|
| NS-present Recall | TP / (TP + FN) | ≥85% | **Primary** |
| Precision | TP / (TP + FP) | ≥70% | Secondary |
| Accuracy | (TP + TN) / Total | ≥85% | Contextual |
| F1 Score | 2×P×R / (P+R) | ≥75% | Secondary |
| AUC-ROC | Area under ROC | ≥0.90 | Contextual |

Where: TP = true NS-present, TN = true BBH, FP = false NS-present (actually BBH), FN = false BBH (actually NS-present)

### Threshold Tuning Code
```python
from sklearn.metrics import precision_recall_curve
import numpy as np

def find_optimal_threshold(y_true, y_prob, min_recall=0.85):
    """
    Find threshold that achieves minimum recall with best precision.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Find all thresholds achieving minimum recall
    valid_mask = recalls[:-1] >= min_recall
    if not valid_mask.any():
        print(f"Warning: Cannot achieve {min_recall} recall")
        return 0.3  # Fallback
    
    # Among valid, choose highest precision
    valid_indices = np.where(valid_mask)[0]
    best_idx = valid_indices[np.argmax(precisions[:-1][valid_mask])]
    
    return thresholds[best_idx]
```

### Training with Class Weights
```python
from sklearn.ensemble import RandomForestClassifier

# Compute class weights inversely proportional to frequency
# With 85 BBH and 5 NS-present: weights ≈ {BBH: 1, NS: 17}
clf = RandomForestClassifier(
    class_weight='balanced',  # Automatically computes inverse frequency
    n_estimators=100
)
```

### Evaluation Report Template
```python
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_classifier(y_true, y_pred, y_prob):
    """Generate comprehensive evaluation report."""
    
    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, 
                                target_names=['BBH', 'NS-present']))
    
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred)
    print(f"              Pred BBH  Pred NS")
    print(f"Actual BBH    {cm[0,0]:7d}  {cm[0,1]:7d}")
    print(f"Actual NS     {cm[1,0]:7d}  {cm[1,1]:7d}")
    
    # Primary metric
    ns_recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    print(f"\n*** NS-Present Recall: {ns_recall:.1%} ***")
```

### Visualization: Precision-Recall Curve
```python
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

def plot_pr_curve(y_true, y_prob, save_path=None):
    """Plot precision-recall curve highlighting operating point."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, 
             label=f'PR Curve (AUC = {pr_auc:.3f})')
    
    # Mark minimum recall target
    plt.axvline(x=0.85, color='r', linestyle='--', 
                label='Target Recall (85%)')
    
    plt.xlabel('Recall (NS-present)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Trade-off', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 9. References

- [GW170817 Multi-Messenger](https://arxiv.org/abs/1710.05833): Science from EM follow-up
- [LVK Alert System](https://emfollow.docs.ligo.org/): Public alert infrastructure
- [Imbalanced Classification](https://imbalanced-learn.org/): Handling class imbalance in ML
- [Precision-Recall Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html): sklearn documentation
- [Cost-Sensitive Learning](https://arxiv.org/abs/2010.01167): Theory and practice

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
