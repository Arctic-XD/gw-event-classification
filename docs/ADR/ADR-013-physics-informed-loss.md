# ADR-013 — Physics-Informed Loss Function

## Metadata

| Field | Value |
|-------|-------|
| **ADR ID** | ADR-013 |
| **Title** | Physics-Informed Loss Function for Chirp Mass Estimation |
| **Status** | Accepted |
| **Created** | 2024-12-25 |
| **Last Updated** | 2024-12-25 |
| **Author(s)** | Project Team |
| **Supersedes** | N/A |
| **Superseded By** | N/A |

---

## 1. Context

Gravitational-wave chirp signals follow precise physical relationships. The instantaneous frequency evolution during inspiral is governed by:

$$f(t) = \frac{1}{\pi} \left( \frac{5}{256} \frac{1}{\tau} \right)^{3/8} \left( \frac{G \mathcal{M}_c}{c^3} \right)^{-5/8}$$

Where:
- $f(t)$ is the instantaneous GW frequency
- $\tau = t_c - t$ is the time to coalescence
- $\mathcal{M}_c = \frac{(m_1 m_2)^{3/5}}{(m_1 + m_2)^{1/5}}$ is the **chirp mass**
- $G$ is the gravitational constant, $c$ is the speed of light

The key relationship is: **chirp rate is determined by chirp mass**

$$\frac{df}{dt} \propto f^{11/3} \mathcal{M}_c^{5/3}$$

This means:
- BBH (high $\mathcal{M}_c$): Faster chirp, lower frequencies, shorter duration
- BNS (low $\mathcal{M}_c$): Slower chirp, higher frequencies, longer duration

Standard CNN classification treats this as a black-box pattern recognition problem. **Physics-informed** approaches incorporate known physical relationships into the learning process.

---

## 2. Problem Statement

Pure data-driven CNNs may:
1. Learn spurious correlations (noise artifacts, not physics)
2. Fail on out-of-distribution examples
3. Provide no physical interpretability
4. Ignore known physics that could improve generalization

**Key Question**: Can we incorporate the chirp mass physics relationship into the CNN training process to improve performance and interpretability?

---

## 3. Decision Drivers

1. **Scientific Novelty**: Physics-informed ML for GW classification is a novel contribution

2. **Interpretability**: Chirp mass estimate provides physical insight into classification

3. **Generalization**: Physics constraints may improve out-of-distribution performance

4. **Reference Project Alignment**: The reference CW project used PINNs successfully

5. **Regularization**: Physics loss acts as regularizer, potentially reducing overfitting

6. **Verification**: Can validate chirp mass estimates against known values

---

## 4. Considered Options

### Option A: Physics-Informed Multi-Task Loss

**Description**: CNN outputs both classification probabilities AND chirp mass estimate. Total loss combines classification loss with physics consistency loss.

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{physics}$$

Where:
- $\mathcal{L}_{CE}$: Cross-entropy classification loss
- $\mathcal{L}_{physics}$: Physics consistency loss
- $\lambda$: Weighting hyperparameter

**Pros**:
- Directly incorporates physics
- Produces interpretable chirp mass output
- Can validate against known $\mathcal{M}_c$ for real events
- Novel contribution

**Cons**:
- More complex architecture
- Requires tuning $\lambda$
- Physics loss formulation non-trivial

### Option B: Physics-Based Feature Engineering Only

**Description**: Extract physics-motivated features (chirp slope → chirp mass estimate) and use as ML input. No physics in loss function.

**Pros**:
- Simpler implementation
- Already doing this in Phase 1

**Cons**:
- Physics not in learning process
- No end-to-end physics integration
- Less novel

### Option C: Pure Data-Driven CNN

**Description**: Standard CNN with only classification loss. No physics constraints.

**Pros**:
- Simplest implementation
- Standard approach

**Cons**:
- May learn spurious patterns
- No interpretable physics output
- Less novel
- May not generalize as well

### Option D: Full PINN with Differential Equations

**Description**: Encode full chirp evolution ODE in loss function.

**Pros**:
- Maximum physics integration

**Cons**:
- Very complex implementation
- May be overkill for classification
- Difficult to train
- Beyond project scope

---

## 5. Decision Outcome

**Chosen Option**: Option A — Physics-Informed Multi-Task Loss

**Architecture**:
```
Input Spectrogram (224×224)
        ↓
    ResNet Backbone
        ↓
    Feature Vector (512-d)
        ↓
   ┌────┴────┐
   ↓         ↓
Classification   Chirp Mass
  Head           Head
   ↓             ↓
P(BBH),        M_c estimate
P(NS-present)
```

**Loss Function**:

$$\mathcal{L}_{total} = \mathcal{L}_{CE}(y, \hat{y}) + \lambda_1 \mathcal{L}_{Mc}(\mathcal{M}_c, \hat{\mathcal{M}}_c) + \lambda_2 \mathcal{L}_{consistency}$$

Where:
- $\mathcal{L}_{CE}$: Cross-entropy for classification
- $\mathcal{L}_{Mc}$: MSE between predicted and true chirp mass (for synthetic data with known $\mathcal{M}_c$)
- $\mathcal{L}_{consistency}$: Physics consistency constraint

**Physics Consistency Loss**:

The chirp mass determines the peak frequency before merger:

$$f_{peak} \approx 4400 \text{ Hz} \times \left( \frac{2.8 M_\odot}{\mathcal{M}_c} \right)^{5/8}$$

So if we extract peak frequency $\hat{f}_{peak}$ from the spectrogram and predict chirp mass $\hat{\mathcal{M}}_c$:

$$\mathcal{L}_{consistency} = \left| \hat{f}_{peak} - 4400 \times \left( \frac{2.8}{\hat{\mathcal{M}}_c} \right)^{5/8} \right|^2$$

This penalizes predictions where the chirp mass is inconsistent with the observed peak frequency.

---

## 6. Consequences

### 6.1 Positive Consequences

- **Novel contribution**: First physics-informed rapid CBC classifier
- **Interpretability**: Chirp mass output explains classification
- **Validation**: Can verify $\mathcal{M}_c$ against GWTC values
- **Regularization**: Physics constraints reduce overfitting
- **Generalization**: May improve O4a transfer
- **Judge appeal**: "Physics-informed" is compelling for science fair

### 6.2 Negative Consequences

- **Complexity**: More complex than pure classification
- **Hyperparameter tuning**: $\lambda_1$, $\lambda_2$ need tuning
- **Training difficulty**: Multi-task learning can be finicky
- **Computational overhead**: Extra forward pass for $\mathcal{M}_c$ head

### 6.3 Neutral Consequences

- Requires synthetic data with known $\mathcal{M}_c$ (already planned)
- Can ablate: compare with/without physics loss
- May not improve accuracy but provides interpretability

---

## 7. Validation

**Success Criteria**:
- Physics-informed model achieves ≥ pure CNN accuracy
- Chirp mass estimates correlate with true values (R² > 0.7)
- Predicted $\mathcal{M}_c$ falls in expected range per class:
  - BBH: 10-50 M☉ typically
  - NS-present: 1-5 M☉ typically
- Ablation shows physics loss contributes (not just overhead)

**Review Date**: Week 7 (PINN implementation)

**Reversal Trigger**:
- Physics loss significantly hurts classification accuracy (>5% drop)
- Chirp mass estimates are uncorrelated with true values
- Training becomes unstable

---

## 8. Implementation Notes

### 8.1 Network Architecture

```python
import torch
import torch.nn as nn
import timm

class GWClassifierWithPhysics(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=2):
        super().__init__()
        
        # Backbone (pretrained)
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        feature_dim = self.backbone.num_features  # e.g., 512 for ResNet18
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Chirp mass regression head
        self.chirp_mass_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensure positive chirp mass
        )
        
    def forward(self, x):
        features = self.backbone(x)
        
        class_logits = self.classifier(features)
        chirp_mass = self.chirp_mass_head(features)
        
        return class_logits, chirp_mass
```

### 8.2 Physics-Informed Loss

```python
class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_mc=0.1, lambda_consistency=0.05):
        super().__init__()
        self.lambda_mc = lambda_mc
        self.lambda_consistency = lambda_consistency
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, class_logits, chirp_mass_pred, 
                class_labels, chirp_mass_true, peak_freq_observed):
        """
        Parameters
        ----------
        class_logits : (batch, 2) classification logits
        chirp_mass_pred : (batch, 1) predicted chirp mass in solar masses
        class_labels : (batch,) true class labels
        chirp_mass_true : (batch,) true chirp mass (from injection params)
        peak_freq_observed : (batch,) peak frequency from spectrogram
        """
        
        # Classification loss
        L_ce = self.ce_loss(class_logits, class_labels)
        
        # Chirp mass regression loss (only for samples with known Mc)
        valid_mc = chirp_mass_true > 0  # Mask for samples with known Mc
        if valid_mc.any():
            L_mc = self.mse_loss(
                chirp_mass_pred[valid_mc], 
                chirp_mass_true[valid_mc].unsqueeze(1)
            )
        else:
            L_mc = 0.0
        
        # Physics consistency loss
        # f_peak ≈ 4400 * (2.8 / Mc)^(5/8)  [Hz]
        f_peak_predicted = 4400 * (2.8 / (chirp_mass_pred + 1e-6))**0.625
        L_consistency = self.mse_loss(f_peak_predicted.squeeze(), peak_freq_observed)
        
        # Total loss
        L_total = L_ce + self.lambda_mc * L_mc + self.lambda_consistency * L_consistency
        
        return L_total, {
            'L_ce': L_ce.item(),
            'L_mc': L_mc.item() if isinstance(L_mc, torch.Tensor) else L_mc,
            'L_consistency': L_consistency.item()
        }
```

### 8.3 Training with Physics Loss

```python
def train_epoch_physics(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        spectrograms = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)
        chirp_mass = batch['chirp_mass'].to(device)
        peak_freq = batch['peak_frequency'].to(device)
        
        optimizer.zero_grad()
        
        class_logits, mc_pred = model(spectrograms)
        
        loss, loss_components = loss_fn(
            class_logits, mc_pred, 
            labels, chirp_mass, peak_freq
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### 8.4 Chirp Mass Validation

```python
def validate_chirp_mass(model, dataloader, device):
    """Compare predicted vs true chirp masses."""
    model.eval()
    
    mc_true_list = []
    mc_pred_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            _, mc_pred = model(batch['spectrogram'].to(device))
            mc_true_list.extend(batch['chirp_mass'].numpy())
            mc_pred_list.extend(mc_pred.cpu().numpy().flatten())
    
    from scipy.stats import pearsonr
    r, p = pearsonr(mc_true_list, mc_pred_list)
    
    print(f"Chirp Mass Correlation: r={r:.3f}, p={p:.2e}")
    
    return r, mc_true_list, mc_pred_list
```

### 8.5 Hyperparameter Recommendations

| Parameter | Recommended Value | Range to Explore |
|-----------|-------------------|------------------|
| $\lambda_{mc}$ | 0.1 | 0.01 - 1.0 |
| $\lambda_{consistency}$ | 0.05 | 0.01 - 0.5 |
| Chirp mass head depth | 2 layers | 1-3 layers |
| Hidden units | 128 | 64-256 |

---

## 9. References

- [Physics-Informed Neural Networks](https://arxiv.org/abs/1711.10561): Original PINN paper
- [Chirp Mass Formula](https://arxiv.org/abs/gr-qc/9402014): GW frequency evolution
- [Multi-Task Learning](https://arxiv.org/abs/1706.05098): Theory and practice
- [GW Parameter Estimation](https://arxiv.org/abs/1409.7215): Standard PE methods
- [PINN for Physical Systems](https://arxiv.org/abs/2001.04536): Applications review

---

## 10. Revision History

| Date | Author | Description |
|------|--------|-------------|
| 2024-12-25 | Project Team | Initial ADR creation |
