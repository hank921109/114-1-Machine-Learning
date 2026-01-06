# Support Vector Machine (SVM) Implementation

## 1. Introduction
本專案展示了線性支持向量機 (Linear SVM) 的完整數學推導與實作。目標是找到一個超平面 (Hyperplane)，在將兩類數據分開的同時，最大化兩者之間的邊界 (Margin)。

## 2. Mathematical Derivation (數學推導)

### 2.1 The Primal Problem (原始問題)
目標是最小化權重向量的長度（最大化 Margin），同時滿足分類約束：

$$\min_{w, b} \frac{1}{2} ||w||^2$$
$$\text{subject to } y_i(w^T x_i + b) \geq 1, \quad \forall i$$

### 2.2 The Dual Problem (對偶問題)
透過引入拉格朗日乘數 (Lagrange Multipliers) $\alpha_i \geq 0$，我們將問題轉化為對偶形式。

拉格朗日函數：
$$L(w, b, \alpha) = \frac{1}{2}||w||^2 - \sum_{i=1}^{n} \alpha_i [y_i(w^T x_i + b) - 1]$$

對 $w$ 和 $b$ 微分並令其為 0，得到 KKT 條件：
1. $\nabla_w L = w - \sum \alpha_i y_i x_i = 0 \implies w = \sum_{i=1}^{n} \alpha_i y_i x_i$
2. $\nabla_b L = - \sum \alpha_i y_i = 0 \implies \sum_{i=1}^{n} \alpha_i y_i = 0$

將上述條件代回原式，得到對偶問題（只需求解 $\alpha$）：
$$\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j (x_i^T x_j)$$

### 2.3 How to Find Optimal Parameters (求解最佳參數 w* 和 b*)
這一步是實作的核心。當我們透過二次規劃 (Quadratic Programming) 演算法解出最佳的 $\alpha_i$ 後，我們可以反推最佳的權重 $w^{\ast}$ 和偏差 $b^{\ast}$。

**1. 求解權重向量 $w^{\ast}$**

根據 KKT 條件，最佳權重 $w^{\ast}$ 是支持向量的線性組合：

$$w^{\ast} = \sum_{i=1}^{n} \alpha_i y_i x_i$$

*注意：只有支持向量 (Support Vectors) 的 $\alpha_i > 0$，非支持向量的 $\alpha_i = 0$。因此 $w^{\ast}$ 僅由少數的支持向量決定。*

**2. 求解偏差 $b^{\ast}$**

對於任意一個支持向量 $x_k$（即 $\alpha_k > 0$ 的點），它必定位於邊界上，滿足：

$$y_k (w^{\ast T} x_k + b^{\ast}) = 1$$

因為 $y_k^2 = 1$ (標籤為 1 或 -1)，我們可以兩邊同乘 $y_k$ 移項得到：

$$b^{\ast} = y_k - w^{\ast T} x_k$$

在實作中，為了數值穩定性，通常會對所有支持向量計算出的 $b^{\ast}$ 取平均值：

$$b^{\ast} = \frac{1}{|S|} \sum_{k \in S} (y_k - w^{\ast T} x_k)$$

其中 $S$ 是支持向量的集合。

## 3. Project Structure
本程式碼分為四個步驟（詳見 ipynb）：
1.  **Data Generation**: 建立 2D 線性可分數據。
2.  **Finding Support Vectors**: 訓練模型並找出關鍵的支持向量。
3.  **Verification**: **關鍵步驟**，手動撰寫程式驗證 $w^{\ast} = \sum \alpha_i y_i x_i$ 公式是否成立。
4.  **Visualization**: 繪製超平面、邊界與決策區域。
