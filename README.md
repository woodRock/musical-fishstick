# Survey on Ordinal Classification

**Jesse Wood, Bach Nguyen, Bing Xue, Mengjie Zhang, Daniel Killeen**

## Abstract
Ordinal classification, or ordinal regression, is a unique supervised learning task where the target variable's categories possess a natural order. This problem is distinct from nominal classification, which ignores order, and metric regression, which assumes equidistant categories. This review surveys the primary families of methods developed to address this challenge, categorizing them into threshold-based models, binary decomposition approaches, and modern deep learning techniques. We discuss the key models within each family, analyze their respective strengths and limitations, and review the standard evaluation metrics for a rigorous comparison. Finally, we highlight significant open challenges and outline promising directions for future research.

## IEEEImpStatement
This review provides a comprehensive and structured overview of the field of ordinal classification, a critical task in many real-world applications such as medicine, social science, and sentiment analysis. By categorizing the diverse methodologies, from classic statistical models to modern deep learning approaches, this work serves as an essential resource for researchers and practitioners. It lowers the barrier to entry for new researchers, helps experienced practitioners select the most appropriate model for their problem, and identifies key open challenges, thereby guiding future research toward areas of greatest need and potential impact.

## IEEEkeywords
Ordinal Classification, Ordinal Regression, Literature Review, Machine Learning, Threshold Models, Deep Learning, Evaluation Metrics

# Introduction
## Definition of Ordinal Classification

Ordinal classification, also known as ordinal regression, is a type of supervised learning problem where the goal is to assign instances to categories that have a natural, inherent order. Formally, given an input instance described by a feature vector $X \in \mathbb{R}^d$, the task is to predict its corresponding target label $Y$, which belongs to a finite set of $k$ ordered categories, denoted as {$C_1, C_2, ..., C_k$}, such that a meaningful ordering relationship exists: $C_1 \prec C_2 \prec ... \prec C_k$.

This task occupies a unique space between standard classification and regression:

- **Contrast with Nominal Classification:** Unlike nominal classification problems where categories are distinct but unordered (e.g., classifying images as `cat`, `dog`, or `bird`), the order between categories in ordinal classification holds significant meaning (e.g., `low`, `medium`, `high` risk). Ignoring this order, as nominal classifiers do, results in a loss of valuable information.
- **Contrast with Metric Regression:** While both ordinal classification and metric regression deal with ordered outcomes, metric regression assumes that the target variable is measured on at least an interval scale, implying that the distances between consecutive values are quantifiable and meaningful (e.g., predicting temperature in Celsius). Ordinal classification relaxes this assumption; the categories have a clear sequence, but the ``distance'' or difference between adjacent categories (e.g., the difference between `mild` and `moderate` disease severity) is not necessarily uniform or precisely defined.

The fundamental challenge in ordinal classification is to develop models that effectively leverage the rank information inherent in the labels, penalizing prediction errors based on their ordinal distance from the true category, rather than treating all misclassifications equally. This conceptual difference is illustrated in Figure~\ref{fig:data_types}.

![A conceptual comparison of data measurement scales. (a) **Nominal** data consists of categories without any inherent order (e.g., fruit types). (b) **Ordinal** data features categories with a meaningful order, but the distances between them are not precisely defined or uniform (e.g., clothing sizes). (c) **Metric** (or Interval/Ratio) data has both a meaningful order and quantifiable, equidistant intervals between values (e.g., numerical measurements).](data_type_contrast.png)

## Importance and Applications

This task is critical in many real-world applications, as summarized in Table~\ref{tab:real_world_examples}. The importance of ordinal classification continues to grow, with recent advancements, particularly driven by deep learning, expanding its application into increasingly complex domains. In medical imaging, state-of-the-art approaches now tackle tasks such as detailed disease severity grading in chest radiographs [Wienholt et al., 2024], predicting biological age from MRI scans while preserving ordinal relationships [Solanky et al., 2024], and quantifying diagnostic certainty levels directly from radiology reports [Fujimoto et al., 2024]. Beyond healthcare, ordinal methods are being applied in environmental science, for instance, using LiDAR data for fine-grained classification of forest vegetation strata [Peña-Alonso et al., 2024], and remain crucial in areas like financial risk assessment (e.g., credit scoring) [Cohen & Singer, 2024]. These modern applications underscore the value of explicitly modeling order for improved accuracy and interpretability.

Table: Examples of Real-World Ordinal Classification Applications

## Scope and Objectives

The primary objective of this review is to provide a comprehensive and structured overview of the field of ordinal classification. We aim to survey the major families of methodologies developed for this task, ranging from traditional statistical models to modern deep learning techniques. This involves:

- Categorizing existing methods into distinct families based on their underlying principles (threshold-based, binary decomposition, deep learning).
- Discussing the key algorithms within each family, explaining their core mechanisms.
- Analyzing the relative strengths, weaknesses, and common assumptions associated with each approach.
- Reviewing standard evaluation metrics specifically designed for ordinal data to enable rigorous comparison.
- Identifying common benchmark datasets used in the literature.

Ultimately, this survey seeks to serve as a valuable resource for both newcomers seeking an introduction to the field and experienced researchers and practitioners looking for a comparative analysis to guide model selection and identify avenues for future research. The scope is focused on supervised learning methods explicitly designed for or adapted to ordinal target variables.

## Structure of the Review

This review is organized as follows: Section II formally defines the ordinal classification problem, establishes notation, and critically examines the evaluation metrics necessary for assessing performance while accounting for label order. Section III discusses common baseline or ``naive'' approaches, such as treating the problem as nominal classification or metric regression, and highlights their inherent limitations. The core of the survey is Section IV, which categorizes and details the three primary families of specialized ordinal classification models: threshold-based methods, binary decomposition strategies, and modern deep learning techniques. Section V provides a comparative analysis, discussing benchmark datasets, comparing model performance, and summarizing the strengths and weaknesses of each family. Finally, Section~\ref{sec:challenges} identifies key open challenges and suggests directions for future research, before Section VII concludes the review with a summary of the key findings.

# Problem Formulation and Evaluation

Having introduced the concept of ordinal classification and its importance, we must now establish a formal understanding of the problem. This next section will define the mathematical notation and, critically, review the specialized evaluation metrics required to properly assess ordinal model performance.

## Formal Notation

Let $X \in \mathbb{R}^d$ represent the input features for an instance, where $d$ is the number of features. The goal is to predict the corresponding ordinal target label $Y$, which belongs to a set of $k$ ordered categories {$C_1, C_2, ..., C_k$}, such that $C_1 \prec C_2 \prec ... \prec C_k$. We can map these categories to integer ranks {$1, 2, ..., k$} for notational convenience, where $y_i$ is the true integer rank for instance $i$ and $\hat{y}_i$ is the predicted integer rank.

## Evaluation Metrics

Standard classification accuracy, which measures the proportion of correctly classified instances (a 0-1 loss), is often insufficient for ordinal problems. It treats all misclassifications equally, ignoring the inherent order. For example, predicting class $C_1$ when the true class is $C_k$ is penalized the same as predicting $C_{k-1}$ when the true class is $C_k$. This fails to capture the severity of errors in an ordinal context, where larger rank differences represent worse predictions. Figure~\ref{fig:evaluation_metrics} illustrates this contrast. Consequently, specialized metrics that account for the ordered nature of the labels are essential.

- **Mean Absolute Error (MAE):** This is arguably the most common and interpretable metric for ordinal tasks. It measures the average absolute difference between the predicted rank ($\hat{y}_i$) and the true rank ($y_i$) across all $N$ instances in the evaluation set:
    $MAE = \frac{1}{N} \sum_{i=1}^N | \hat{y}_i - y_i |$
    MAE directly reflects the average magnitude of prediction errors in terms of rank distance. A lower MAE indicates better performance. It penalizes errors linearly based on their distance from the true rank.

- **Quadratic Weighted Kappa (QWK):** This metric measures the agreement between predicted and true ratings, corrected for chance agreement. It is particularly robust because it uses quadratic weights to penalize disagreements. The weight $w_{ij}$ assigned to a disagreement between true class $i$ and predicted class $j$ is typically calculated as:
    $w_{ij} = \frac{(i-j)^2}{(k-1)^2}$
    This means that larger errors (greater distance between $i$ and $j$) are penalized much more heavily than smaller errors. QWK ranges from -1 (total disagreement) to 1 (perfect agreement), with 0 indicating agreement equivalent to chance. Higher QWK values indicate better performance.

- **Other Metrics:** While MAE and QWK are prevalent, other metrics are sometimes used. **Mean Squared Error (MSE)** ($\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$) can be employed, but like QWK, its quadratic penalty makes it more sensitive to large errors or outliers compared to MAE. Standard **classification accuracy** is occasionally reported for comparison but, as discussed, provides an incomplete picture by ignoring the ordinal structure. Balanced accuracy might also be considered, especially in cases of class imbalance, but still treats errors nominally.

The fundamental difference between nominal and ordinal evaluation, highlighting the varying costs assigned to misclassifications, is visualized in Figure~\ref{fig:evaluation_metrics}.

![Conceptual comparison of evaluation metrics for a 5-class problem. (a) **Standard Accuracy** (Nominal Loss) uses a 0-1 cost, treating all misclassifications as equally incorrect. (b) **Ordinal Metrics** (such as MAE or QWK) use a weighted cost. The penalty for an error is proportional (MAE) or quadratically proportional (QWK) to its distance from the true class, making a large error (e.g., predicting $C_1$ when the true class is $C_5$) far more costly than a small error (e.g., predicting $C_4$ when the true class is $C_5$).](figures/evaluation_metrics_contrast.png)

# Baseline and Naive Approaches

With a clear problem formulation and the correct evaluation metrics established, we first turn our attention to the most straightforward methods for tackling this task. These ``naive'' approaches adapt standard machine learning techniques by either ignoring the ordinal nature of the labels or making simplifying assumptions about them. The following section reviews these common baselines, highlighting their inherent limitations to motivate the need for specialized ordinal techniques.

## Ignoring the Order (Nominal Classification)

A simple baseline is to treat the ordinal classification problem as a standard multiclass nominal classification problem. This involves using algorithms designed for unordered categories, such as:
- Multinomial Logistic Regression (Softmax Regression)
- Support Vector Machines (using one-vs-rest or one-vs-one strategies)
- Decision Trees (e.g., CART, C4.5)
- Random Forests
- Neural Networks with a softmax output layer and cross-entropy loss.

**Limitation:** The primary drawback of this approach is the complete loss of the ordering information present in the labels ($C_1 \prec C_2 \prec ... \prec C_k$). The model learns to discriminate between categories but is not aware that misclassifying $C_1$ as $C_k$ is a more severe error than misclassifying $C_1$ as $C_2$. This often leads to suboptimal performance when evaluated using appropriate ordinal metrics like MAE or QWK.

## Assuming Equidistance (Metric Regression)

Another common naive approach is to map the ordinal categories {$C_1, ..., C_k$} to numerical values, typically integers {$1, ..., k$}, and then treat the problem as a standard metric regression task. Algorithms used include:
- Linear Regression
- Support Vector Regression (SVR)
- Regression Trees
- Gradient Boosting Machines (for regression)
- Neural Networks with a single linear output unit and a regression loss (e.g., MSE, MAE).

The continuous output predicted by the regression model is then typically rounded to the nearest integer (or thresholded) to obtain the final ordinal class prediction.

**Limitation:** This approach makes a strong, often invalid, assumption that the ``distance'' or interval between consecutive categories is equal and meaningful (e.g., the difference between``Mild'' and``Moderate'' disease is quantitatively the same as between ``Moderate'' and ``Severe''). This equidistance assumption rarely holds for true ordinal scales. Furthermore, the final rounding step can introduce biases, especially for predictions near the midpoint between two integer ranks. While this method implicitly uses some order information (by mapping to numbers), the flawed assumption about the scale can limit its effectiveness.

# Families of Ordinal Classification Models

The shortcomings of nominal and regression-based baselines underscore the necessity of models designed specifically for ordinal data. Therefore, the core of this review, presented in the next section, is a comprehensive survey of the main families of ordinal classification models. We will categorize and analyze threshold-based models, binary decomposition methods, and modern deep learning approaches.

This is the core of the review, organized by methodology. We illustrate the taxonomy of the three different families of ordinal classification in Figure~\ref{fig:taxonomy}.

![A taxonomy of ordinal classification methods, providing a visual map for the review. The primary distinction is made between Naive Approaches (which ignore or misuse the ordinal information) and Specialized Ordinal Models. These specialized models are categorized into the three main families: Threshold-Based, Binary Decomposition, and Deep Learning.](figures/taxonomy_flowchart.png)

## Family 1: Threshold-Based Models

The first and most established family of ordinal models is built on the concept of a continuous latent variable. The core idea is that this unobserved variable $z$ is ``sliced'' by $k-1$ ordered thresholds ($\theta_1, \dots, \theta_{k-1}$) to produce the $k$ observed categories. The most prominent example is the **Proportional Odds Model (POM)** [McCullagh, 1980], also known as Ordinal Logistic Regression. This model is itself a specific instance of the broader **Cumulative Link Models (CLMs)** [McCullagh, 1980] family, which generalizes the approach for different link functions (e.g., probit or complementary log-log). This same latent variable concept was later adapted for machine learning, leading to **Support Vector Ordinal Regression (SVOR)** [Herbrich et al., 1999; Chu & Keerthi, 2007], which implements the thresholds as a set of parallel separating hyperplanes in a large-margin framework. A conceptual diagram of this latent variable approach is shown in Figure~\ref{fig:threshold_model}.

![Conceptual diagram of Threshold-Based Ordinal Models (e.g., the Proportional Odds Model). A continuous, unobserved latent variable $z^*$ (here, following a logistic distribution) is partitioned into $k$ discrete ordinal categories ($C_1, \dots, C_k$) by $k-1$ ordered thresholds ($\theta_1, \dots, \theta_{k-1}$). An observed instance falls into category $C_j$ if its latent variable value $z^*$ lies between $\theta_{j-1}$ and $\theta_j$.](figures/threshold_based_ordinal_models.png)

## Family 2: Binary Decomposition Models

A second major family recasts the $k$-class ordinal problem into a series of $k-1$ binary classification sub-problems. This approach is not a single model but a meta-strategy that can use any binary classifier (such as an SVM or decision tree) as its base. The most common, order-preserving strategy is to build $k-1$ cumulative classifiers, each answering the question``is the true class $y > C_j$?'' [Frank & Hall, 2001]. Alternative decomposition schemes also exist, such as comparing **adjacent categories** (e.g., $C_j$ vs. $C_{j+1}$) or using immediate partitions (e.g., $C_1$ vs. {$C_2\dots C_k$}) [Agresti, 2010]. Two of the most common decomposition strategies are illustrated in Figure~\ref{fig:binary_decomposition}.

![Visualization of two common binary decomposition strategies for a $k=4$ class problem. (a) The **Cumulative** strategy [Frank & Hall, 2001] builds $k-1$ classifiers, where each classifier $j$ separates the cumulative set of classes {$C_1, \dots, C_j$} from all classes above them {$C_{j+1}, \dots, C_k$}. (b) The **Adjacent Categories** strategy [Agresti, 2010] also builds $k-1$ classifiers, but each one only compares two neighboring classes, $C_j$ vs. $C_{j+1}$.](figures/binary_decomposition_models.png)

## Family 3: Deep Learning and Modern Approaches

The third family adapts modern deep learning methods to the ordinal setting. This is primarily achieved in two ways. The first is by developing specialized **ordinal-specific loss functions** that teach the network about the ordered label structure. Instead of penalizing all misclassifications equally (like cross-entropy), these losses penalize predictions that are ``further away'' from the true rank. Key examples include **CORAL (Consistent Rank Logits)** [Cao et al., 2020], its successor **CORN (Conditional Ordinal Regression for Neural Networks)** [Shi et al., 2021], and losses based on the **Earth Mover's Distance (EMD)** [De Matos et al., 2019]. The second approach involves modifying the network architecture itself, for example by designing **ordinal output layers** that directly model the cumulative probabilities [Niu et al., 2016], mirroring the structure of classic threshold models. Figure~\ref{fig:deep_learning_arch} contrasts these two popular deep learning strategies.

![Common deep learning strategies for ordinal classification. (a) The **Ordinal Output Layer** approach [Niu et al., 2016] modifies the network's final layer to have $k-1$ sigmoid neurons, each predicting a cumulative probability $P(y > C_j)$, which are then combined. (b) The **Ordinal Loss Function** approach [Cao et al., 2020; Shi et al., 2021] uses a standard $k$-class softmax output layer but replaces the nominal cross-entropy loss with a specialized ordinal loss (e.g., CORAL, EMD) that penalizes predictions based on their rank distance from the true label.](figures/deep_learning_ordinal_models.png)

# Comparative Analysis and Discussion

After systematically reviewing the three primary families of ordinal models, it is essential to synthesize this information and understand their practical trade-offs. The following section provides a comparative analysis, discussing common benchmark datasets and summarizing the relative strengths and weaknesses of each model category to guide practitioner choice.

## Benchmark Datasets
A summary of several common benchmark datasets used in ordinal classification literature is provided in Table~\ref{tab:dataset_properties}.

Table: Summary of Common Ordinal Classification Benchmark Datasets

### Wine Quality

Wine Quality [Cortez et al., 2009] is one of the most popular datasets for ordinal tasks. It consists of two separate datasets (one for red wine, one for white wine) where the goal is to predict the quality of the wine based on physicochemical properties (e.g., fixed acidity, alcohol, pH).

**Ordinal Task:**
The target variable is quality, an ordinal score given by experts, typically ranging from 3 to 8.

### Car Evaluation

The Car Evaluation [Bohanec & Rajkovic, 1990] dataset, derived from a simple hierarchical decision model, is a classic example of a purely categorical dataset where the target variable is inherently ordered.

**Ordinal Task:**
The target variable is class, which represents the acceptability of a car. The classes are ``unacc'' (unacceptable), ``acc'' (acceptable), ``good'', and ``vgood'' (very good).

### Poker Hand

The Poker Hand [Cattral & Oppacher, 2007] dataset is a common benchmark for classification. While often treated as a nominal classification problem, the target variable (the poker hand) has a clear and well-defined ordinal ranking.

**Ordinal Task:**
The target variable is the poker hand, ranked from 0 to 9 (0: Nothing in hand, 1: One pair, 2: Two pairs, ..., 9: Royal flush).

### Housing Datasets (e.g., Boston Housing)

The original Boston Housing [Harrison & Rubinfeld, 1978] dataset is a classic regression problem. However, it is very common in ordinal regression literature for the target variable MEDV (median value of homes) to be binned into a set of ordered categories (e.g., ``low,'' ``medium,'' ``high,'' ``very high'').

Ordinal Task: Predict the binned (discretized) median value of owner-occupied homes.

### Diabetic Retinopathy Detection

A high-stakes medical imaging task, the Diabetic Retinopathy Detection dataset [Kaggle, 2015] is a prominent high-dimensional benchmark. It consists of tens of thousands of high-resolution fundus images of retinas, sourced from a Kaggle competition.

**Ordinal Task:**
Clinicians grade the severity of retinopathy on a 0-4 ordinal scale (0: None, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative). The goal is to train a model to replicate this ordinal assessment, where misclassifying a ``Severe'' case as ``Mild'' is a far more critical error than classifying it as ``Moderate''.

### Facial Age Estimation (AFAD)

Age estimation from facial images is a common computer vision benchmark for ordinal regression. Datasets like the Asian Face Age Dataset (AFAD) [Niu et al., 2016] contain hundreds of thousands of face images labeled with their corresponding age.

**Ordinal Task:** The goal is to predict the age of the person in the image. This is treated as an ordinal problem, as predicting age 40 for a 41-year-old is a much smaller error than predicting age 20. The ages are often treated as direct ordinal categories or binned into ordered groups.

### Sentiment Review Datasets (Amazon/Yelp)

In Natural Language Processing (NLP), sentiment analysis is a quintessential ordinal task. Large-scale datasets like the Amazon Product Reviews [Ni et al., 2019] or the Yelp Open Dataset [Yelp, Inc., 2024] provide millions of user-submitted reviews. The raw text is converted into high-dimensional feature vectors (e.g., using TF-IDF or, more commonly, deep learning embeddings).

**Ordinal Task:** The goal is to predict the user's star rating (typically 1 to 5 stars) based on the text of their review. The 1-5 star scale is inherently ordered, and a 4-star prediction for a 5-star review is a much better model outcome than a 1-star prediction.

## Performance Comparison

Table: Comparing Model Performance on Benchmark Datasets. Values are mean $\pm$ std over 30 runs. A (*) indicates a result is statistically significantly better than the Decision Tree baseline $(p < 0.05)$. The best MAE result for each dataset is highlighted in **bold**.

The analysis of the low-dimensional datasets reveals that modern, ordinal-aware machine learning models achieve the most competitive performance. On three of the four datasets—Wine Quality, Boston Housing, and Poker Hand—the best-performing model (lowest MAE) was either a Family 3 deep learning model (MLP (EMD Loss) or CORN) or a Family 1 threshold-based machine learning model (SVOR). On Boston Housing, the `CORN (MLP Base)` (MAE 0.2804) and `SVOR` (MAE 0.2889) models are in a clear top-tier, significantly outperforming other models. On Wine Quality, the `MLP (EMD Loss)` (MAE 0.4664) and `SVOR` (MAE 0.4666) are statistically tied for first place. This indicates that for complex tabular data, both specialized deep learning losses and kernel-based threshold methods provide a distinct and measurable advantage.

The Car Evaluation dataset presents a notable exception to this trend. On this task, the standard Decision Tree baseline model achieves the best MAE (0.0252), narrowly outperforming all other models, including the strong `SVOR` (MAE 0.0375). This result is likely attributable to the dataset's known origin from a simple, hierarchical decision model [Bohanec & Rajkovic, 1990], a structure that a decision tree is ideally suited to capture. This finding underscores that for simpler, rule-based datasets, the added complexity of neural networks may not provide any benefit.

The results also highlight the inconsistent performance or failure of certain model families. The classic statistical `CLM (Ordinal Ridge)` model, while interpretable, is clearly not competitive on these tasks, performing particularly poorly on Car Evaluation and Poker Hand (the latter resulting in a QWK of 0.0000). More striking is the unreliability of the `Adjacent (MLP Base)` binary decomposition method. This model failed entirely on the Car Evaluation dataset (Balanced Acc. 0.0000) and performing as the worst-performing model on Boston Housing, suggesting this specific decomposition strategy may be unstable or fundamentally unsuited for these particular problems.

Finally, a direct comparison between the naive `MLP (Classification)` and its ordinal-aware counterparts (e.g., `CORN`, `EMD`, and `SVOR`) is illuminating. On Boston Housing and Poker Hand, the specialized ordinal models demonstrate a clear, statistically significant superiority over the naive nominal classification approach across all ordinal metrics. However, on the Wine Quality dataset, the performance of all MLP-based models is tightly clustered. The best ordinal model (MLP (EMD Loss)) offers only a marginal, likely insignificant, improvement over the naive MLP. This suggests that while specialized ordinal losses are generally beneficial, their impact can be modest on particularly noisy or imbalanced datasets.

A visual summary of these performance metrics for the MLP-based models is presented in Figure~\ref{fig:performance_summary_grid}.

![Summary of model performance across key metrics and datasets. The right column, including the bottom, shows metrics where higher is better (QWK, Accuracy, Balanced Accuracy), while the left column, excluding the bottom, shows error metrics where lower is better (MAE, MSE). All evaluations use an MLP base model.](figures/performance_summary_chart.png)

The analysis of the generated figures largely corroborates the findings from the detailed results table, highlighting clear trends in model performance across the low-dimensional datasets. Deep learning models utilizing ordinal-aware loss functions, such as MLP (EMD Loss) and CORN, consistently demonstrate strong performance, often securing the lowest Mean Absolute Error (MAE) and highest Quadratic Weighted Kappa (QWK) on datasets like Boston Housing and Poker Hand. This visual evidence underscores the advantage of explicitly accounting for the ordinal nature of the target variable in tasks with complex underlying patterns.

The Car Evaluation dataset presents a particularly interesting case. Here, the Support Vector Ordinal Regression (SVOR) model stands out, achieving the best performance for both MAE and QWK, significantly outperforming the standard Decision Tree model and other MLP-based approaches. This highlights SVOR's particular suitability for this rule-based, categorical dataset. While the Decision Tree model also shows competitive MAE, and several MLP-based approaches perform comparably for metrics like QWK and Accuracy, SVOR's overall dominance on Car Evaluation is a notable finding.

Conversely, some model families exhibit clear underperformance or instability. The classic statistical CLM (Ordinal Ridge) model consistently ranks among the lowest performers across all datasets and metrics. More strikingly, the Adjacent (MLP Base) model shows significant instability, failing entirely on the Car Evaluation dataset (0.0000 Balanced Accuracy) and performing poorly on Boston Housing. This suggests that this specific decomposition strategy may be fundamentally unsuited or unstable for these problems. The comparison between naive MLP (Classification) and its ordinal-aware counterparts, including SVOR, further illustrates the benefits of specialized ordinal losses, showing clear superiority on Boston Housing and Poker Hand, though the impact is more modest on datasets like Wine Quality, where all MLP-based models, along with SVOR, perform within a tight cluster.

While evaluation results for a Support Vector Ordinal Regression (SVOR) model on the Poker dataset are now available, it is crucial to understand that its training inherently presents significant computational challenges. The Poker dataset, comprising 25,011 samples each with 10 features, demands considerable processing. Given that standard SVM training algorithms, upon which SVOR is based, typically exhibit a time complexity ranging from $O(n^2)$ to $O(n^3)$ with respect to the number of samples (n), a dataset of this magnitude implies a substantial computational burden. Furthermore, the utilization of a Radial Basis Function (RBF) kernel, while essential for capturing non-linear relationships, often exacerbates this complexity by implicitly mapping data into a higher-dimensional space. Consequently, achieving trained SVOR models with an RBF kernel on this dataset theoretically requires a considerable amount of time, potentially spanning hours or even days on conventional hardware, even with modern implementations incorporating various optimization techniques to mitigate these demands.

## Strengths and Weaknesses Summary

Finally, the relative strengths and weaknesses of the three main model families are summarized in Table~\ref{tab:strengths_weaknesses}.

Table: Comparative Analysis of Ordinal Classification Model Families

The **Threshold-Based Models (Family 1)** must be analyzed in two distinct parts. The classic *statistical* models (e.g., POM [McCullagh, 1980] and our `CLM (Ordinal Ridge)`) are theoretically attractive due to their high interpretability. However, their rigid ``Parallel Lines Assumption'' is a critical weakness. The results demonstrate this: the `CLM (Ordinal Ridge)` model fails completely on the `Poker Hand` dataset (QWK 0.0000) and is uncompetitive on all other tasks. In stark contrast, the *machine learning* threshold models, like `SVOR` [Chu & Keerthi, 2007; Herbrich et al., 1999], are highly competitive. `SVOR` achieved top-tier, statistically significant performance on both `Wine Quality` and `Boston Housing`, proving that the latent threshold concept is extremely effective when implemented in a flexible, large-margin framework.

The **Binary Decomposition (Family 2)** family is theoretically flexible, as it avoids the parallel lines assumption and can leverage any binary classifier [Frank & Hall, 2001]. Its main drawback, as noted in the paper, is the potential for ``inconsistent probabilities'' and the computational cost of training $k-1$ models. The experimental results, however, reveal a much more severe and practical weakness for the specific `Adjacent (MLP Base)` strategy. This model was highly unstable, failing catastrophically on the `Car Evaluation` dataset (0.0000 Balanced Accuracy) and performing as the worst model by a large margin on `Boston Housing`. This suggests that while theoretically sound, certain decomposition strategies are not practically viable or robust.

The **Deep Learning (Family 3)** approaches emerge as one of the clear winners in terms of performance on complex, low-dimensional data. This family's strength lies in its ability to combine powerful non-linear feature extraction with an ordinal-aware objective. The results show that `MLP (EMD Loss)` [De Matos et al., 2019] and `CORN (MLPBase)` [Shi et al., 2021] achieve the best MAE on three of the four datasets (tying with `SVOR` on one). Their superiority is cemented by their statistically significant out-performance of the naive `MLP (Classification)`, proving the value of encoding ordinal information in the loss function. The primary weakness of this family, aside from its ``black box'' nature, is that model choice is critical. The results show that not all ordinal losses are equal; the `CORAL (MLP Base)` model [Cao et al., 2020], for example, consistently underperformed.

Finally, the results highlight a crucial strength of **Naive Approaches** on simple, rule-based problems. The naive `Decision Tree` baseline was the winner on the `Car Evaluation` dataset. This provides essential context: while modern ordinal-aware ML is the most powerful approach for complex tasks, the ``strength'' of a simpler, interpretable model should not be discounted for datasets that match its underlying assumptions.

# Open Challenges and Future Research Directions

While our discussion highlights the significant progress in the field, it also reveals several limitations and unresolved questions. To conclude our analysis, this section looks forward, identifying the key open challenges and outlining promising directions for future research in ordinal classification.

- **Scalability (large $k$ or large $n$):**
    A primary challenge is scalability. Many established methods, especially in the binary decomposition family, become computationally infeasible as the number of categories ($k$) grows, as they require training $k-1$ separate models [Frank & Hall, 2001]. Similarly, while deep learning approaches are well-suited for large numbers of instances ($n$), the training time for classic statistical or kernel-based models like Support Vector Ordinal Regression (SVOR) can become prohibitive [Herbrich et al., 1999; Chu & Keerthi, 2007]. Future research must focus on developing models that are efficient in both large-$k$ and large-$n$ regimes.

- **Interpretability of complex models:**
    As evidenced by the performance in Table III, a clear trade-off exists between interpretability and performance. The best-performing deep learning models [Niu et al., 2016; De Matos et al., 2019; Cao et al., 2020; Shi et al., 2021] are often ``black boxes,'' a significant weakness that limits their adoption in high-stakes fields like medicine or finance where explanatory power is crucial. In contrast, highly interpretable threshold models [McCullagh, 1980] often lag in performance. A critical avenue for future work is the application and adaptation of eXplainable AI (XAI) techniques to these specialized ordinal deep learning models.

- **Uncertainty quantification:**
    Most ordinal models provide a single point prediction (the most likely rank) but fail to quantify the model's confidence in that prediction. This is a major limitation, as a prediction of ``Moderate Severity'' with 99% confidence is far more actionable than one with 55% confidence. The potential for inconsistent probabilities in binary decomposition models [Frank & Hall, 2001] further complicates this issue. Developing frameworks for reliable uncertainty quantification, perhaps through Bayesian ordinal regression or specialized ensemble methods, is essential for moving ordinal models into safety-critical decision-making loops.

- **New application domains:**
    While research has long focused on tabular data [Cortez et al., 2009; Bohanec & Rajkovic, 1990; Harrison & Rubinfeld, 1978], static images [Kaggle, 2015; Niu et al., 2016], and text analysis [Ni et al., 2019; Yelp, Inc., 2024], a promising frontier is the extension of ordinal methods to more complex data types. Recent work has already begun to push into high-dimensional medical imaging for assessing disease severity from chest radiographs [Wienholt et al., 2024], classifying diagnostic certainty from radiology reports [Fujimoto et al., 2024], and predicting brain age from MRI scans [Solanky et al., 2024]. Similarly, ordinal deep learning is being applied in environmental science with LiDAR point-cloud data [Peña-Alonso et al., 2024] and in finance for resource allocation problems [Cohen & Singer, 2024]. Future work should continue this trend, extending ordinal-aware models to new domains such as time-series forecasting and graph-structured data.

# Conclusion

This review has provided a comprehensive survey of ordinal classification, a unique supervised learning task defined by a target variable whose categories possess a natural order.

We began by formally defining this problem, contrasting it with nominal classification, which discards valuable order information, and metric regression, which makes the often-invalid assumption of equidistant categories. We also emphasized the necessity of using specialized evaluation metrics like Mean Absolute Error (MAE) and Quadratic Weighted Kappa (QWK), which, unlike standard accuracy, appropriately penalize prediction errors based on their ordinal distance from the true rank.

The core of this survey categorized the diverse methodologies into three primary families, as illustrated in our taxonomy (Figure~\ref{fig:taxonomy}). We examined the classic **Threshold-Based Models**, noting the critical distinction between statistically rigorous but rigid models like the Proportional Odds Model [McCullagh, 1980] and their high-performance machine learning counterparts like SVOR [Chu & Keerthi, 2007; Herbrich et al., 1999]. We then reviewed the flexible **Binary Decomposition** strategies [Frank & Hall, 2001], which reframe the problem into a series of binary classifications but can suffer from instability or inconsistent probabilities. Finally, we detailed the **Deep Learning Approaches**, which have achieved state-of-the-art performance by integrating specialized ordinal loss functions (like CORAL [Cao et al., 2020], CORN [Shi et al., 2021], or EMD [De Matos et al., 2019]) or novel output layers [Niu et al., 2016] directly into modern neural network architectures.

Our comparative analysis, supported by experimental results on benchmark datasets (Table~\ref{tab:performance_comparison}), reinforces the central thesis of this review: *explicitly modeling the ordinal structure is critical for optimal performance*. We demonstrated that modern, ordinal-aware models—spanning both threshold-based machine learning (SVOR) and specialized deep learning (CORN, EMD)—statistically outperform naive approaches on complex tasks. By summarizing the clear trade-offs between model interpretability, flexibility, and predictive power (Table~\ref{tab:strengths_weaknesses}), this survey provides a guide for practitioners to select the most appropriate model and for researchers to address the open challenges that remain in this important field.

# References

- Agresti, A. (2010). *Analysis of ordinal categorical data*. John Wiley & Sons.
- Alcalá-Fdez, J. et al. (2011). KEEL data-mining software tool: data set repository, integration of algorithms and experimental analysis framework. *Journal of Multiple-Valued Logic and Soft Computing*, 17(2-3), 255-287.
- Bache, K. & Lichman, M. (2013). *UCI Machine Learning Repository*. University of California, Irvine, School of Information and Computer Science. [Online]. Available: http://archive.ics.uci.edu/ml
- Bohanec, M. & Rajkovic, V. (1990). Expert system for decision making. *Sistemica*, 1(1), 145-157.
- Cao, W., Mirjalili, V. & Raschka, S. (2020). Rank consistent ordinal regression for neural networks with application to age estimation. *Pattern Recognition Letters*, 140, 325-331.
- Cattral, R. & Oppacher, F. (2007). Discovering rules in the poker hand dataset. In *Proceedings of the 2007 GECCO conference companion on Genetic and evolutionary computation*, 2487-2492.
- Chu, W. & Keerthi, S. S. (2007). Support vector ordinal regression. *Journal of Machine Learning Research*, 8(Mar), 597-613.
- Cohen, I. R. & Singer, G. (2024). *Resource allocation in ordinal classification problems*. Working Paper. Available at: https://ilanrcohen.droppages.com/pdfs/Resource-Constraint-classisifcation-ordinal.pdf
- Cortez, P. et al. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547-553.
- De Matos, P. T. et al. (2019). Ordinal classification with Earth Mover's Distance for content-based image retrieval. In *2019 International Joint Conference on Neural Networks (IJCNN)*, 1-8.
- Fujimoto, K., Tanabe, N. & Yoshiura, T. (2024). Classification of Diagnostic Certainty in Radiology Reports with Deep Learning. *Stud Health Technol Inform*, 310, 569-573.
- Frank, E. & Hall, M. (2001). A simple approach to ordinal classification. In *European conference on machine learning*, 145-156.
- Harrison, D. & Rubinfeld, D. L. (1978). Hedonic housing prices and the demand for clean air. *Journal of Environmental Economics and Management*, 5(1), 81-102.
- Herbrich, R., Graepel, T. & Obermayer, K. (1999). Support vector learning for ordinal regression. In *Proceedings of the Ninth International Conference on Artificial Neural Networks (ICANN 99)*, 97-102.
- Kaggle. (2015). *Diabetic Retinopathy Detection*. [Online]. Available: https://www.kaggle.com/c/diabetic-retinopathy-detection
- Peña-Alonso, C. et al. (2024). Deep Ordinal Classification in Forest Areas Using Light Detection and Ranging Point Clouds. *Sensors (Basel)*, 24(7), 2168.
- McCullagh, P. (1980). Regression models for ordinal data. *Journal of the Royal Statistical Society, Series B (Methodological)*, 42(2), 109-142.
- Ni, J., Li, L. & McAuley, J. (2019). Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 188-197.
- Niu, Z. et al. (2016). Ordinal regression with multiple output cnn for age estimation. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, 4920-4928.
- Solanky, B. et al. (2024). Ordinal Classification with Distance Regularization for Robust Brain Age Prediction. In *Medical Image Computing and Computer Assisted Intervention – MICCAI 2024*, 14963, 403–413. Springer Nature Switzerland.
- Shi, X., Cao, W. & Lau, J. H. (2021). Deep conditional ordinal regression for neural networks. In *Proceedings of the 35th AAAI Conference on Artificial Intelligence*.
- Wienholt, P. et al. (2024). An Ordinal Regression Framework for a Deep Learning Based Severity Assessment for Chest Radiographs. *arXiv preprint arXiv:2402.05685*.
- Yelp, Inc. (2024). *Yelp Open Dataset*. [Online]. Available: https://www.yelp.com/dataset