\documentclass[journal]{IEEEtai}

\usepackage[colorlinks,urlcolor=blue,linkcolor=blue,citecolor=blue]{hyperref}
\usepackage{color,array}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsfonts}
\usepackage{siunitx} % For aligning numbers in tables
\usepackage{subcaption}

%% \jvol{XX}
%% \jnum{XX}
%% \paper{1234567}
%% \pubyear{2020}
%% \publisheddate{xxxx 00, 0000}
%% \currentdate{xxxx 00, 0000}
%% \doiinfo{TQE.2020.Doi Number}

\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\setcounter{page}{1}
%% \setcounter{secnumdepth}{0}


\begin{document}


\title{Survey on Ordinal Classification} 

\author{Jesse Wood, \IEEEmembership{Member, IEEE}, Bach Nguyen \IEEEmembership{Member, IEEE}, Bing Xue \IEEEmembership{Fellow, IEEE}, Mengjie Zhang \IEEEmembership{Fellow, IEEE}, Daniel Killeen \IEEEmembership{Member, IEEE}
\thanks{This manuscript was submitted on the 27th of October 2025. This work was supported by the MBIE Fund on Research Program under contract C11X2001.}
\thanks{J. Wood, B. Nguyen, B. Xue, M. Zhang are associated with the Center for Data Science and Artificial Intelligence (CDSAI) at Victoria University of Wellington, Kelburn Parade, Wellington, 6012, New Zealand}
\thanks{D. Killen is associated with the Seafood Technologies department of Plant and Food Research New Zealand, Akerston Street, Nelson, 7010, New Zealand}
\thanks{This paragraph will include the Associate Editor who handled your paper.}}
\markboth{Journal of IEEE Transactions on Artificial Intelligence, Vol. 00, No. 0, Month 2020}
{J. Wood \MakeLowercase{\textit{et al.}}: Suvery on Ordinal Classification}

\maketitle

\begin{abstract}
Ordinal classification, or ordinal regression, is a unique supervised learning task where the target variable's categories possess a natural order. This problem is distinct from nominal classification, which ignores order, and metric regression, which assumes equidistant categories. This review surveys the primary families of methods developed to address this challenge, categorizing them into threshold-based models, binary decomposition approaches, and modern deep learning techniques. We discuss the key models within each family, analyze their respective strengths and limitations, and review the standard evaluation metrics for a rigorous comparison. Finally, we highlight significant open challenges and outline promising directions for future research.
\end{abstract}

\begin{IEEEImpStatement}
This review provides a comprehensive and structured overview of the field of ordinal classification, a critical task in many real-world applications such as medicine, social science, and sentiment analysis. By categorizing the diverse methodologies, from classic statistical models to modern deep learning approaches, this work serves as an essential resource for researchers and practitioners. It lowers the barrier to entry for new researchers, helps experienced practitioners select the most appropriate model for their problem, and identifies key open challenges, thereby guiding future research toward areas of greatest need and potential impact.
\end{IEEEImpStatement}

\begin{IEEEkeywords}
Ordinal Classification, Ordinal Regression, Literature Review, Machine Learning, Threshold Models, Deep Learning, Evaluation Metrics
\end{IEEEkeywords}

\section{Introduction}
\label{sec:intro}

In many real-world machine learning problems, from medical diagnoses to customer satisfaction surveys, the target variable is not just a set of discrete categories but a set of \textit{ordered} categories. This fundamental distinction forms the basis of ordinal classification, the subject of this review. This section will define the core problem, highlight its importance, and outline the structure of the paper.

\subsection{Definition of Ordinal Classification}

Ordinal classification, also known as ordinal regression, is a type of supervised learning problem where the goal is to assign instances to categories that have a natural, inherent order. Formally, given an input instance described by a feature vector $X \in \mathbb{R}^d$, the task is to predict its corresponding target label $Y$, which belongs to a finite set of $k$ ordered categories, denoted as \{$C_1, C_2, ..., C_k$\}, such that a meaningful ordering relationship exists: $C_1 \prec C_2 \prec ... \prec C_k$.

This task occupies a unique space between standard classification and regression:

\begin{itemize}
    \item \textbf{Contrast with Nominal Classification:} Unlike nominal classification problems where categories are distinct but unordered (e.g., classifying images as `cat', `dog', or `bird'), the order between categories in ordinal classification holds significant meaning (e.g., `low', `medium', `high' risk). Ignoring this order, as nominal classifiers do, results in a loss of valuable information.
    \item \textbf{Contrast with Metric Regression:} While both ordinal classification and metric regression deal with ordered outcomes, metric regression assumes that the target variable is measured on at least an interval scale, implying that the distances between consecutive values are quantifiable and meaningful (e.g., predicting temperature in Celsius). Ordinal classification relaxes this assumption; the categories have a clear sequence, but the ``distance'' or difference between adjacent categories (e.g., the difference between `mild' and `moderate' disease severity) is not necessarily uniform or precisely defined.
\end{itemize}

The fundamental challenge in ordinal classification is to develop models that effectively leverage the rank information inherent in the labels, penalizing prediction errors based on their ordinal distance from the true category, rather than treating all misclassifications equally. This conceptual difference is illustrated in Figure~\ref{fig:data_types}.


\begin{figure*}
    \centering
    \includegraphics[width=\linewidth]{data_type_contrast.png}
    \caption{A conceptual comparison of data measurement scales. (a) \textbf{Nominal} data consists of categories without any inherent order (e.g., fruit types). (b) \textbf{Ordinal} data features categories with a meaningful order, but the distances between them are not precisely defined or uniform (e.g., clothing sizes). (c) \textbf{Metric} (or Interval/Ratio) data has both a meaningful order and quantifiable, equidistant intervals between values (e.g., numerical measurements).}
\label{fig:data_types}
\end{figure*}

\subsection{Importance and Applications}

\par This task is critical in many real-world applications, as summarized in Table~\ref{tab:real_world_examples}. The importance of ordinal classification continues to grow, with recent advancements, particularly driven by deep learning, expanding its application into increasingly complex domains. In medical imaging, state-of-the-art approaches now tackle tasks such as detailed disease severity grading in chest radiographs \cite{Wienholt2024}, predicting biological age from MRI scans while preserving ordinal relationships \cite{ORDERLoss2024}, and quantifying diagnostic certainty levels directly from radiology reports \cite{DiagnosticCertainty2024}. Beyond healthcare, ordinal methods are being applied in environmental science, for instance, using LiDAR data for fine-grained classification of forest vegetation strata \cite{LiDARForest2024}, and remain crucial in areas like financial risk assessment (e.g., credit scoring) \cite{Cohen2024}. These modern applications underscore the value of explicitly modeling order for improved accuracy and interpretability.

\begin{table*}[htbp]
  \centering
  \caption{Examples of Real-World Ordinal Classification Applications}
  \label{tab:real_world_examples}
  \begin{tabular}{lll}
    \toprule
    \textbf{Domain} & \textbf{Application Example} & \textbf{Ordered Classes ($C_1 < \dots < C_k$)} \\
    \midrule
    Medicine & Disease Severity & \{None, Mild, Moderate, Severe, Proliferative\} \\
    Sentiment Analysis & Product Star Rating & \{1-Star, 2-Star, 3-Star, 4-Star, 5-Star\} \\
    Social Science & Likert Scale Survey & \{Strongly Disagree, Disagree, Neutral, Agree, Strongly Agree\} \\
    Finance & Credit Rating & \{AAA, AA, A, BBB, BB, B, \dots\} \\
    Computer Vision & Age Estimation & \{18-25, 26-35, 36-45, \dots\} \\
    \bottomrule
  \end{tabular}
\end{table*}

\subsection{Scope and Objectives}

The primary objective of this review is to provide a comprehensive and structured overview of the field of ordinal classification. We aim to survey the major families of methodologies developed for this task, ranging from traditional statistical models to modern deep learning techniques. This involves:

\begin{itemize}
    \item Categorizing existing methods into distinct families based on their underlying principles (threshold-based, binary decomposition, deep learning).
    \item Discussing the key algorithms within each family, explaining their core mechanisms.
    \item Analyzing the relative strengths, weaknesses, and common assumptions associated with each approach.
    \item Reviewing standard evaluation metrics specifically designed for ordinal data to enable rigorous comparison.
    \item Identifying common benchmark datasets used in the literature.
\end{itemize}

Ultimately, this survey seeks to serve as a valuable resource for both newcomers seeking an introduction to the field and experienced researchers and practitioners looking for a comparative analysis to guide model selection and identify avenues for future research. The scope is focused on supervised learning methods explicitly designed for or adapted to ordinal target variables.

\subsection{Structure of the Review}
\label{sec:structure}

This review is organized as follows: Section II formally
defines the ordinal classification problem, establishes nota-
tion, and critically examines the evaluation metrics necessary
for assessing performance while accounting for label order.
Section III discusses common baseline or ``naive" approaches,
such as treating the problem as nominal classification or
metric regression, and highlights their inherent limitations.
The core of the survey is Section IV, which categorizes
and details the three primary families of specialized ordinal
classification models: threshold-based methods, binary decom-
position strategies, and modern deep learning techniques.
Section V provides a comparative analysis, discussing benchmark
datasets, comparing model performance, and summarizing the
strengths and weaknesses of each family.
Finally, Section~\ref{sec:challenges} % <-- Corrected from ??
identifies key open challenges and suggests directions for
future research, before Section VII concludes the review with
a summary of the key findings.

% --- 2. PROBLEM FORMULATION AND EVALUATION ---
\section{Problem Formulation and Evaluation}
\label{sec:formulation}

% --- CONNECTING SENTENCE ---
Having introduced the concept of ordinal classification and its importance, we must now establish a formal understanding of the problem. This next section will define the mathematical notation and, critically, review the specialized evaluation metrics required to properly assess ordinal model performance.

\subsection{Formal Notation}

Let $X \in \mathbb{R}^d$ represent the input features for an instance, where $d$ is the number of features. The goal is to predict the corresponding ordinal target label $Y$, which belongs to a set of $k$ ordered categories \{$C_1, C_2, ..., C_k$\}, such that $C_1 \prec C_2 \prec ... \prec C_k$. We can map these categories to integer ranks \{$1, 2, ..., k$\ } for notational convenience, where $y_i$ is the true integer rank for instance $i$ and $\hat{y}_i$ is the predicted integer rank.

\subsection{Evaluation Metrics}

Standard classification accuracy, which measures the proportion of correctly classified instances (a 0-1 loss), is often insufficient for ordinal problems. It treats all misclassifications equally, ignoring the inherent order. For example, predicting class $C_1$ when the true class is $C_k$ is penalized the same as predicting $C_{k-1}$ when the true class is $C_k$. This fails to capture the severity of errors in an ordinal context, where larger rank differences represent worse predictions. Figure~\ref{fig:evaluation_metrics} illustrates this contrast. Consequently, specialized metrics that account for the ordered nature of the labels are essential.

\begin{itemize}
    \item \textbf{Mean Absolute Error (MAE):} This is arguably the most common and interpretable metric for ordinal tasks. It measures the average absolute difference between the predicted rank ($\hat{y}_i$) and the true rank ($y_i$) across all $N$ instances in the evaluation set:
    \begin{equation}
        MAE = \frac{1}{N} \sum_{i=1}^N | \hat{y}_i - y_i |
    \end{equation}
    MAE directly reflects the average magnitude of prediction errors in terms of rank distance. A lower MAE indicates better performance. It penalizes errors linearly based on their distance from the true rank.

    \item \textbf{Quadratic Weighted Kappa (QWK):} This metric measures the agreement between predicted and true ratings, corrected for chance agreement. It is particularly robust because it uses quadratic weights to penalize disagreements. The weight $w_{ij}$ assigned to a disagreement between true class $i$ and predicted class $j$ is typically calculated as:
    \begin{equation}
        w_{ij} = \frac{(i-j)^2}{(k-1)^2}
    \end{equation}
    This means that larger errors (greater distance between $i$ and $j$) are penalized much more heavily than smaller errors. QWK ranges from -1 (total disagreement) to 1 (perfect agreement), with 0 indicating agreement equivalent to chance. Higher QWK values indicate better performance.

    \item \textbf{Other Metrics:} While MAE and QWK are prevalent, other metrics are sometimes used. \textbf{Mean Squared Error (MSE)} ($\frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$) can be employed, but like QWK, its quadratic penalty makes it more sensitive to large errors or outliers compared to MAE. Standard \textbf{classification accuracy} is occasionally reported for comparison but, as discussed, provides an incomplete picture by ignoring the ordinal structure. Balanced accuracy might also be considered, especially in cases of class imbalance, but still treats errors nominally.
\end{itemize}

\par The fundamental difference between nominal and ordinal evaluation, highlighting the varying costs assigned to misclassifications, is visualized in Figure~\ref{fig:evaluation_metrics}.

\begin{figure*}
    \centering
    \includegraphics[width=\linewidth]{evaluation_metrics_contrast.png}
    \caption{Conceptual comparison of evaluation metrics for a 5-class problem. (a) \textbf{Standard Accuracy} (Nominal Loss) uses a 0-1 cost, treating all misclassifications as equally incorrect. (b) \textbf{Ordinal Metrics} (such as MAE or QWK) use a weighted cost. The penalty for an error is proportional (MAE) or quadratically proportional (QWK) to its distance from the true class, making a large error (e.g., predicting $C_1$ when the true class is $C_5$) far more costly than a small error (e.g., predicting $C_4$ when the true class is $C_5$).}
\label{fig:evaluation_metrics}
\end{figure*}

% --- 3. BASELINE AND NAIVE APPROACHES ---
\section{Baseline and Naive Approaches}
\label{sec:baseline}

% --- CONNECTING SENTENCE ---
With a clear problem formulation and the correct evaluation metrics established, we first turn our attention to the most straightforward methods for tackling this task. These ``naive" approaches adapt standard machine learning techniques by either ignoring the ordinal nature of the labels or making simplifying assumptions about them. The following section reviews these common baselines, highlighting their inherent limitations to motivate the need for specialized ordinal techniques.

\subsection{Ignoring the Order (Nominal Classification)}

A simple baseline is to treat the ordinal classification problem as a standard multiclass nominal classification problem. This involves using algorithms designed for unordered categories, such as:
\begin{itemize}
    \item Multinomial Logistic Regression (Softmax Regression)
    \item Support Vector Machines (using one-vs-rest or one-vs-one strategies)
    \item Decision Trees (e.g., CART, C4.5)
    \item Random Forests
    \item Neural Networks with a softmax output layer and cross-entropy loss.
\end{itemize}

\textbf{Limitation:} The primary drawback of this approach is the complete loss of the ordering information present in the labels ($C_1 \prec C_2 \prec ... \prec C_k$). The model learns to discriminate between categories but is not aware that misclassifying $C_1$ as $C_k$ is a more severe error than misclassifying $C_1$ as $C_2$. This often leads to suboptimal performance when evaluated using appropriate ordinal metrics like MAE or QWK.

\subsection{Assuming Equidistance (Metric Regression)}

Another common naive approach is to map the ordinal categories \{$C_1, ..., C_k$\ } to numerical values, typically integers \{$1, ..., k$\ }, and then treat the problem as a standard metric regression task. Algorithms used include:
\begin{itemize}
    \item Linear Regression
    \item Support Vector Regression (SVR)
    \item Regression Trees
    \item Gradient Boosting Machines (for regression)
    \item Neural Networks with a single linear output unit and a regression loss (e.g., MSE, MAE).
\end{itemize}

The continuous output predicted by the regression model is then typically rounded to the nearest integer (or thresholded) to obtain the final ordinal class prediction.

\textbf{Limitation:} This approach makes a strong, often invalid, assumption that the ``distance" or interval between consecutive categories is equal and meaningful (e.g., the difference between``Mild" and``Moderate" disease is quantitatively the same as between ``Moderate" and ``Severe"). This equidistance assumption rarely holds for true ordinal scales. Furthermore, the final rounding step can introduce biases, especially for predictions near the midpoint between two integer ranks. While this method implicitly uses some order information (by mapping to numbers), the flawed assumption about the scale can limit its effectiveness.


% --- 4. FAMILIES OF ORDINAL CLASSIFICATION MODELS ---
\section{Families of Ordinal Classification Models}
\label{sec:models}

% --- CONNECTING SENTENCE ---
The shortcomings of nominal and regression-based baselines underscore the necessity of models designed specifically for ordinal data. Therefore, the core of this review, presented in the next section, is a comprehensive survey of the main families of ordinal classification models. We will categorize and analyze threshold-based models, binary decomposition methods, and modern deep learning approaches.

This is the core of the review, organized by methodology. We illustrate the taxonomy of the three different families of ordinal classification in Figure~\ref{fig:taxonomy}.

\begin{figure*}
    \centering
    \includegraphics[width=\linewidth]{taxonomy_flowchart.png}
    \caption{A taxonomy of ordinal classification methods, providing a visual map for the review. The primary distinction is made between Naive Approaches (which ignore or misuse the ordinal information) and Specialized Ordinal Models. These specialized models are categorized into the three main families: Threshold-Based, Binary Decomposition, and Deep Learning.}
\label{fig:taxonomy}
\end{figure*}

\subsection{Family 1: Threshold-Based Models}

The first and most established family of ordinal models is built on the concept of a continuous latent variable. The core idea is that this unobserved variable $z$ is ``sliced" by $k-1$ ordered thresholds ($\theta_1, \dots, \theta_{k-1}$) to produce the $k$ observed categories. The most prominent example is the \textbf{Proportional Odds Model (POM)} \cite{mccullagh1980regression}, also known as Ordinal Logistic Regression. This model is itself a specific instance of the broader \textbf{Cumulative Link Models (CLMs)} \cite{mccullagh1980regression} family, which generalizes the approach for different link functions (e.g., probit or complementary log-log). This same latent variable concept was later adapted for machine learning, leading to \textbf{Support Vector Ordinal Regression (SVOR)} \cite{herbrich1999support,chu2007support}, which implements the thresholds as a set of parallel separating hyperplanes in a large-margin framework. A conceptual diagram of this latent variable approach is shown in Figure~\ref{fig:threshold_model}.

\begin{figure}
    \centering
    \includegraphics[width=\linewidth]{threshold_based_ordinal_models.png}
    \caption{Conceptual diagram of Threshold-Based Ordinal Models (e.g., the Proportional Odds Model). A continuous, unobserved latent variable $z^*$ (here, following a logistic distribution) is partitioned into $k$ discrete ordinal categories ($C_1, \dots, C_k$) by $k-1$ ordered thresholds ($\theta_1, \dots, \theta_{k-1}$). An observed instance falls into category $C_j$ if its latent variable value $z^*$ lies between $\theta_{j-1}$ and $\theta_j$.}
\label{fig:threshold_model}
\end{figure}

\subsection{Family 2: Binary Decomposition Models}

A second major family recasts the $k$-class ordinal problem into a series of $k-1$ binary classification sub-problems. This approach is not a single model but a meta-strategy that can use any binary classifier (such as an SVM or decision tree) as its base. The most common, order-preserving strategy is to build $k-1$ cumulative classifiers, each answering the question``is the true class $y > C_j$?