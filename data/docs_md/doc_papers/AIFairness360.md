
## Terminology
In this section, we briefly define specialized terminology from the field of fairness in machine learning. 
1. Favorable label: A favorable label is a label whose value corresponds to an outcome that provides an advantage to the recipient. Examples are receiving a loan, being hired for a job, and not being arrested. 
2. Protected attribute: A protected attribute is an attribute that partitions a population into groups that have parity in terms of benefit received. Examples include race, gender, caste, and religion. Protected attributes are not universal, but are application specific. 
3. Privileged Value: A privileged value of a protected attribute indicates a group that has historically been at a systematic advantage. 
4. Group fairness: Group fairness is the goal of groups defined by protected attributes receiving similar treatments or outcomes. 
5. Individual fairness: Individual fairness is the goal of similar individuals receiving similar treatments or outcomes. Bias is a systematic error. 
6. Fairness: In the context of fairness, we are concerned with unwanted bias that places privileged groups at a systematic advantage and unprivileged groups at a systematic disadvantage. 
7. Fairness metric: A fairness metric is a quantification of unwanted bias in training data or models. 
8. Bias mitigation algorithm: A bias mitigation algorithm is a procedure for reducing unwanted bias in training data or models.

## Metrics

### Statistical Parity Difference
This is the difference in the probability of favorable outcomes between the unprivileged and privileged groups. This can be computed both from the input dataset as well as from the dataset output from a classifier (predicted dataset). A value of 0 implies both groups have equal benefit, a value less than 0 implies higher benefit for the privileged group, and a value greater than 0 implies higher benefit for the unprivileged group.

### Disparate Impact
This is the ratio in the probability of favorable outcomes between the unprivileged and privileged groups. This can be computed both from the input dataset as well as from the dataset output from a classifier (predicted dataset). A value of 1 implies both groups have equal benefit, a value less than 1 implies higher benefit for the privileged group, and a value greater than 1 implies higher benefit for the unprivileged group.

### Average odds difference
This is the average of difference in false positive rates and true positive rates between unprivileged and privileged groups. This method needs to be computed using the input and output datasets to a classifier. A value of 0 implies both groups have equal benefit, a value less than 0 implies higher benefit for the privileged group and a value greater than 0 implies higher benefit for the unprivileged group.

### Equal opportunity difference
This is the difference in true positive rates between unprivileged and privileged groups. This method needs to be computed using the input and output datasets to a classifier. A value of 0 implies both groups have equal benefit, a value less than 0 implies higher benefit for the privileged group and a value greater than 0 implies higher benefit for the unprivileged group.


## Bias mitigation Approaches
Bias mitigation algorithms attempt to improve the fairness metrics by modifying the training data, the learning algorithm, or the predictions. These algorithm categories are known as pre-processing, in-processing, and post-processing, respectively. 
The bias mitigation algorithm categories are based on the location where these algorithms can intervene in a complete machine learning pipeline. If the algorithm is allowed to modify the training data, then pre-processing can be used. If it is allowed to change the learning procedure for a machine learning model, then in-processing can be used. If the algorithm can only treat the learned model as a black box without any ability to modify the training data or learning algorithm, then only post-processing can be used.

### Pre-processing algorithms
Pre-processing algorithms attempt to improve the fairness metrics by modifying the training data. 
The pre-processing algorithms include:
1. Reweighing: Reweighing generates weights for the training examples in each (group, label) combination differently to ensure fairness before classification. 
2. Optimized preprocessing: Optimized preprocessing learns a probabilistic transformation that edits the features and labels in the data with group fairness, individual distortion, and data fidelity constraints and objectives. 
3. Learning fair representations: Learning fair representations finds a latent representation that encodes the data well but obfuscates information about protected attributes. 
4. Disparate impact remover: Disparate impact remover edits feature values to increase group fairness while preserving rank-ordering within groups.

### In-processing algorithms
In-processing algorithms attempt to improve the fairness metrics by modifying the learning algorithm. 
1. Adversarial debiasing: Adversarial debiasing learns a classifier to maximize prediction accuracy and simultaneously reduce an adversarys ability to determine the protected attribute from the predictions. This approach leads to a fair classifier as the predictions cannot carry any group discrimination information that the adversary can exploit. 
2. Prejudice remover: Prejudice remover adds a discrimination-aware regularization term to the learning objective.

### Post-processing algorithm
Post-processing algorithms attempt to improve the fairness metrics by modifying the predictions. 
1. Equalized odds postprocessing:Equalized odds postprocessing solves a linear program to find probabilities with which to change output labels to optimize equalized odds.
2. Calibrated equalized odds postprocessing: Calibrated equalized odds postprocessing optimizes over calibrated classifier score outputs to find probabilities with which to change output labels with an equalized odds objective.
3. Reject option classification: Reject option classification gives favorable outcomes to unprivileged groups and unfavorable outcomes to privileged groups in a confidence band around the decision boundary with the highest uncertainty.

