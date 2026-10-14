# RAG Evaluation Quality Report

## 1. Failure classification

| category          |   count |
|:------------------|--------:|
| OK                |      14 |
| OVER_REFUSAL      |       5 |
| REVIEW_MANUALLY   |       3 |
| DATASET_ERROR     |       1 |
| INCOMPLETE_ANSWER |       1 |


## 2. Ragas averages

| index                                |   mean |
|:-------------------------------------|-------:|
| llm_context_precision_with_reference |  0.8   |
| context_recall                       |  1     |
| faithfulness                         |  0.72  |
| answer_relevancy                     |  0.824 |

NaN cells: 1

## 3. DeepEval averages

| index        |   mean |
|:-------------|-------:|
| precision    |  0.875 |
| recall       |  1     |
| faithfulness |  1     |
| relevancy    |  1     |

Rows scored: 5

## 5. Lowest scoring rows

| id   |   faithfulness |
|:-----|---------------:|
| Q004 |            0   |
| Q005 |            0.6 |
| Q001 |            1   |


## 6. Known limitations

- Scenario `inj_link` not run (out of memory on the test machine).

- Small local model used for generation in scenarios; judges used a stronger model.
