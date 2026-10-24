本项目使用 [Kaggle Loan Approval Prediction — Playground Series S4E10](https://www.kaggle.com/competitions/playground-series-s4e10/data)。官方说明训练集与测试集由模型基于源数据生成，属于合成竞赛数据，不是真实银行授信样本。

登录 Kaggle，在竞赛 Data 页面按页面要求获取文件，将 `train.csv`、`test.csv`、`sample_submission.csv` 放在本目录。前两份是运行所需；第三份提供提交格式参考。原始 CSV 默认不提交到 GitHub；使用数据应遵循来源页面的条款。

| 文件 | 行数 | 注释 |
|---|---:|---|
| train.csv | 58,645 | 有标签数据，内部再划分训练/验证集 |
| test.csv | 39,098 | 无标签竞赛预测集，不能计算本地 AUC/KS |
| sample_submission.csv | 39,098 | `id,loan_status` 提交格式示例 |


| 字段 | 
|---|---|
| id | 行标识，不参与建模 |
| person_age | 年龄 |
| person_income | 收入；不擅自推断币种 |
| person_home_ownership | 住房持有状态 |
| person_emp_length | 工作年限 |
| loan_intent | 贷款用途 |
| loan_grade | 贷款等级；与利率信息高度重叠，建模前移除 |
| loan_amnt | 贷款金额 |
| loan_int_rate | 贷款利率 |
| loan_percent_income | 贷款金额与收入的比例，原数据已有字段 |
| cb_person_default_on_file | 历史违约记录标记；不等同于本笔贷款的未来结果 |
| cb_person_cred_hist_length | 信用历史长度 |
| loan_status | 二元预测标签；本项目将 1 作为风险事件 |

