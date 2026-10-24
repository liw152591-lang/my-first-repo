# 基于 WOE 与逻辑回归的贷款风险评分卡

**阅读入口：** [完整分析 Notebook](credit_scorecard.ipynb) · [可运行脚本](credit_scorecard.py) · [数据说明](data/README.md) · [结果表](reports/tables) 

## 结果概览

在 58,645 条有标签记录中，按标签分层进行 80%/20% 划分，随机种子为 42。测试文件有 39,098 条无标签记录，仅用于生成预测。

| 集合 | 样本数 | 标签 1 占比 | AUC | KS |
|---|---:|---:|---:|---:|
| 训练集 | 46,916 | 14.24% | 0.9069 | 0.6664 |
| 验证集 | 11,729 | 14.24% | **0.9091** | **0.6820** |

最终模型使用 9 个 WOE 特征。

![验证集 KS](reports/figures/valid_ks.png)

![验证集分组事件率与 Lift](reports/figures/valid_decile.png)

第二张图采用**训练集十分位边界**，验证集各组人数不一定相等。最低分组事件率约 80.42%，Lift 约 5.65；最高分组事件率约 0.44%。

## 分析过程与主要选择

| 步骤 | 实际做法 
|---|---|---|
| EDA | 检查类型、缺失、重复、标签比例、数值及类别分布 
| 异常记录 | 保留原始样本，记录质量问题 
| 数据划分 | 分层随机划分，固定种子 
| WOE | 仅训练集拟合 scorecardpy 自动决策树分箱，其他集合应用同一规则 
| 相关性 | 移除 `loan_grade_woe` | 与利率 WOE 相关系数约 0.913，信息高度重叠 
| VIF / IV | 含截距的 VIF 诊断，并展示 IV 
| 变量筛选 | 向前加入、向后剔除的逐步逻辑回归，p 阈值均为 0.05 
| 建模 | L2 逻辑回归，C=1，lbfgs 
| 评分卡 | 分箱 WOE 与回归系数换算为分值
最终变量为收入、利率、贷款收入比、住房状态、贷款用途、工龄、历史违约标记、年龄和贷款金额。具体分箱见 [woe_bins.csv](reports/tables/woe_bins.csv)。

## 评分如何得到

令`p=P(loan_status=1)`，评分公式为：

```text
B = PDO / ln(2)
A = points0 + B × ln(odds0)
Score = A − B × ln[p / (1 − p)]
```

- `points0=600`，`odds0=1/20` 是坏/好比，`PDO=50`。
- 600 分对应模型违约概率约 4.76%；好/坏比翻倍，评分增加 50 分。
- 总分为基础分加各变量分箱分值，各项四舍五入为整数。

查看 [评分卡明细](reports/tables/scorecard.csv)。

## 模型评估和阈值划分

Notebook ：

- 五档边界仅由训练集确定。验证集中 E→A 事件率约为 55.76%、8.65%、3.97%、2.45%、0.74%。[评级表](reports/tables/valid_grades.csv)
- 固定训练集分箱后，分数 PSI 在 train/valid 间约 0.0011、train/test 间约 0.0004，说明当前分布差异较小。[PSI 表](reports/tables/psi.csv)


## 运行

已在 Python 3.10.21、pandas 2.3.0、scikit-learn 1.7.0、scorecardpy 0.1.9.7 环境复算。完整建模依赖： [requirements.txt](requirements.txt)。


## 目录

```text
├── README.md  
├── requirements.txt             #依赖
├── credit_scorecard.py           # 主流程代码
├── credit_scorecard.ipynb        # 分析说明 + 执行输出
├── data/README.md               # 数据来源、字段与口径
├── reports/
│   ├── figures/                 # 可展示的评估图
│   └── tables/                  # 指标、分箱、评分卡及运行记录
└── outputs/                     # 本地逐样本预测/分数/提交文件，Git 忽略
```


## 学习过程与参考

本项目为个人学习项目。核心建模流程参考公开资料完成。代码及流程仍然有许多不足之处！目前仍在更新学习中！

- [Kaggle 竞赛与数据](https://www.kaggle.com/competitions/playground-series-s4e10/data)
- [scorecardpy 官方仓库](https://github.com/ShichenXie/scorecardpy)
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [statsmodels Logit](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html)
- [风控评分卡流程与思路]https://www.bbbdata.com/text/58
以上为数据和参考资料。
