# statkit 統計工具箱

**只靠 NumPy 的可重現統計工具，專為研究與論文設計。** 論文會用到的檢定、效果量、信賴區間和報告格式都有，
而且直接輸出 APA 格式。

所有 p 值、信賴區間和效果量，都在測試裡和 SciPy、statsmodels（誤差 1e-8 以下）以及 G\*Power 的公開數值逐項核對。
同一個種子（seed）在任何電腦上都得到一樣的結果。[English](README.md)

## 安裝

```bash
pip install git+https://github.com/Benjaminwz/statkit
```

需要 Python 3.9 以上、NumPy 1.22 以上。

## 快速開始

```python
import statkit as sk

對照組 = [5.1, 4.9, 6.2, 5.8, 5.5, 4.7, 5.3, 6.0]
實驗組 = [6.4, 7.1, 6.8, 7.5, 6.2, 7.9, 6.6, 7.2, 6.9]

結果 = sk.ttest_ind(對照組, 實驗組)      # 預設用 Welch t 檢定
print(結果.apa())
# t(14.78) = -5.84, p < .001, 95% CI [-2.07, -0.96], g = -2.69, 95% CI [-3.98, -1.36]
```

每個檢定都回傳一個結果物件：統計量、p 值、自由度、信賴區間、效果量（含精確的信賴區間）。
`結果.apa()` 是一行 APA 第 7 版格式，`print(結果)` 是完整報告，`結果.to_dict()` 可轉成字典存成 JSON。

## 有哪些功能

| 類別 | 函式 |
| --- | --- |
| 描述統計 | `describe`（平均、標準差、標準誤、中位數、四分位距、偏態、峰度、平均數信賴區間） |
| 參數檢定 | `ttest_ind`（Welch／Student）、`ttest_rel`（成對）、`ttest_1samp`、`anova_oneway`（一般／Welch）、`levene`（變異數同質性） |
| 無母數檢定 | `mannwhitneyu`、`wilcoxon`（精確或近似、含同分校正）、`kruskal` |
| 常態性檢定 | `shapiro`（Shapiro-Wilk） |
| 次數與比例 | `chi2_contingency`（卡方、Cramér's V）、`fisher_exact`、`proportion_ci`（Wilson、Clopper-Pearson 等） |
| 事後比較 | `tukey_hsd`（Tukey-Kramer、學生化全距分布）、`games_howell`（不假設變異數相等）、`dunn`（Kruskal-Wallis 之後用，含同分校正與 p 值調整）、`pairwise_ttests`；回傳 `PosthocResult` 表格 |
| 等效性檢定 | `tost_ind`、`tost_rel`、`tost_1samp`（兩個單尾檢定 TOST；報告 `1 - 2α` 信賴區間，所以結論和區間一定一致） |
| 相關 | `pearsonr`、`spearmanr`、`kendalltau`（含信賴區間） |
| 效果量 | `cohens_d`、`hedges_g`、`glass_delta`、`cohens_dz`、精確信賴區間（`hedges_g_ci` 等）、`cliffs_delta`、`prob_superiority` |
| 重抽樣 | `bootstrap_ci`（percentile、basic、BCa）、`bootstrap_diff_ci`（獨立或成對）、`permutation_test`（單尾／雙尾、成對、精確列舉、自訂統計量） |
| 多重比較校正 | `bonferroni`、`sidak`、`holm`、`holm_sidak`、`hochberg`、`benjamini_hochberg`、`benjamini_yekutieli`、`adjust_pvalues` |
| 檢定力與樣本數 | `power_ttest`、`sample_size_ttest`、`min_detectable_effect` |
| 統合分析 | `meta_analysis`（固定／隨機效果、Q、I²、tau²、預測區間） |
| 信度與一致性 | `cronbach_alpha`（含信賴區間）、`cohens_kappa`（可加權、含信賴區間）、`icc`／`icc_table`（Shrout & Fleiss 的六種組內相關係數，含信賴區間） |

## 幾個設計上的選擇

- 兩組比較**預設用 Welch t 檢定**：變異數相等時幾乎沒損失，不相等時仍然正確。要用 Student 請加 `equal_var=True`。
- 效果量的信賴區間是**精確的**（反推非中心 t 分布），不是大樣本近似；Hedges' g 用精確的 gamma 函數校正。
- 排序檢定在樣本小又沒有同分時用精確分布，其他情況用含同分校正與連續性校正的常態近似。
- 蒙地卡羅 p 值有「加一校正」，所以不會剛好是 0。
- 刻意不做迴歸、廣義線性模型、混合模型，這些請用 statsmodels。

## 指令列

```bash
statkit 對照組.csv 實驗組.csv --column score --format text
statkit describe data.csv --column score
statkit adjust 0.001 0.02 0.04 0.3 --method holm
statkit power --effect-size 0.5        # 80% 檢定力每組需要幾人
statkit posthoc data.csv --group-col group --value-col score --method tukey   # 或 games-howell、dunn、ttest
statkit tost a.csv b.csv --column score --low -0.5 --high 0.5                 # 成對資料加 --paired
statkit icc ratings.csv --columns rater1 rater2 rater3                        # 六種 ICC
```

`posthoc` 讀長格式 CSV（一列一個觀察值），`icc` 讀寬格式（一列一位受試者）。

Excel 匯出的 CSV（UTF-8 含 BOM）可以直接讀。

## 授權

MIT
