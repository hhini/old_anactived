# 🏃‍♂️ SCD-ActiPredict AI: Community SCD Physical Inactivity Prediction System
### 社区主观认知下降（SCD）老年人身体活动不足风险智能预测与临床决策支持系统

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![GitHub repo](https://img.shields.io/badge/GitHub-hhini%2Fold__anactived-blue?logo=github)](https://github.com/hhini/old_anactived)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit%201.35%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)

---

## 📖 项目简介 (Project Overview)

随着全球人口老龄化的加剧，**主观认知下降（Subjective Cognitive Decline, SCD）**已成为阿尔茨海默病（AD）等痴呆疾病的超早期预警窗口。保持充足的**身体活动（Physical Activity）**能够显著延缓认知衰退、促进神经可塑性。然而，社区 SCD 老年人身体活动不足发生率居高不下，且往往受心理、生理、社会支持与社区环境等多维因素复杂交织影响。

**SCD-ActiPredict AI** 是一个面向社区卫生服务中心、全科医学科与公共卫生流调人员的临床级智能预测与决策支持系统。本项目基于权威老年医学流调队列数据，经贝叶斯超参数调优对比了 7 大主流机器学习算法，甄选出泛化性能优异的 **XGBoost 冠军模型**，并深度融合 **SHAP (Shapley Additive Explanations)** 可解释性人工智能技术与临床列线图（Nomogram），打造具备大厂产品级质感与科学严谨性的交互式网页应用。

---

## 🏆 核心算法性能 (Champion Model Performance)

在 7 大候选模型中，经过严谨的 5 折分层交叉验证与独立测试集盲测，**XGBoost** 模型综合辨别力（AUC）与特异度表现最佳，被确立为本系统的核心计算引擎：

| 模型算法 (Algorithm) | 测试集 AUC (Test AUC) | 95% 置信区间 (95% CI) | 准确率 (Accuracy) | 敏感度 (Sensitivity) | 特异度 (Specificity) | F1-Score | Brier 分数 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **XGBoost (冠军模型)** | **0.8120** | **[0.7274 - 0.8910]** | **72.41%** | **48.78%** | **85.33%** | **0.5556** | **0.1771** |
| 🥈 **LightGBM** | 0.8037 | [0.7081 - 0.8841] | 75.00% | 60.98% | 82.67% | 0.6329 | 0.1828 |
| 🥉 **Logistic Regression** | 0.8013 | [0.7118 - 0.8739] | 74.14% | 43.90% | 90.67% | 0.5455 | 0.1743 |
| 🔹 **Random Forest** | 0.7805 | [0.6894 - 0.8650] | 74.14% | 53.66% | 85.33% | 0.5946 | 0.1929 |
| 🔹 **K-Nearest Neighbors** | 0.7634 | [0.6731 - 0.8524] | 73.28% | 48.78% | 86.67% | 0.5634 | 0.1815 |
| 🔹 **Support Vector Machine** | 0.7366 | [0.6391 - 0.8280] | 70.69% | 53.66% | 80.00% | 0.5641 | 0.1880 |
| 🔹 **Decision Tree** | 0.6714 | [0.5735 - 0.7705] | 70.69% | 51.22% | 81.33% | 0.5526 | 0.2337 |

> **临床决策切点设定 (Clinical Cut-offs)**：
> - **普查筛查预警切点 ($P \ge 0.270$)**：高灵敏度优先，最大化捕捉社区隐匿性不活动老年人。
> - **专科强化干预切点 ($P \ge 0.500$)**：高特异度优先，针对高危衰弱个体实施跨学科定制运动处方。

---

## 🔬 核心筛选预测因子 (Core Screened Predictors)

经单因素假设检验、多重共线性（VIF）检验、10折交叉验证 LASSO 惩罚回归与多因素二元 Logistic 回归最终严格筛选出 **6 项核心独立特征**（包含 4 大危险因素与 2 大保护因素，全部 $P < 0.05$）：

```mermaid
graph LR
    subgraph 4大独立致病危险因素 (Risk Factors)
        R1["静态非益智类爱好<br/>aOR=5.87, P < 0.001"]
        R2["既往外伤跌倒病史<br/>aOR=4.66, P < 0.001"]
        R3["高体质指数 (BMI)<br/>aOR=1.31, P < 0.001"]
        R4["高龄增长 (Age)<br/>aOR=1.06, P = 0.038"]
    end
    subgraph 2大独立保护抵御因素 (Protective Factors)
        P1["家庭人均月收入阶梯<br/>aOR=0.62, P = 0.002"]
        P2["高血红蛋白浓度 (Hb)<br/>aOR=0.96, P = 0.002"]
    end
    subgraph 预测结局 (Outcome)
        OUT["社区SCD老年人<br/>身体活动不足高风险"]
    end
    R1 --> OUT
    R2 --> OUT
    R3 --> OUT
    R4 --> OUT
    P1 -.->|显著降低风险| OUT
    P2 -.->|显著降低风险| OUT
```

---

## ✨ 系统功能特性 (System Features)

本系统采用现代科技大厂级（Apple/Vercel Design System）设计理念，具备响应式排版与极速响应能力：

1. **🧮 临床智能风险计算器 (Clinical Risk Calculator)**
   - **4类典型病例预设**：一键装填“低危活力型”、“中危过渡型”、“高危共病型”、“超高危极度衰弱型”；
   - **交互式参数配置**：滑动条与单选卡片精细输入生理、心理、社会与社区环境 11 项参数；
   - **动态半圆风险仪表盘**：动态渲染风险百分比并显示当前危险等级与普查/专科双轨预警指示灯；
   - **单病例 SHAP 归因瀑布图**：可视化量化每个特征对当前个体风险预测的正向推高或负向拉低贡献值；
   - **反事实生活方式模拟器**：支持医生与患者互动调整自我效能感与社交频率，实时推演干预后风险降幅。

2. **🏆 七大算法竞技场 (Benchmark & Arena)**
   - 7大算法全维度指标横向对比表格；
   - ROC-AUC / Precision-Recall 双集曲线交互对比；
   - 六维雷达图（Radar Chart）立体展现模型平衡度；
   - 高清混淆矩阵大拼图与三阶风险分层率比图。

3. **🔬 特征归因与列线图 (XAI & Nomogram)**
   - 6大核心独立影响因素病理学与循证医学干预机制解析；
   - 筛选特征精选森林图与全局 SHAP 特征重要性条形图/蜂群分布图；
   - 纸质临床诊断列线图（Nomogram）方便床旁量化打分。

4. **📘 社区分级防控与干预指南 (Intervention Protocols & SOP)**
   - 社区“防-筛-控”三级分层管理网络；
   - 适老化微运动方案（八段锦、太极、"坐45动5分" Snack Exercise 零食运动法则）；
   - 基层医护人员三步闭环落地 SOP 操作指引。

---

## 🚀 本地快速启动 (Local Quickstart)

### 1. 克隆代码仓库
```bash
git clone https://github.com/hhini/old_anactived.git
cd old_anactived
```

### 2. 创建并激活虚拟环境 (推荐)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖包
```bash
pip install -r requirements.txt
```

### 4. 启动 Streamlit 应用
```bash
streamlit run app.py
```
> 应用将在浏览器中自动打开：`http://localhost:8501`。

---

## 🌐 部署至 Streamlit Community Cloud

本项目已完成 Streamlit Community Cloud 部署环境适配，可通过 GitHub 仓库一键免费部署：

### 第一步：初始化并推送到 GitHub 仓库
在本地项目根目录运行以下命令：
```bash
# 1. 初始化 Git 仓库
git init

# 2. 添加所有必要工程文件
git add app.py streamlit_app.py requirements.txt README.md .gitignore .streamlit/ models/ figures/

# 3. 提交至本地分支
git commit -m "feat: release SCD-ActiPredict AI production-grade web application"

# 4. 关联 GitHub 远程仓库 (用户仓库: hhini/old_anactived)
git remote add origin https://github.com/hhini/old_anactived.git

# 5. 推送至主分支
git branch -M main
git push -u origin main --force
```

### 第二步：在 Streamlit Community Cloud 创建应用
1. 访问 [Streamlit Community Cloud](https://share.streamlit.io/) 并登录您的 GitHub 账号；
2. 点击右上角 **"New app"**；
3. 填写部署信息：
   - **Repository**: `hhini/old_anactived`
   - **Branch**: `main`
   - **Main file path**: `app.py` 或 `streamlit_app.py`（两者均支持，完全互通）
4. 点击 **"Deploy!"**；
5. 系统将在 1~2 分钟内自动完成环境构建与依赖解析，生成公开访问的专属二级域名（如：`https://old-anactived.streamlit.app`）。

---

## 📂 项目结构说明 (Repository Structure)

```text
old_anactived/
├── .streamlit/
│   └── config.toml             # Streamlit 现代化高保真主题与服务配置
├── figures/                    # Web 应用交互可视化素材
├── models/                     # 预训练模型权重与归一化标尺
│   ├── best_model.pkl          # 最优调优 XGBoost 预测引擎
│   ├── standard_scaler.pkl     # 11 维特征标准化转换器
│   └── ...                     # 候选模型权重
├── results/                    # 模型评测横向排行榜数据 (08_multimodel_performance_comparison.csv)
├── app.py                      # 核心生产级 Streamlit Web 预测系统
├── streamlit_app.py            # Streamlit Cloud 兼容入口镜像文件
├── requirements.txt            # 云端运行依赖规范文件
├── .gitignore                  # Git 忽略配置
└── README.md                   # 详细中英文项目文档
```

---

## 🛡️ 鲁棒性与隐私安全保障 (Robustness & Security)

- **无缝回退兜底（Zero-Crash Fallback）**：系统内置了纯数学解析解与双曲 Sigmoid 参数化计算器。在任何云端依赖或跨平台反序列化异常场景下，系统均可自动平滑降级，确保 100% 稳定运行。
- **数据隐私保护**：系统完全在用户浏览器端或隔离沙箱中进行单病例实时内存推理，严禁向任何第三方服务器回传、存储或持久化患者的敏感生理与健康数据。

---

## 📄 许可证 (License)

本项目遵循 [MIT License](LICENSE) 许可协议。
仅供学术交流与科研使用。临床决策请结合全科医生线下综合面诊评估。
