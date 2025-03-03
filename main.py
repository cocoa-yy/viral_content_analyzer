import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import openai
import os

# 初始化系统
st.set_page_config(layout="wide", page_title="爆款工厂")
st.session_state.setdefault("selected_cases", [])
COLD_START_DATA = Path(__file__).parent / "cold_start_data.json"

openai.api_key = "111"
# 设置 OpenAI API Key（从环境变量中读取）
# openai.api_key = os.getenv("OPENAI_API_KEY")
# if not openai.api_key:
#     st.error("请设置环境变量 OPENAI_API_KEY")
#     st.stop()

# 大模型分析函数
def analyze_with_llm(content, title, platform):
    prompt = f"""
    对以下内容进行爆款分析：
    标题：{title}
    平台：{platform}
    内容：{content}

    请生成一个爆款归因报告，包含：
    1. **内容结构得分（满分100）**：分析结构并给出得分和建议。
    2. **情绪设计图谱**：分析情绪设计。
    3. **可复用爆款公式**：根据平台提供创作公式。
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# 数据管理层
def load_cold_data():
    if COLD_START_DATA.exists():
        with open(COLD_START_DATA, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "cases": [
            {
                "id": 1,
                "title": "关税加征后，中国汽车出口为何暴增？",
                "platform": "微信",
                "metrics": {"reads": 100000, "likes": 8500},
                "content": "美国加征25%关税的背景下...",
                "tags": ["反直觉", "数据对比", "产业分析"],
                "added_time": "2023-08-20"
            }
        ]
    }

def save_cold_data(data):
    with open(COLD_START_DATA, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

# 核心功能模块
def find_hot_cases():
    st.header("🔍 爆款发现工坊")
    tab1, tab2, tab3 = st.tabs(["案例库浏览", "数据抓取", "人工添加"])

    with tab1:
        data = load_cold_data()
        df = pd.DataFrame(data["cases"])
        col1, col2 = st.columns(2)
        with col1:
            platform_filter = st.multiselect("平台筛选", ["微信", "B站", "微博"])
        with col2:
            sort_by = st.selectbox("排序方式", ["综合传播力", "最新添加"])

        # 搜索功能
        search_keyword = st.text_input("搜索关键词")
        if platform_filter:
            df = df[df["platform"].isin(platform_filter)]
        if search_keyword:
            df = df[df['title'].str.contains(search_keyword, case=False) | df['content'].str.contains(search_keyword, case=False)]
        if sort_by == "综合传播力":
            df["传播力"] = df["metrics"].apply(lambda x: x["reads"] * 0.6 + x["likes"] * 0.4)
            df = df.sort_values("传播力", ascending=False)

        st.dataframe(df[["title", "platform", "tags", "added_time"]], use_container_width=True, height=400)
        selected_indices = st.multiselect("选择案例编号", options=range(len(df)), format_func=lambda x: df.iloc[x]['title'])
        if st.button("确认选择"):
            st.session_state.selected_cases = [df.iloc[i].to_dict() for i in selected_indices]
            st.success(f"已选择 {len(selected_indices)} 个案例")

    with tab2:
        st.info("数据抓取功能开发中...")
    with tab3:
        with st.form("添加案例"):
            title = st.text_input("标题*")
            platform = st.selectbox("平台*", ["微信", "B站", "微博"])
            content = st.text_area("内容")
            if st.form_submit_button("添加到案例库"):
                new_case = {
                    "id": datetime.now().timestamp(),
                    "title": title,
                    "platform": platform,
                    "content": content,
                    "metrics": {"reads": 0, "likes": 0},
                    "tags": [],
                    "added_time": datetime.now().strftime("%Y-%m-%d")
                }
                data = load_cold_data()
                data["cases"].append(new_case)
                save_cold_data(data)
                st.success("案例添加成功！")

def analyze_case():
    st.header("🔬 爆款拆解实验室")
    if not st.session_state.selected_cases:
        st.warning("请先到「找爆款」选择案例")
        return

    cases = st.session_state.selected_cases
    for case in cases:
        st.subheader(f"分析对象：{case['title']}")

        # 仪表盘
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("阅读量", case["metrics"]["reads"])
        with col2:
            st.metric("点赞量", case["metrics"]["likes"])
        with col3:
            st.metric("传播力", case["metrics"]["reads"] * 0.6 + case["metrics"]["likes"] * 0.4)

        # 分析选项卡
        tab1, tab2 = st.tabs(["内容结构", "传播模式"])
        with tab1:
            st.subheader("内容基因分析")
            st.write("**高频词**：芯片 | 关税 | 国产化 | 逆袭")
            st.write("**情感倾向**：积极（0.72）")
        with tab2:
            st.subheader("传播路径模拟")
            st.line_chart({"阅读量": [1000, 8500, 24000, 100000]})

        # 爆款归因报告
        st.subheader("爆款归因报告")
        analysis_report = analyze_with_llm(case['content'], case['title'], case['platform'])
        st.markdown(analysis_report)

        # 存储分析结果
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = []
        st.session_state.analysis_results.append({
            "title": case['title'],
            "report": analysis_report
        })

def generate_article():
    st.header("✍️ 爆款应用车间")

    if not st.session_state.get("selected_cases"):
        st.error("请先选择参考案例")
        return

    cases = st.session_state.selected_cases
    st.subheader("参考爆款：")
    for case in cases:
        st.write(f"- {case['title']}")

    # 展示主要策略
    if "analysis_results" in st.session_state:
        st.subheader("主要策略：")
        for result in st.session_state.analysis_results:
            st.write(f"**{result['title']}**")
            st.markdown(result['report'])  # 暂时展示整个报告
    else:
        st.warning("未找到分析结果，请先进行拆解分析")

    # 用户输入
    theme = st.text_input("输入您的文章主题")
    background = st.text_area("输入背景信息（可选）")

    # 默认背景信息
    if not background and "analysis_results" in st.session_state:
        background = "示例关键词"  # 后续可解析报告提取关键词

    if st.button("生成初稿"):
        if not theme:
            st.error("请输入文章主题")
            return

        sample = f"""
## 生成稿：{theme}

**参考爆款**：{', '.join([case['title'] for case in cases])}

**背景信息**：{background}

**初稿内容**：
- 开头：采用参考爆款的冲突前置策略...
- 中段：结合用户主题和背景信息...
- 结尾：添加行动号召...
        """
        st.markdown(sample)

# 主界面导航
st.sidebar.title("导航")

st.sidebar.markdown("""
<style>
/* 侧边栏整体样式 */
section[data-testid="stSidebar"] {
    background-color: #f0f2f6;
}

/* 确保 radio 容器宽度充满侧边栏 */
section[data-testid="stSidebar"] div[data-testid="stRadio"] {
    width: 100%;
}

/* 美化 radio 按钮并设置宽度自适应 */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
    font-size: 18px;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 5px;
    display: flex; /* 使用 flex 布局 */
    align-items: center; /* 垂直居中 */
    width: 100%;
    box-sizing: border-box;
    white-space: nowrap; /* 防止换行 */
}

/* 鼠标悬停效果 */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    background-color: #e0e2e6;
}

/* 调整小圆点和文字间距 */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label input[type="radio"] {
    margin-right: 8px; /* 小圆点和文字之间的间距 */
}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", ["🔍 找爆款", "🔬 拆爆款", "✍️ 造爆款"])
if page == "🔍 找爆款":
    find_hot_cases()
elif page == "🔬 拆爆款":
    analyze_case()
else:
    generate_article()