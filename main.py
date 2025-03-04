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
COLD_START_DATA = Path(__file__).parent / "cold_start_data-3.json"

openai.api_key = "111"


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
                "id": "示例",
                "title": "#中方对美国部分进口商品加征关税#",
                "author": "韩路",
                "interaction_score": 51.3492063492063,
                "content": "中国反制加税来了，大排量汽车+10%。 幸亏F-150猛禽我买完了。 注：2.5升以上（不包含2.5）就属于大排量。 #中方对美国部分进口商品加征关税#\n\n",
                "region": ["中国", "美国"],
                "theme": ["经济-关税"],
                "publish_time": ["2025-02-04 13:29:03"],
                "platform": "微博",
                "link": "https://www.weibo.com/1192966660/PcIKnBqdT",
                "added_time": "2025-03-04 16:02:00"
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

        # 将 interaction_score 转换为整数
        df["interaction_score"] = df["interaction_score"].apply(lambda x: int(x))
        # 将 publish_time 列表转换为单个字符串（取第一个值）
        # df["publish_time"] = df["publish_time"].apply(lambda x: x[0] if x else None)
        # df["added_time"] = df["added_time"].apply(lambda x: x[0] if x else None)

        col1, col2 = st.columns(2)
        with col1:
            platform_options = ["微信", "b站", "微博"]
            platform_filter = st.multiselect("平台筛选", platform_options, default=platform_options)
            # region_input = st.text_input("输入地区（用逗号分隔）", value="美国,中国,墨西哥,欧洲,加拿大")
            # theme_input = st.text_input("输入主题（用逗号分隔）", value="经济,政治,文化,军事")
            # region_filter = [r.strip() for r in region_input.split(",")]
            # theme_filter = [t.strip() for t in theme_input.split(",")]
            # 处理平台筛选逻辑
        with col2:
            sort_by = st.selectbox("排序方式", ["默认", "综合传播力", "最新发布", "最新添加"])

        # 搜索功能（从 title 和 content 检索）
        search_keyword = st.text_input("搜索关键词")
        if platform_filter:
            df = df[df["platform"].isin(platform_filter)]
        # if region_filter:
        #     df = df[df["region"].apply(lambda x: any(region in x for region in region_filter))]
        # if theme_filter:
        #     df = df[df["theme"].apply(lambda x: any(theme in x for theme in theme_filter))]
        if search_keyword:
            df = df[df['title'].str.contains(search_keyword, case=False) | df['content'].str.contains(search_keyword,
            case=False)| df['author'].str.contains(search_keyword,case=False)]

        # 排序
        if sort_by == "默认":
            df = df.sort_values("id", ascending=True)
        elif sort_by == "综合传播力":
            df = df.sort_values("interaction_score", ascending=False)
        elif sort_by == "最新发布":
            df = df.sort_values("publish_time", ascending=False)
        elif sort_by == "最新添加":
            df = df.sort_values("added_time", ascending=False)

        # 修改表头为中文并调整列宽
        df_display = df.rename(columns={
            "id": "爆款编号",
            "title": "标题",
            "author": "作者",
            "interaction_score": "综合传播影响力",
            "content": "内容",
            "region": "地区",
            "theme": "主题",
            "publish_time": "发布时间",
            "platform": "平台",
            "link": "链接",
            "added_time": "添加时间",
        })

        st.dataframe(
            df_display[["爆款编号", "标题", "作者", "综合传播影响力", "内容", "地区", "主题", "发布时间", "平台", "链接", "添加时间"]],
            column_config={
                "标题": st.column_config.Column(width="medium"),  # 设置“标题”列宽度为大
                "案例编号": st.column_config.Column(width="medium"),
                "作者": st.column_config.Column(width="small"),
                "互动得分": st.column_config.Column(width="small"),
                "内容": st.column_config.Column(width="large"),
                "地区": st.column_config.Column(width="small"),
                "主题": st.column_config.Column(width="small"),
                "发布时间": st.column_config.Column(width="medium"),
                "平台": st.column_config.Column(width="small"),
                "链接": st.column_config.Column(width="medium"),
                "添加时间": st.column_config.Column(width="medium"),
            },
            use_container_width=True,
            height=400
        )
        # 单选案例
        selected_index = st.selectbox("选择案例编号", options=range(len(df)), format_func=lambda x: df.iloc[x]['id'])
        if st.button("确认选择"):
            st.session_state.selected_cases = [df.iloc[selected_index].to_dict()]
            selected_title = df.iloc[selected_index]['title']
            st.success(f"已选择案例：{selected_title}")

    with tab2:
        st.info("数据抓取功能开发中...")
    with tab3:
        with st.form("添加案例"):
            # 获取现有案例的最大 id
            existing_ids = [case["id"] for case in data["cases"] if
                            isinstance(case["id"], str) and case["id"].startswith("VC")]
            if existing_ids:
                max_id_num = max(int(id_str.replace("VC", "")) for id_str in existing_ids)
                new_id_num = max_id_num + 1
            else:
                new_id_num = 1  # 如果没有现有 id，从 VC0001 开始
            new_id = f"VC{new_id_num:04d}"  # 格式化为 VC0001, VC0002 等
            title = st.text_input("标题*")
            platform = st.selectbox("平台*", ["微信", "B站", "微博"])
            content = st.text_area("内容*")
            link = st.text_area("链接")
            if st.form_submit_button("添加到案例库"):
                new_case = {
                    "id": new_id,
                    "title": title,
                    "author": "",
                    "interaction_score": 0,
                    "content": content,
                    "region": [],
                    "theme": [],
                    "publish_time": "",
                    "platform":platform,
                    "link": link,
                    "added_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

        # 仪表盘（根据新数据结构调整）
        col1, col2 = st.columns(2)
        with col1:
            st.metric("互动得分", case["interaction_score"])
        with col2:
            st.metric("发布平台", case["platform"])

        # 分析选项卡
        tab1, tab2 = st.tabs(["内容结构", "传播模式"])
        with tab1:
            st.subheader("内容基因分析")
            st.write("**高频词**：芯片 | 关税 | 国产化 | 逆袭")
            st.write("**情感倾向**：积极（0.72）")
        with tab2:
            st.subheader("传播路径模拟")
            st.line_chart({"互动得分": [10, 20, 50, case["interaction_score"]]})

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
            st.markdown(result['report'])
    else:
        st.warning("未找到分析结果，请先进行拆解分析")

    # 用户输入
    theme = st.text_input("输入您的文章主题")
    background = st.text_area("输入背景信息（可选）")

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
    display: flex;
    align-items: center;
    width: 100%;
    box-sizing: border-box;
    white-space: nowrap;
}

/* 鼠标悬停效果 */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    background-color: #e0e2e6;
}

/* 调整小圆点和文字间距 */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label input[type="radio"] {
    margin-right: 8px;
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