import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 初始化系统
st.set_page_config(layout="wide", page_title="爆款工厂")
st.session_state.setdefault("selected_cases", [])
COLD_START_DATA = Path(__file__).parent / "cold_start_data-3.json"

# 设置硅基流动的 API 密钥和基础 URL
SILICONFLOW_API_KEY = "sk-lcbzpsmvbqftjlaivznvmhyxinatxgyibapndaxvdsaalhdz"
client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url="https://api.siliconflow.cn/v1"
)

# 加载自定义字体
font_path = Path(__file__).parent / "simhei.ttf"
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning("未找到 simhei.ttf 字体文件，中文可能无法正确显示！")


# 绘制雷达图的函数
def plot_radar_chart(scores, title="", color='blue'):
    labels = list(scores.keys())
    values = list(scores.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))  # 调整为较小尺寸以适应三列
    ax.fill(angles, values, color=color, alpha=0.25)  # 使用传入的颜色填充
    ax.plot(angles, values, color=color, linewidth=2)  # 使用传入的颜色绘制线条

    # 在雷达图上标注分数，移到图内
    for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
        # 计算向内的偏移位置，value - 10 表示往内移动10个单位
        inner_value = max(value - 10, 0)  # 确保不会移到负值区域
        ax.text(angle, inner_value, f'{int(value)}',
                ha='center', va='center',
                fontsize=10,  # 增大字体
                fontweight='bold',  # 加粗
                color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))  # 添加白色背景框

    ax.set_yticklabels([])  # 隐藏径向刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)  # 调整字体大小适应小图
    ax.set_title(title, fontsize=12, pad=10)  # 添加标题
    return fig

# 大模型分析函数
def analyze_with_llm(prompt, expect_json=False):
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to provide structured output when requested."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} if expect_json else None
        )
        content = response.choices[0].message.content
        return json.loads(content) if expect_json else content
    except Exception as e:
        st.error(f"API 调用失败: {e}")
        return None

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
        df["interaction_score"] = df["interaction_score"].apply(lambda x: int(x))

        col1, col2 = st.columns(2)
        with col1:
            platform_options = ["微信", "b站", "微博"]
            platform_filter = st.multiselect("平台筛选", platform_options, default=platform_options)
        with col2:
            sort_by = st.selectbox("排序方式", ["默认", "综合传播力", "最新发布", "最新添加"])

        search_keyword = st.text_input("搜索关键词")
        if platform_filter:
            df = df[df["platform"].isin(platform_filter)]
        if search_keyword:
            df = df[df['title'].str.contains(search_keyword, case=False) |
                    df['content'].str.contains(search_keyword, case=False) |
                    df['author'].str.contains(search_keyword, case=False)]

        if sort_by == "默认":
            df = df.sort_values("id", ascending=True)
        elif sort_by == "综合传播力":
            df = df.sort_values("interaction_score", ascending=False)
        elif sort_by == "最新发布":
            df = df.sort_values("publish_time", ascending=False)
        elif sort_by == "最新添加":
            df = df.sort_values("added_time", ascending=False)

        df_display = df.rename(columns={
            "id": "爆款编号", "title": "标题", "author": "作者", "interaction_score": "综合传播影响力",
            "content": "内容", "region": "地区", "theme": "主题", "publish_time": "发布时间",
            "platform": "平台", "link": "链接", "added_time": "添加时间",
        })

        st.dataframe(
            df_display[["爆款编号", "标题", "作者", "综合传播影响力", "内容", "地区", "主题", "发布时间", "平台", "链接", "添加时间"]],
            column_config={
                "标题": st.column_config.Column(width="medium"), "案例编号": st.column_config.Column(width="medium"),
                "作者": st.column_config.Column(width="small"), "互动得分": st.column_config.Column(width="small"),
                "内容": st.column_config.Column(width="large"), "地区": st.column_config.Column(width="small"),
                "主题": st.column_config.Column(width="small"), "发布时间": st.column_config.Column(width="medium"),
                "平台": st.column_config.Column(width="small"), "链接": st.column_config.Column(width="medium"),
                "添加时间": st.column_config.Column(width="medium"),
            },
            use_container_width=True,
            height=400
        )
        selected_index = st.selectbox("选择案例编号", options=range(len(df)), format_func=lambda x: df.iloc[x]['id'])
        if st.button("确认选择"):
            st.session_state.selected_cases = [df.iloc[selected_index].to_dict()]
            if "analysis_results" in st.session_state:
                del st.session_state.analysis_results
            st.success(f"已选择案例：{df.iloc[selected_index]['title']}")
            st.info("请点击侧边栏的「🔬 拆爆款」进入下一步分析")

    with tab2:
        st.info("数据抓取功能开发中...")
    with tab3:
        with st.form("添加案例"):
            existing_ids = [case["id"] for case in data["cases"] if isinstance(case["id"], str) and case["id"].startswith("VC")]
            new_id_num = max(int(id_str.replace("VC", "")) for id_str in existing_ids) + 1 if existing_ids else 1
            new_id = f"VC{new_id_num:04d}"
            title = st.text_input("标题*")
            platform = st.selectbox("平台*", ["微信", "B站", "微博"])
            content = st.text_area("内容*")
            link = st.text_area("链接")
            if st.form_submit_button("添加到案例库"):
                new_case = {
                    "id": new_id, "title": title, "author": "", "interaction_score": 0, "content": content,
                    "region": [], "theme": [], "publish_time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    "platform": platform, "link": link, "added_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

    case = st.session_state.selected_cases[0]
    st.subheader(f"分析对象：{case['title']}")
    st.write(f"链接：{case['link']}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("作者", case["author"])
    with col2:
        st.metric("综合传播影响力", case["interaction_score"])
    with col3:
        st.metric("发布平台", case["platform"])
    with col4:
        st.metric("发布时间", case["publish_time"][:10])

    st.subheader("爆款归因报告")

    if "analysis_results" not in st.session_state:
        with st.spinner("正在分析，请稍候...请耐心等待30秒左右"):
            prompt1 = f"""
            你是一个爆款内容分析专家。请根据以下信息分析该内容的爆款潜力，并为以下8个维度打分（0-100分）：
            - 主题匹配：内容主题与平台的适配度
            - 主要表达：视频的核心观点
            - 样态呈现：纪录片、知识分享、Vlog或是哪种样态，有没有特色或吸引力
            - 叙事角度：切入点、关联点
            - 叙事结构：是否采用吸引人的叙事方式（如案例教学、倒叙、冲突递进等）
            - 标题：关键词密度、情感强度、信息明确度、悬念/冲突设计 等
            - 内容要素：信息密度、梗密度、反转等
            - 情绪触发：共情、争议、爽感等点有多少，哪个位置

            输入信息：
            - 平台：{case['platform']}
            - 标题：{case['title']}
            - 作者：{case['author']}
            - 内容：{case['content']}
            - 发布时间：{case['publish_time'][0]}

            输出格式：{{
                "radar_scores": {{
                    "主题匹配": score,
                    "主要表达": score,
                    "样态呈现": score,
                    "叙事角度": score,
                    "叙事结构": score,
                    "标题": score,
                    "内容要素": score,
                    "情感触发": score
                }}
            }}
            只返回得分，不需要解释。
            """
            radar_result = analyze_with_llm(prompt1, expect_json=True)
            if radar_result is None:
                return
            radar_scores = radar_result["radar_scores"]

            prompt2 = f"""
            你是一个爆款内容分析专家。基于以下雷达图得分，选取得分最高的前3个维度，分析其具体爆款原因。每次分析需结合输入信息，围绕以下角度展开：
            - 主题匹配：内容主题与平台的适配度
            - 主要表达：视频的核心观点
            - 样态呈现：纪录片、知识分享、Vlog或是哪种样态，有没有特色或吸引力
            - 叙事角度：切入点、关联点
            - 叙事结构：是否采用吸引人的叙事方式（如案例教学、倒叙、冲突递进等）
            - 标题：关键词密度、情感强度、信息明确度、悬念/冲突设计 等
            - 内容要素：信息密度、梗密度、反转等
            - 情绪触发：共情、争议、爽感等点有多少，哪个位置

            输入信息：
            - 平台：{case['platform']}
            - 标题：{case['title']}
            - 作者：{case['author']}
            - 内容：{case['content']}
            - 发布时间：{case['publish_time'][0]}
            - 雷达图得分：{json.dumps(radar_scores)}

            输出格式：
            1. [维度名称]（得分：XX）
               分析原因，直接输出原因（150-200字）
            2. [维度名称]（得分：XX）
               分析原因，直接输出原因（150-200字）
            3. [维度名称]（得分：XX）
               分析原因，直接输出原因（150-200字）
            """
            detailed_analysis = analyze_with_llm(prompt2, expect_json=False)
            if detailed_analysis is None:
                return

            prompt3 = f"""
            你是一个爆款内容分析专家。基于以下详细拆解结果，总结该内容最突出的三个亮点，作为可复用的爆款经验。每条亮点需简洁（50-80字），并与前面的分析关联。

            输入信息：
            - 详细拆解结果：
              {detailed_analysis}

            输出格式：
            1. 亮点描述，直接输出分析结果（50-80字）
            2. 亮点描述，直接输出分析结果（50-80字）
            3. 亮点描述，直接输出分析结果（50-80字）
            """
            highlights = analyze_with_llm(prompt3, expect_json=False)
            if highlights is None:
                return

            st.session_state.analysis_results = {
                "radar_scores": radar_scores,
                "detailed_analysis": detailed_analysis,
                "highlights": highlights
            }

    analysis = st.session_state.analysis_results

    # 分三列展示雷达图
    st.write("### 爆款因素得分")
    col1, col2, col3 = st.columns(3)  # 三列布局

    # 宏观层面：主题匹配、主要表达
    macro_scores = {
        "主题匹配": analysis["radar_scores"]["主题匹配"],
        "主要表达": analysis["radar_scores"]["主要表达"]
    }
    with col1:
        st.write("**宏观层面**")
        fig_macro = plot_radar_chart(macro_scores, title="宏观层面得分", color='blue')  # 蓝色
        st.pyplot(fig_macro, use_container_width=True)

    # 中观层面：样态呈现、叙事角度、叙事结构
    meso_scores = {
        "样态呈现": analysis["radar_scores"]["样态呈现"],
        "叙事角度": analysis["radar_scores"]["叙事角度"],
        "叙事结构": analysis["radar_scores"]["叙事结构"]
    }
    with col2:
        st.write("**中观层面**")
        fig_meso = plot_radar_chart(meso_scores, title="中观层面得分", color='green')  # 绿色
        st.pyplot(fig_meso, use_container_width=True)

    # 微观层面：标题、内容要素、情绪触发
    micro_scores = {
        "标题": analysis["radar_scores"]["标题"],
        "内容要素": analysis["radar_scores"]["内容要素"],
        "情感触发": analysis["radar_scores"]["情感触发"]
    }
    with col3:
        st.write("**微观层面**")
        fig_micro = plot_radar_chart(micro_scores, title="微观层面得分", color='red')  # 红色
        st.pyplot(fig_micro, use_container_width=True)

    # 展示详细拆解和高分维度
    st.write("### 高分维度拆解")
    st.markdown(analysis["detailed_analysis"])

    st.write("### 亮点概括")
    st.markdown(analysis["highlights"])

    st.success("分析完成！")
    st.info("请点击侧边栏的「✍️ 造爆款」生成文章")

def generate_article():
    st.header("✍️ 爆款应用车间")
    if not st.session_state.get("selected_cases"):
        st.error("请先选择参考案例")
        return

    cases = st.session_state.selected_cases
    st.subheader("参考爆款：")
    for case in cases:
        st.write(f"- {case['title']}")

    if "analysis_results" in st.session_state:
        analysis = st.session_state.analysis_results
        st.subheader("主要策略：")
        st.markdown(analysis["highlights"])
    else:
        st.warning("未找到分析结果，请先进行拆解分析")

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
待开发！！！😊😊😊
        """
        st.markdown(sample)

# 主界面导航
st.sidebar.title("导航")
st.sidebar.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #f0f2f6;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] {
    width: 100%;
}
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
section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    background-color: #e0e2e6;
}
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