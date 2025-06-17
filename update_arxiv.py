import os
import re
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from datetime import datetime
import pytz
import concurrent.futures

# --- 配置 (无变化) ---
KIMI_API_KEY = os.environ.get("KIMI_API_KEY")
ARXIV_URL = "https://arxiv.org/list/econ/recent"
README_PATH = "README.md"
START_COMMENT = "<!-- ARXIV_PAPERS_START -->"
END_COMMENT = "<!-- ARXIV_PAPERS_END -->"

# --- Kimi API 客户端 (无变化) ---
if KIMI_API_KEY:
    kimi_client = OpenAI(api_key=KIMI_API_KEY, base_url="https://api.moonshot.cn/v1")
else:
    print("错误：未找到 KIMI_API_KEY 环境变量。")
    kimi_client = None

# --- 其他辅助函数 (无变化) ---
def translate_with_kimi(text):
    if not kimi_client or not text or "暂无摘要" in text:
        return "翻译失败"
    try:
        print(f"  > 正在翻译: '{text[:40].replace(os.linesep, ' ')}...'")
        response = kimi_client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "system", "content": "你是一个专业的经济学领域翻译助手。请将以下英文内容准确、流畅地翻译成中文，注意只输出对应的翻译结果就好。"}, {"role": "user", "content": text}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  > Kimi 翻译 API 调用失败: {e}")
        return "翻译失败"

def process_single_paper(paper_info, session):
    title = paper_info['title']
    print(f"-> 开始处理: {title}")
    try:
        detail_response = session.get(paper_info['url'], timeout=20)
        detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
        abstract_block = detail_soup.find('blockquote', class_='abstract')
        abstract = abstract_block.text.replace('Abstract:', '').strip() if abstract_block else '暂无摘要'
    except Exception as e:
        print(f"  > 获取摘要失败 for {title}: {e}")
        abstract = '暂无摘要'
    paper_info['abstract'] = abstract
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as translator_executor:
        title_future = translator_executor.submit(translate_with_kimi, title)
        abstract_future = translator_executor.submit(translate_with_kimi, abstract)
        paper_info['title_cn'] = title_future.result()
        paper_info['abstract_cn'] = abstract_future.result()
    return paper_info

# --- 核心修改部分 ---
def fetch_and_process_papers():
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})

    print(f"正在访问 arXiv 经济学最新论文页面: {ARXIV_URL}")
    response = session.get(ARXIV_URL, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    # 1. 首先，精准定位到包含所有内容的核心 <dl> 容器
    articles_container = soup.find('dl', id='articles')
    if not articles_container:
        print("错误：在页面上未找到 id 为 'articles' 的 <dl> 容器，无法继续。")
        return None

    # 2. 从这个容器内部查找日期标题 h3
    h3_tag = articles_container.find('h3')
    if not h3_tag:
        print("错误：在 'articles' 容器内未找到 <h3> 日期标签。")
        return None
        
    date_text = h3_tag.text.strip()
    
    # 3. 使用更健壮的正则表达式从 'Tue, 17 Jun 2025' 格式中提取日期
    match = re.search(r'(\d{1,2})\s(\w{3})\s(\d{4})', date_text)
    if not match:
        print(f"错误：无法从标题 '{date_text}' 中提取完整的年月日信息。")
        return None
    
    day_from_page = match.group(1)
    month_str_from_page = match.group(2)
    year_from_page = match.group(3)
    
    # 4. 获取当前美国东部时间的日期进行比对
    eastern_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(eastern_tz)
    current_et_day = now_et.strftime('%-d') # 不带前导零的天
    current_et_month_str = now_et.strftime('%b') # 月份缩写，如 Jun
    current_et_year = now_et.strftime('%Y') # 四位数年份
    
    print(f"arXiv 页面日期: {day_from_page} {month_str_from_page} {year_from_page}")
    print(f"当前 ET 日期: {current_et_day} {current_et_month_str} {current_et_year}")
    
    # 5. 进行严格的年月日比对
    if not (day_from_page == current_et_day and month_str_from_page == current_et_month_str and year_from_page == current_et_year):
        print("日期不匹配，判断为今日无更新。")
        return None
        
    print("日期匹配成功，开始解析论文列表。")

    # 6. 从同一个 'articles' 容器中解析论文列表
    papers_to_process = []
    for dt in articles_container.find_all('dt'):
        dd = dt.find_next_sibling('dd')
        if not dd: continue
        title_div = dd.find('div', class_='list-title')
        authors_div = dd.find('div', 'list-authors')
        id_link_tag = dt.find('a', title='Abstract')
        if not id_link_tag: continue
        
        paper_id = id_link_tag.text.strip()
        title = title_div.text.replace('Title:', '').strip()
        authors = [a.text.strip() for a in authors_div.find_all('a')]
        url = f"https://arxiv.org/abs/{paper_id}"
        papers_to_process.append({'title': title, 'authors': authors, 'url': url})

    if not papers_to_process:
        print("日期匹配，但未解析到任何论文。")
        return None

    print(f"发现了 {len(papers_to_process)} 篇今日新论文。开始并行处理...")
    processed_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_paper = {executor.submit(process_single_paper, paper, session): paper for paper in papers_to_process}
        for future in concurrent.futures.as_completed(future_to_paper):
            try:
                result = future.result()
                if result:
                    processed_results.append(result)
                    print(f"✓ 处理完成: {result['title'][:60]}...")
            except Exception as exc:
                print(f"✗ 处理论文时出错: {exc}")
    return processed_results

# --- generate_markdown 和 update_readme (无变化) ---
def generate_markdown(results):
    if not results:
        return "今日无新论文更新。"
    eastern_tz = pytz.timezone('US/Eastern')
    update_time_str = datetime.now(eastern_tz).strftime('%Y-%m-%d ET')
    title_list_parts = [f"*(Updated on: {update_time_str})*\n"]
    for i, res in enumerate(results):
        title_list_parts.append(f"{i+1}. **[{res['title']}]({res['url']})**<br/>{res['title_cn']}\n   - *Authors: {', '.join(res['authors'])}*")
    details_parts = ["\n---\n\n## 文章概览\n"]
    for res in results:
        details_parts.extend([f"### {res['title_cn']}", f"**[{res['title']}]({res['url']})**\n", f"**Authors**: {', '.join(res['authors'])}\n", f"**Abstract**: {res['abstract']}\n", f"**摘要**: {res['abstract_cn']}\n", "---"])
    return "\n".join(title_list_parts) + "\n\n" + "\n".join(details_parts)

def update_readme(content):
    with open(README_PATH, 'r', encoding='utf-8') as f:
        readme_content = f.read()
    pattern = f"({re.escape(START_COMMENT)})(.*?)({re.escape(END_COMMENT)})"
    new_readme = re.sub(pattern, f"\\1\n{content}\n\\3", readme_content, flags=re.DOTALL)
    with open(README_PATH, 'w', encoding='utf-8') as f:
        f.write(new_readme)
    print("README.md 更新成功！")


# --- main 函数 (无变化) ---
if __name__ == "__main__":
    if not kimi_client:
        exit(1)
    papers_data = fetch_and_process_papers()
    if papers_data:
        markdown_output = generate_markdown(papers_data)
        update_readme(markdown_output)
    else:
        eastern_tz = pytz.timezone('US/Eastern')
        update_time_str = datetime.now(eastern_tz).strftime('%Y-%m-%d ET')
        update_readme(f"*(Updated on: {update_time_str})*\n\n今日无新发表的经济学论文。")
