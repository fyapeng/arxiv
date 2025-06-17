import os
import re
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from datetime import datetime, timezone, timedelta
import concurrent.futures

# --- 配置 ---
KIMI_API_KEY = os.environ.get("KIMI_API_KEY")
ARXIV_URL = "https://arxiv.org/list/econ/recent"
README_PATH = "README.md"
START_COMMENT = "<!-- ARXIV_PAPERS_START -->"
END_COMMENT = "<!-- ARXIV_PAPERS_END -->"

# --- Kimi API 客户端 ---
if KIMI_API_KEY:
    kimi_client = OpenAI(api_key=KIMI_API_KEY, base_url="https://api.moonshot.cn/v1")
else:
    print("错误：未找到 KIMI_API_KEY 环境变量。")
    kimi_client = None

def translate_with_kimi(text):
    if not kimi_client or not text or "暂无摘要" in text:
        return "翻译失败（API未配置或文本为空）"
    try:
        print(f"  > 正在翻译: '{text[:40].replace(os.linesep, ' ')}...'")
        response = kimi_client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system", "content": "你是一个专业的经济学领域翻译助手。请将以下英文内容准确、流畅地翻译成中文。"},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  > Kimi 翻译 API 调用失败: {e}")
        return "翻译失败"

def process_single_paper(paper_info, session):
    """处理单篇论文的函数（获取摘要并翻译）"""
    title = paper_info['title']
    print(f"-> 开始处理: {title}")

    # 获取摘要
    try:
        detail_response = session.get(paper_info['url'], timeout=20)
        detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
        abstract_block = detail_soup.find('blockquote', class_='abstract')
        abstract = abstract_block.text.replace('Abstract:', '').strip() if abstract_block else '暂无摘要'
    except Exception as e:
        print(f"  > 获取摘要失败 for {title}: {e}")
        abstract = '暂无摘要'
        
    paper_info['abstract'] = abstract

    # 并行翻译
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as translator_executor:
        title_future = translator_executor.submit(translate_with_kimi, title)
        abstract_future = translator_executor.submit(translate_with_kimi, abstract)
        
        paper_info['title_cn'] = title_future.result()
        paper_info['abstract_cn'] = abstract_future.result()

    return paper_info

def fetch_and_process_papers():
    """获取并处理 arXiv 的新论文"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    print(f"正在访问 arXiv 经济学最新论文页面: {ARXIV_URL}")
    response = session.get(ARXIV_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    # 检查今天是否有更新
    # arXiv 的服务器时间是 EST/EDT (美国东部时间), 我们需要和当前 UTC 时间对比
    h3_date = soup.find('h3')
    if not h3_date or 'entries for' not in h3_date.text:
        print("未找到日期标题，可能页面结构已更改。")
        return None
    
    # 简单的检查：如果 "recent" 页面没有 "new" 的文章标记，可能就没有新内容
    if not soup.find('span', class_='new'):
         print("页面上未发现'new'标记的文章，判断为今日无更新。")
         return None

    papers_to_process = []
    # 查找所有新论文的列表项
    dt_elements = soup.find_all('dt')
    for dt in dt_elements:
        # 只处理标记为 "new" 的论文
        if dt.find('span', class_='new'):
            dd = dt.find_next_sibling('dd')
            if not dd: continue

            title_div = dd.find('div', class_='list-title')
            authors_div = dd.find('div', class_='list-authors')
            
            paper_id_tag = dt.find('a', title='Abstract')
            if not paper_id_tag: continue
            paper_id = paper_id_tag.text.strip()
            
            title = title_div.text.replace('Title:', '').strip()
            authors = [a.text.strip() for a in authors_div.find_all('a')]
            url = f"https://arxiv.org/abs/{paper_id}"

            papers_to_process.append({
                'title': title,
                'authors': authors,
                'url': url
            })

    if not papers_to_process:
        print("今日无新论文更新。")
        return None

    print(f"发现了 {len(papers_to_process)} 篇新论文。开始并行处理...")
    
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

def generate_markdown(results):
    if not results:
        return "今日无新论文更新。"

    # CST (UTC+8)
    cst_time = datetime.now(timezone(timedelta(hours=8)))
    update_time_str = cst_time.strftime('%Y-%m-%d %H:%M:%S CST')

    title_list_parts = [f"*(Updated on: {update_time_str})*\n"]
    for i, res in enumerate(results):
        title_list_parts.append(
            f"{i+1}. **[{res['title']}]({res['url']})**<br/>{res['title_cn']}\n"
            f"   - *Authors: {', '.join(res['authors'])}*"
        )
    
    details_parts = ["\n---\n\n## 文章概览\n"]
    for res in results:
        details_parts.extend([
            f"### {res['title_cn']}",
            f"**[{res['title']}]({res['url']})**\n",
            f"**Authors**: {', '.join(res['authors'])}\n",
            f"**Abstract**: {res['abstract']}\n",
            f"**摘要**: {res['abstract_cn']}\n",
            "---"
        ])
    
    return "\n".join(title_list_parts) + "\n\n" + "\n".join(details_parts)

def update_readme(content):
    with open(README_PATH, 'r', encoding='utf-8') as f:
        readme_content = f.read()
    pattern = f"({re.escape(START_COMMENT)})(.*?)({re.escape(END_COMMENT)})"
    new_readme = re.sub(pattern, f"\\1\n{content}\n\\3", readme_content, flags=re.DOTALL)
    with open(README_PATH, 'w', encoding='utf-8') as f:
        f.write(new_readme)
    print("README.md 更新成功！")

if __name__ == "__main__":
    if not kimi_client:
        exit(1)
        
    papers_data = fetch_and_process_papers()
    if papers_data:
        markdown_output = generate_markdown(papers_data)
        update_readme(markdown_output)
    else:
        # 如果没有新论文，也更新一下提示信息
        cst_time = datetime.now(timezone(timedelta(hours=8)))
        update_time_str = cst_time.strftime('%Y-%m-%d %H:%M:%S CST')
        update_readme(f"*(Updated on: {update_time_str})*\n\n今日无新论文。")
