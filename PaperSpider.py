from bs4 import BeautifulSoup
import requests
from translate import Translator

# https://www.sciencedirect.com/journal/remote-sensing-of-environment/vol/226/suppl/C\\

# Url_Top1 = "https://www.sciencedirect.com/journal/remote-sensing-of-environment/vol/220/suppl/C"
# Url_Top1 = "https://www.sciencedirect.com/search/advanced?qs=urban&pub=Remote%20Sensing%20of%20Environment&cid=271745&show=100&sortBy=date"
# Url_Top1 = "https://www.sciencedirect.com/search/advanced?qs=urban&pub=Remote%20Sensing%20of%20Environment&cid=271745&show=100&sortBy=date&offset=100"
Url_Top1 = "https://www.sciencedirect.com/search/advanced?pub=Remote%20Sensing%20of%20Environment&cid=271745&qs=urban&show=100"
# Url_IEEE = 'https://ieeexplore.ieee.org/xpl/mostRecentIssue.jsp?filter=issueId%20EQ%20%224358825%22&rowsPerPage=100&pageNumber=1&resultAction=REFINE&resultAction=ROWS_PER_PAGE&isnumber=4358825'

headers = {
    'User-Agent':
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36"
}

r = requests.get(Url_Top1, headers=headers)
html = r.text
# print(html)
soup = BeautifulSoup(html, 'html.parser')

div_items_Top1 = soup.find_all(
    # 'span', class_='js-article-title')  # Remote Sensing of Environment
    'a',
    class_='result-list-title-link u-font-serif text-s'
)  # Remote Sensing of Environment search

# div_items_IEEE = soup.find_all('a', class_='art-abs-url') # IEEE Transactions on Geoscience and Remote Sensing

print(len(div_items_Top1))

nameArr = []
for div in div_items_Top1:
    name = div.get_text().replace('\n', '') + '\n'
    print(name)
    name_zh = None
    translator = Translator(to_lang="zh")
    name_zh = translator.translate(name)
    print(name_zh)
    nameArr.append(name + '|' + name_zh + '\n')
    # break

open('PaperName.txt', 'w', encoding='utf-8').writelines(nameArr)
