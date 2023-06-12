from requests_html import HTMLSession
from requests.exceptions import ConnectionError
import time

session = HTMLSession()

headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/113.0'}
base_url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/'
end_url = '/pdf'

file = open('pmc articles.txt', 'r')
ids = file.readlines()
for pmc in ids:
    try:
        pmcid = pmc.strip()
        time.sleep(1)
        request = session.get(base_url + pmcid + '/', headers=headers, timeout=5)
        pdf_url = 'https://www.ncbi.nlm.nih.gov' + request.html.find('a.int-view', first=True).attrs['href']
        time.sleep(1)
        request = session.get(pdf_url, stream=True)
        with open(pmcid + '.pdf', 'wb') as file:
            for chunk in request.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    except ConnectionError as err:
        out = open('ConnectionError_pmcids.txt', 'a')
        out.write(pmcid + '\n')