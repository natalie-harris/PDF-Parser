from Bio import Entrez
from requests_html import HTMLSession
from requests.exceptions import ConnectionError
import time

# Set your email (required by NCBI)
Entrez.email = "mzg857@vols.utk.edu"

# Define the search term
search_term = "spruce budworm outbreak"

# Search for PubMed IDs (PMIDs) of papers matching the search term
handle = Entrez.esearch(db="pubmed", term=search_term, retmax=100)
record = Entrez.read(handle)
handle.close()

# List of PubMed IDs (PMIDs)
pmids = record["IdList"]

# Fetch details for each PMID to get PMCID
pmcids = []
for pmid in pmids:
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
    text = handle.read()
    handle.close()
    
    # Extract PMCID from the fetched text
    prefix = "PMC - "
    lines = text.split("\n")
    for line in lines:
        if line.startswith(prefix):
            line = line[len(prefix):]
            pmcids.append(line)

print(pmcids)
# Note: Ensure that you do not exceed the rate limits set by NCBI for API requests.


# Now attempt scraping the pdfs using these pmcids
session = HTMLSession()

headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/113.0'}
base_url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/'
end_url = '/pdf'


for pmc in pmcids:
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
