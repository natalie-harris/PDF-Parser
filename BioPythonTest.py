from Bio import Entrez

# Always tell NCBI who you are
Entrez.email = "mzg857@vols.utk.edu"

# Define the search term
search_term = "spruce budworm outbreak"

# Use the eSearch function and parse the result to a handle
handle = Entrez.esearch(db="pubmed", term=search_term)

# Parse the results
record = Entrez.read(handle)

# The result contains several elements, but we are interested in the 'IdList' which is a list of publication IDs.
id_list = record["IdList"]

# Now we can fetch details of these publications using 'efetch' function
for pubmed_id in id_list:
    handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="xml")
    paper_records = Entrez.read(handle)
    handle.close()  # Close the handle as soon as we're done with it

    for paper in paper_records["PubmedArticle"]:
        title = paper["MedlineCitation"]["Article"]["ArticleTitle"]
        try:
            abstract = paper["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0]
        except KeyError:  # Not all papers have an Abstract
            abstract = "No abstract available"

        print(f"Title: {title}\nAbstract: {abstract}\n")
