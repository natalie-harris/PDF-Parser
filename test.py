import PyPDF2
pdf_reader = PyPDF2.PdfReader("Blais 1981.pdf")
num_pages = len(pdf_reader.pages)

# concat all text into pdf_text string
pdf_text = ''
for page_number in range(num_pages):
    page = pdf_reader.pages[page_number]
    pdf_text += page.extract_text()

print(pdf_text)