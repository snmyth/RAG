from pypdf import PdfReader

reader = PdfReader("test.pdf")

text = ""
chunks = []

for page in reader.pages:
    text+= page.extract_text()
    while len(text)>=500:
        chunks.append(text[:500])
        text = text[500:]
if text:
    chunks.append(text)
print(len(chunks))