# AI-For-Sustainability
End-to-end LLM pipeline to quantify the impact of AI towards SDGs



# HLD

Data

We have text files for all reports, converted via PyMuPDF and OCR.
Then for each text file we use spacy sentencizer for splitting into sentences

Extraction
Keywords 
AI : Use NIST 412 keywords
SDG : Use 169 + 17 SDG defns

Use sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, for computing sentencesxkeywords matrix of embeddings
Filter out based on a threshold



We need to quantify the impact of AI on achieving SDGs

If we do sentence/passage level split, we can then use classifications - 

all that mentions AI or SDG



-> Classify that have both AI and SDG

Normalize by total SDG mentions per report./ total passages, we can see that later. 

We classify each sentence into PAO, and Sentiment, and then see what's what?


# LLD
Data

We have text files for all reports, converted via PyMuPDF and OCR.
Then for each text file we use spacy sentencizer for splitting into sentences

Extraction
Keywords 
AI : Use NIST 412 keywords
SDG : Use 169 + 17 SDG defns

Use sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, for computing sentencesxkeywords matrix of embeddings
Filter out based on a threshold
Also use Fuzzy matching for keywords form both AI and SDG (need to create a bag od words from SDG definitions)

Second Filter. Classify each filtered sentence into AI/SDG True/False. 

Classification

Run all sentences having AI and SDG True from last filter and classify PAO/Sentiment

Then scoring based on this data


