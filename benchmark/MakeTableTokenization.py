from transformers import AutoModel, AutoTokenizer

words = [
	"asymptomatique",
	"blépharorraphie",
	"bradycardie",
	"bronchographie",
	"bronchopneumopathie",
	"dysménorrhée",
	"glaucome",
	"IRM",
	"kystectomie",
	"neuroleptique",
	"nicotine",
	"poliomyélite",
	"rhinopharyngite",
	"toxicomanie",
	"vasoconstricteur",
]

mapping = {
    "BioMedTok/BPE-HF-NACHOS-FR-Morphemes" : "BPE-HF-NACHOS-FR-Morphemes",
    "BioMedTok/BPE-HF-PubMed-FR-Morphemes" : "BPE-HF-PubMed-FR-Morphemes",
    "BioMedTok/BPE-HF-CC100-FR-Morphemes" : "BPE-HF-CC100-FR-Morphemes",
    "BioMedTok/BPE-HF-Wikipedia-FR-Morphemes" : "BPE-HF-Wikipedia-FR-Morphemes",
    "BioMedTok/BPE-HF-NACHOS-FR" : "BPE-HF-NACHOS-FR",
    "BioMedTok/BPE-HF-PubMed-FR" : "BPE-HF-PubMed-FR",
    "BioMedTok/BPE-HF-CC100-FR" : "BPE-HF-CC100-FR",
    "BioMedTok/BPE-HF-Wikipedia-FR" : "BPE-HF-Wikipedia-FR",
    "BioMedTok/SentencePieceBPE-PubMed-FR-Morphemes" : "SentencePieceBPE-PubMed-FR-Morphemes",
    "BioMedTok/SentencePieceBPE-Wikipedia-FR-Morphemes" : "SentencePieceBPE-Wikipedia-FR-Morphemes",
    "BioMedTok/SentencePieceBPE-NACHOS-FR-Morphemes" : "SentencePieceBPE-NACHOS-FR-Morphemes",
    "BioMedTok/SentencePieceBPE-CC100-FR-Morphemes" : "SentencePieceBPE-CC100-FR-Morphemes",
    "BioMedTok/SentencePieceBPE-PubMed-FR" : "SentencePieceBPE-PubMed-FR",
    "BioMedTok/SentencePieceBPE-Wikipedia-FR" : "SentencePieceBPE-Wikipedia-FR",
    "BioMedTok/SentencePieceBPE-NACHOS-FR" : "SentencePieceBPE-NACHOS-FR",
    "BioMedTok/SentencePieceBPE-CC100-FR" : "SentencePieceBPE-CC100-FR",
}

results = {}

for word in words:

    if word not in results:
        results[word] = []

    for model_name in ["almanach/camemberta-base","camembert-base","flaubert/flaubert_base_uncased","Dr-BERT/DrBERT-7GB","microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext","xlm-roberta-base"]:

        print(mapping[model_name])

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenized_sentence = tokenizer.tokenize(word)
        tokenized_sentence = [t.replace("##","").replace("▁","").replace("</w>","") for t in tokenized_sentence if len(t.replace("##","").replace("▁","").replace("</w>","")) > 0]
        results[word].append(tokenized_sentence)

        print(tokenized_sentence)
        print()

text = ""

text += """
\\begin{table}[]
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{cccccccc}
\\hline
\\textbf{} &
\\multicolumn{3}{c}{\\textbf{French Generalist}} &
\\multicolumn{3}{c}{\\textbf{French Biomedical}} &
\\textbf{English Biomedical} \\\\
\\hline
\\textbf{Term} &
\\textbf{CamemBERTa} &
\\textbf{CamemBERT} &
\\textbf{FlauBERT} &
\\textbf{DrBERT} &
\\textbf{DrBERT CP PubMedBERT} &
\\textbf{CamemBERT-BIO} &
\\textbf{PubMedBERT} \\\\
\\hline
"""

for w in results:
    text += "\\textit{" + w + "} & " + " & ".join(["-".join(r) if len(r) > 1 else "\\checkmark" for r in results[w]]) + " \\\\ \n"

text += """
\\hline
\\end{tabular}%
}
\\end{table}
"""

print(text)
