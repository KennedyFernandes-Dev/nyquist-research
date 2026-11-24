# Verificação bibliográfica

Consulta em 24 de novembro de 2026. As entradas do artigo correspondem às chaves de `paper/references.bib`. Uma ficha de catálogo serve para conferir a edição. As fontes técnicas abertas ou originais sustentam as afirmações sobre algoritmos e modelagem.

| Chave | Evidência consultada | Uso |
| --- | --- | --- |
| nyquist1932 | [Registro Wiley, DOI 10.1002/j.1538-7305.1932.tb02344.x](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1932.tb02344.x), pp. 126–147. Metadados conferidos no Crossref | Origem histórica da análise de regeneração por frequência |
| astrom2008 | [Sítio dos autores em Caltech, cap. 9](https://www.cds.caltech.edu/~murray/amwiki/Frequency_Domain_Analysis.html), seções 2–4, e [cap. 10](https://www.cds.caltech.edu/~murray/amwiki/PID_Control.html) | Convenção de Nyquist, margens, relações de Bode, PID |
| ogata2010 | [Catálogo público do IFS, ISBN 9788576058106](https://www.ifs.edu.br/images/licitacoes/Inexigibilidades_2024/SEI_23060.002179_2023_25_compressed.pdf), [Pearson, conteúdo da 5ª edição em inglês](https://www.pearson.com/en-us/subject-catalog/p/modern-control-engineering/P200000003521/9780133002256) | Bibliografia recomendada, cotejo de capítulos de estabilidade, lugar e resposta em frequência. O ano segue a edição indicada no enunciado e no catálogo IFS |
| nise2017 | [Catálogo do IFTM, 7ª edição, LTC](https://biblioteca.iftm.edu.br/Acervo/Detalhe/14784), sumário | Bibliografia recomendada, capítulos de estabilidade, lugar e resposta em frequência |
| routh1877 | [Digitalização da edição Macmillan](https://books.google.com/books/about/A_Treatise_on_the_Stability_of_a_Given_S.html?id=2EBHMU9J4HEC) | Contexto histórico da contagem algébrica |
| evans1950 | [Artigo original digitalizado](https://bwang-ccny.github.io/files/ME41100/Root_Locus_Method.pdf), pp. 66–69. [DOI conferido no Crossref](https://api.crossref.org/works/10.1109%2FT-AIEE.1950.5060121) | Lugar das raízes e variação paramétrica |
| bode1945 | [Edição original digitalizada](https://jontalle.web.engr.illinois.edu/Public/BOOKS/Bode-NetworkAnalysisFeedbackAmplifierDesign.pdf) e [catálogo Smithsonian](https://www.si.edu/object/siris_sil_245712) | Fundamentação histórica da representação de magnitude e fase |
| baker1996 | [Cambridge University Press, DOI 10.1017/CBO9780511530074](https://www.cambridge.org/core/books/pade-approximants/A98B8DD2AFC0AF094EB6BF96C5C77A06), caps. 1–3 | Definição e propriedades de aproximantes racionais |
| wilkinson1959 | [Registro Crossref, DOI 10.1007/BF01386381](https://api.crossref.org/works/10.1007%2FBF01386381), pp. 150–166 | Condicionamento de raízes em relação aos coeficientes |
| zhang2001 | [Artigo original arquivado na Purdue](https://engineering.purdue.edu/~zak/networked_systems.pdf) e [registro Crossref](https://api.crossref.org/works/10.1109%2F37.898794) | Atraso de comunicação em sistemas de controle em rede |

[IEEE Author Center](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/) e [modelo IEEE Conference](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn) fixam a classe `IEEEtran`, modo `conference`. [Python-control](https://python-control.readthedocs.io/en/latest/generated/control.nyquist_response.html) documenta a convenção do contorno de validação. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.step.html) documenta a resposta temporal usada como referência.


A declaração de uso de IA segue as orientações de transparência do [IEEE Author Center](https://magazines.ieeeauthorcenter.ieee.org/get-started-with-ieee-magazines/publishing-ethics/guidelines-policies/submission-and-peer-review-policies/). A [comunicação institucional da PUC Goiás](https://impressoes.pucgoias.edu.br/queda-de-braco/) também descreve a declaração quando a ferramenta tem influência relevante e a revisão humana antes da submissão.
