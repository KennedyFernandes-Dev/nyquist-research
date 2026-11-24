# Análise comparativa de estabilidade de sistemas dinâmicos

Ferramenta de pesquisa em Python para laços LIT contínuos SISO com realimentação unitária negativa. O projeto compara Nyquist, Routh–Hurwitz, lugar das raízes e Bode em quatro plantas, incluindo um modelo de atraso puro de transporte.

## Executar

Para reproduzir o ambiente medido, use Git, uv e CPython 3.12.14. O uv obtém o Python declarado em `.python-version` quando necessário. Cada comando deve ser executado a partir da raiz deste repositório.

~~~sh
uv sync --locked --all-extras --group security
uv run --locked streamlit run application/streamlit/app.py
~~~

A interface abre no endereço exibido pelo Streamlit, normalmente `http://localhost:8501`. Selecione uma das plantas A–D ou informe coeficientes em ordem decrescente. A interface aceita ganho e atraso em segundos, compara as análises e exporta os resultados. Os gráficos de frequência usam o atraso exponencial. O degrau e as análises polinomiais usam a ordem de Padé indicada.

## Arquitetura

| Caminho | Conteúdo |
| --- | --- |
| `src/nyquist_research/` | Função de transferência, Horner, contorno e envoltórios, Routh, lugar das raízes, Bode, Padé, atraso crítico e resposta ao degrau |
| `validation/` | Referências em SciPy, Python-control, SymPy, Mpmath e fórmulas analíticas |
| `experiments/` e `results/` | Configurações, scripts de execução, medições brutas, resumos, gráficos e tabelas |
| `application/streamlit/` | Interface científica em Streamlit e Plotly |
| `firmware/hil/` | Simulação do PID com atraso de comunicação e código para futura conexão a bancada |
| `paper/` | Fonte LaTeX e bibliografia do artigo em IEEE Conference |

O núcleo usa Python, biblioteca padrão e NumPy. O teste de importações verifica a separação entre o núcleo e as bibliotecas de validação.

## Resultados da validação

A suíte automatizada tem 103 testes. Os experimentos registram 129 verificações em 14 pontos das plantas A–D e 132 medições de desempenho nas ordens 5, 10 e 20. Consulte o [relatório de validação](results/processed/validation.json) e o [resumo dos benchmarks](results/processed/benchmark-summary.json).

## Plantas e interpretação

| Caso | Modelo de malha aberta sem o ganho nominal | Ganho nominal | Fenômeno |
| --- | --- | ---: | --- |
| A | `6/((s+1)(s+2)(s+3))` | 1 | Ordem 3, polos à esquerda |
| B | `1/((s-1)(s+2))` | 4 | Polo aberto no semiplano direito |
| C | `(1-s)/((s+1)(s+2))` | 1 | Zero no semiplano direito |
| D | `2 exp(-0.8 s)/(s+1)` | 1 | Atraso puro de 0,8 s |

Em B, Nyquist conta \(P=1\), \(N=-1\) e \(Z=0\). Em D, a primeira fronteira de atraso é \(2\pi/(3\sqrt{3})\approx1{,}2092\) s. As margens de Bode são associadas a cada cruzamento. A estabilidade absoluta segue a contagem de polos ou o resultado de Nyquist.

~~~python
from nyquist_research.nyquist import nyquist
from nyquist_research.transfer_function import TransferFunction

plant = TransferFunction([1], [1, 1, -2], gain=4)
result = nyquist(plant)
print(result.status, result.p, result.n, result.z)  # stable 1 -1 0
~~~

## Reproduzir os experimentos

~~~sh
uv run --locked python -m experiments.scripts.reproduce
~~~

A execução grava CSV e JSON em `results/raw/` e `results/processed/`. As figuras vetoriais e as tabelas ficam em `results/figures/` e `results/tables/`. Consulte a [metodologia experimental](docs/experimental-method.md) para tolerâncias, repetições, dados de alta precisão, sensibilidade em ordens 5, 10 e 20 e origem de cada figura. Tempos de CPU e de parede dependem da máquina e da carga. As comparações devem conservar as configurações informadas no estudo.

A reprodução da bancada simulada está descrita em [firmware/hil](firmware/hil/README.md). Os registros dessa trilha são saídas de simulação. A execução física depende de um ESP32 e do enlace serial.

## Verificar e compilar

~~~sh
uv run --locked pytest --cov=nyquist_research --cov-report=term-missing
uv run --locked ruff check src tests validation experiments application firmware
uv run --locked mypy
uv run --locked --group security bandit -q -r src application firmware
uv run --locked --group security pip-audit
uv run --locked python paper/build_tables.py
tectonic --keep-logs paper/main.tex
~~~

O artigo referencia as figuras vetoriais do repositório. Tectonic 0.17 compila `paper/main.pdf` com a classe oficial IEEEtran e BibTeX.

## Limitações e contribuições

Uma malha finita pode deixar de resolver uma passagem muito próxima do ponto crítico. Nyquist registra o diagnóstico quando a resolução ou a cota da cauda é insuficiente. Atraso bipróprio com ganho de passagem direta de módulo pelo menos unitário requer tratamento próprio. O uso de Padé é limitado à banda de interesse. Polos próximos e polinômios de alta ordem elevam o erro de localização das raízes. Os casos simulados com jitter ilustram o comportamento de uma discretização e não certificam estabilidade sob todos os atrasos variáveis.

As [referências verificadas](research/source-verification/references.md) documentam as fontes do artigo. O código do projeto está sob a licença [MIT](LICENSE).
