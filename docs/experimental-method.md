# Método experimental e rastreabilidade

## Reprodução

A partir da raiz do repositório, instale as dependências de análise e execute:

```bash
uv sync --locked --all-extras
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --locked python -m experiments.scripts.reproduce
```

O programa gera os arquivos CSV e JSON usados no artigo. Uma falha de validação retorna código diferente de zero e fica em `results/processed/validation.json`. Para reproduzir apenas os PDFs a partir dos dados salvos, execute `uv run --locked python -m experiments.scripts.figures`. Tempos, dispersão e características da execução variam conforme máquina e carga. O ambiente e as versões efetivas ficam em `results/raw/environment.json`.

## Hipóteses e plantas

As quatro funções são próprias, SISO e em tempo contínuo, sob realimentação unitária negativa. Os parâmetros nominais e os coeficientes em ordem decrescente vêm de `experiments/configurations/systems.json`. O ganho multiplica o numerador. A estabilidade em malha fechada refere-se ao posicionamento estrito de polos no semiplano esquerdo. Polos estritamente à direita dominam a classificação e o eixo imaginário é uma condição de fronteira.

| Planta | Modelo do laço | Hipótese e caso analítico |
| --- | --- | --- |
| A | 6K/[(s+1)(s+2)(s+3)] | Três dinâmicas estáveis em cascata. Routh dá 0 ≤ K < 10 para ganho não negativo. Em K=10, o par está em ±j√11 e o terceiro polo em −6. |
| B | K/[(s−1)(s+2)] | Modelo de posição linearizado com amortecimento positivo e rigidez efetiva negativa. A soma vetorial em realimentação desloca o termo constante de −2 para K−2. K>2 estabiliza. A fronteira está em DC e P=1 mesmo quando a malha fechada estabiliza. |
| C | K(1−s)/[(s+1)(s+2)] | Zero em +1 e polos em −1 e −2. O denominador em malha fechada tem coeficientes (1, 3-K, 2+K). Para K não negativo, a fronteira ocorre em K=3 com ±j√5. O termo dominante do numerador torna a resposta inicial inversa no nominal. |
| D | 2e^(−τs)/(s+1) | Dinâmica de primeira ordem com atraso em um canal de comunicação. Para ganho nominal, o módulo unitário ocorre em √3 rad/s e o primeiro atraso crítico é $2\pi/(3\sqrt{3})$ s. |

A, B e C têm pontos nominais, na fronteira e instáveis. Em D há pontos sem atraso, nominal, na primeira fronteira, após a primeira fronteira e após duas travessias de pares conjugados. Para D, $s+1+2e^{-\tau s}=0$ implica $s_k=W_k(-2\tau e^\tau)/\tau-1$. Há travessias em $\tau=(2\pi/3+2k\pi)/\sqrt{3}$, para k inteiro não negativo. A parte real de $ds/d\tau$ em cada travessia é $\omega^2/[(1+\tau)^2+(\omega\tau)^2]>0$. A contagem analítica fornece a referência global e os ramos $k=-10,\ldots,10$ de Lambert W confirmam os polos dominantes com avaliação em 65 dígitos.

## Oráculos e critérios

- A resposta racional é obtida com SciPy `signal.freqresp` e multiplicada de forma independente por $e^{-j\omega\tau}$ quando há atraso. A validação usa 1001 frequências de $10^{-3}$ a $10^2$ rad/s. O limiar absoluto e relativo é $10^{-10}$ para estas funções bem escaladas. A resposta complexa já testa conjuntamente módulo e fase.
- SymPy `nroots` resolve a versão racional reconstruída por coeficientes exatos com saída de 35 dígitos. Polos com $\lvert\operatorname{Re}s\rvert\le10^{-8}$ compõem o diagnóstico de eixo dos casos propositalmente marginais. P, N, Z e estado são comparados em valores inteiros ou categorias. Python-control serve como referência adicional de N nas plantas racionais fora da fronteira.
- Polos e zeros abertos são comparados com Python-control até $10^{-9}$. O Lugar das Raízes usa 101 ganhos em relação à planta sem multiplicador nominal, de K=0 a K=20. Em cada ganho é comparado o conjunto de polos de Python-control, pareado pelo custo total de distância euclidiana de SciPy, com tolerância máxima $2\times10^{-6}$ s⁻¹. As três fronteiras racionais também são confrontadas com seus valores algébricos em ganho e frequência. Para D, o lugar das raízes é do modelo racional Padé [3/3].
- O degrau usa 301 amostras uniformes de 0 a 15 s e é confrontado com SciPy `signal.step` no modelo de malha fechada de Python-control. A tolerância absoluta $2\times10^{-6}$ absorve diferenças entre os esquemas de propagação numérica. O degrau de D também usa Padé [3/3].
- As margens racionais são confrontadas com `control.stability_margins(returnall=True)`. A tolerância é $2\times10^{-6}$ rad/s na frequência e $2\times10^{-5}$ no valor em graus ou ganho linear. O ponto DC de A satisfaz módulo unitário por igualdade direta, e o ponto DC de B tem MG=0,5. Registram-se esses contatos separadamente das margens convencionais para frequência positiva. Para D, $\mathrm{MF}=120^\circ-180\tau\sqrt{3}/\pi$. A primeira MG é verificada com a raiz de $\arctan\omega+\tau\omega=\pi$, e as margens seguintes são listadas no arquivo nominal na banda avaliada.
- Padé [1/1], [2/2], [3/3] e [5/5] são verificados por produto cruzado contra Mpmath, que calcula a aproximação a partir da série de Taylor em 80 dígitos. A tolerância de coeficientes é $10^{-10}$. A varredura de erro complexo e de fase usa 801 frequências de 0,01 a 30 rad/s e resume a banda até 5 rad/s. O erro de fase usa desenrolamento da razão complexa. Os limiares aproximados de atraso são calculados pela abscissa espectral em um intervalo que envolve a primeira fronteira de D, e confrontados com o Padé e os polos de Python-control até $10^{-8}$ s. A referência do atraso exato é analítica com a mesma tolerância em segundos.
- Os limiares de comparação foram escolhidos acima do arredondamento de dupla precisão e abaixo das escalas experimentais que distinguem as fronteiras. Cada comparação publica observado, tolerância e resultado booleano.

Rotinas consultadas nas documentações oficiais incluem [Python-control 0.10.2](https://python-control.readthedocs.io/en/0.10.2/response.html), [margens de Python-control](https://python-control.readthedocs.io/en/stable/generated/control.stability_margins.html), [SciPy freqresp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqresp.html), [SciPy step](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.step.html), [SciPy linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) e [Mpmath 1.3.0](https://mpmath.org/doc/current/mpmath.pdf).

### Divergências e resolução

- Em D, a primeira execução a 5 s retornou diagnóstico inconclusivo com 192 pontos-base, cinco refinamentos globais, 7296 pontos de contorno e incremento angular máximo de 1,624 rad. Uma tentativa com 1024 pontos-base convergiu a P=0, N=4, Z=4, em acordo com $0{,}098403\pm j0{,}537326$ e $0{,}004429\pm j1{,}678639$ obtidos pela referência. O refinamento local introduzido em a33a71a permitiu ao padrão convergir em 1034 amostras, com incremento máximo de 0,665 rad. A malha de 1024 pontos-base concordou novamente. O arquivo `results/raw/nyquist-refinement-baseline.json` preserva o diagnóstico inicial.
- Em D a $1{,}01\tau_*$, a malha exata já tem um par à direita; Padé [1/1] e [2/2] ainda preveem estabilidade. [3/3] e [5/5] reproduzem a classificação nesse ponto. A diferença é um erro de aproximação do atraso, rastreado em `results/processed/pade-summary.json`.
- O cruzamento de ganho de A em DC não aparece na referência convencional de margem de fase. A igualdade de ganho estático é verificada à parte. A margem em ω>0 fica indisponível onde não há cruzamento. Para B, o cruzamento de fase em DC e a MG negativa em decibéis refletem a direção da variação de ganho para atingir a fronteira. A classificação nominal é verificada por Nyquist e Routh.
- Em ordem alta, resíduos normalizados próximos da precisão de máquina coexistiram com erros grandes nas posições dos polos. As raízes do polinômio de coeficientes arredondados foram recalculadas com Mpmath antes de atribuir a parcela adicional ao cálculo em dupla precisão.

## Escalabilidade e precisão

O caso de desempenho usa denominadores $\prod_{i=1}^n(s+a_i)$, com $a_i$ uniformemente espaçado em [0,4; 3,2] e numerador constante tal que $L(0)=0{,}4$. Cada fator cresce em módulo com ω, portanto $\lvert L(j\omega)\rvert\le0{,}4$. Como P=0, Nyquist deve obter N=Z=0, o que dá uma referência analítica de estabilidade em todas as três ordens. Os ensaios de sensibilidade comparam a família dispersa com polos entre −3,2 e −0,4 e a família agrupada entre −1,3 e −0,8. Os coeficientes são formados em 80 dígitos e arredondados uma vez para float64. Uma perturbação relativa de $10^{-10}$ é aplicada aos coeficientes de ordem inferior, com sinais de semente registrada. Cada tabela registra resíduo normalizado por $\sum_j \lvert a_j\rvert\lvert s\rvert^{n-j}$, erro absoluto máximo após pareamento, erro relativo normalizado pela maior magnitude de referência e deslocamento após perturbação. O condicionamento relativo de primeira ordem do polo simples r usa $\sum_j \lvert a_j\rvert\lvert r\rvert^{n-j}/\lvert r p'(r)\rvert$.

Para o benchmark, Bode recebe 1001 frequências e o Lugar das Raízes 48 ganhos. Os algoritmos têm dois aquecimentos e 11 repetições por método e ordem. São 132 amostras. O comando registra tempo de CPU por `process_time_ns` e parede por `perf_counter_ns`. Médias, medianas, desvios-padrão amostrais, mínimos e máximos vêm das amostras brutas. Uma execução adicional com `tracemalloc` estima pico de alocações rastreadas pelo Python. A métrica de memória não é RSS e pode omitir reservas de bibliotecas nativas. Preparação dos modelos, importações, escrita de arquivos, geração das figuras, interface e cálculo das referências não entram no cronômetro.

Horner custa $O(mn)$ para m frequências e polinômios de grau até n. O contorno pode adicionar pontos perto de polos, atraso e do ponto crítico. Routh preenche $O(n^2)$ posições no caso regular. O caminho de aritmética racional exata tem custos dependentes dos algarismos. O cálculo de raízes por companheira e o pareamento global do Lugar das Raízes custam até $O(n^3)$ por ganho nas rotinas densas utilizadas. Bode acrescenta a obtenção de raízes de polinômios de cruzamento e refinamento para atraso. As medições aqui contrastam cargas definidas de tarefas diferentes e não estimam um expoente assintótico com apenas três ordens.

## Dados e figuras

| Figura vetorial | Origem em `results/` |
| --- | --- |
| `nyquist-abcd.pdf` | `raw/nyquist-contours.csv` e `processed/plant-summary.json` |
| `bode-abcd.pdf` | `raw/frequency.csv` e `processed/plant-summary.json` |
| `root-locus-abcd.pdf` | `raw/root-locus.csv` e `processed/plant-summary.json` |
| `stability-margins.pdf` | `processed/plant-summary.json` |
| `delay-pade.pdf` | `raw/pade-frequency-error.csv`, `raw/pade-delay-sweep.csv`, `processed/pade-summary.json` |
| `scalability.pdf` | `processed/benchmark-summary.json`, derivado de `raw/benchmark-timings.csv` |
| `numerical-precision.pdf` | `processed/benchmark-summary.json`, derivado de `raw/precision.json` |
| `step-response.pdf` | `raw/step.csv` |
