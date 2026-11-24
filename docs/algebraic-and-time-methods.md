# Métodos algébricos e resposta ao degrau

Considere realimentação unitária negativa e polinômios \(A(s)\) e \(B(s)\) com coeficientes reais em ordem decrescente. A característica racional é \(p_K(s)=A(s)+KB(s)\).

## Routh-Hurwitz

As duas primeiras linhas recebem os coeficientes alternados de \(p_K\). A eliminação que gera as demais linhas usa a diferença entre produtos cruzados e o pivô da linha anterior. Cada mudança de sinal na primeira coluna indica uma raiz no semiplano direito. A rotina retira fatores exatos \(s^m\) antes de montar a tabela e inclui suas multiplicidades no diagnóstico do eixo imaginário.

Cancelamento numérico ou pivô nulo aciona o recálculo com frações exatas dos dados binários de entrada. No caso de uma linha nula, a derivada do polinômio auxiliar substitui a linha. Um pivô isolado nulo é substituído por um \(\varepsilon>0\) proporcional à escala da linha. O algoritmo compara contagens para \(\varepsilon\) normalizado de \(10^{-24}\) e \(10^{-48}\). Se os sinais implicam contagens diferentes, o resultado é indeterminado. A concordância entre as duas avaliações é uma verificação numérica de sensibilidade.

Para localizar pares no eixo, o algoritmo usa as partes par e ímpar em \(x=s^2\), de modo que \(p_K(s)=E(x)+sO(x)\). Toda raiz comum real negativa em \(x\) dá um par \(s=\pm j\sqrt{-x}\). O máximo divisor comum de \(E\) e \(O\) e a sequência de Sturm contam as raízes negativas com multiplicidade. Uma linha nula pode aparecer também em um fator com raízes reais simétricas. A tabela flutuante usa \(O(n^2)\) operações e memória. O custo do recálculo depende também do tamanho em bits das frações.

## Lugar das raízes

Para cada ganho da malha, o núcleo calcula os autovalores da matriz companheira de \(p_K\) com numpy.roots. Uma atribuição húngara implementada no núcleo minimiza a soma das distâncias quadráticas entre os polos de ganhos adjacentes. Pontos múltiplos podem admitir mais de uma rotulação igualmente válida. Para \(g\) ganhos e ordem \(n\), o modelo usual para a geração e associação das raízes é \(O(gn^3)\). A resolução dos ramos depende do espaçamento dos ganhos. Eventos de redução do grau identificam ramos que deixam o plano finito.

Os candidatos a cruzamento finito não dependem da malha de ganhos. No eixo, o ganho \(K=-A(j\omega)/B(j\omega)\) deve ser real e não negativo. Assim, a parte imaginária de \(A(j\omega)B(-j\omega)\) precisa ser zero. As potências ímpares de \(A(s)B(-s)\) fornecem, após extrair \(\omega\), um polinômio real em \(\omega^2\). O núcleo resolve esse polinômio, testa separadamente \(\omega=0\) e verifica o resíduo na característica. As tolerâncias numéricas podem juntar frequências muito próximas ou perder raízes repetidas mal condicionadas. Se a equação de fase é identicamente nula, o resultado informa a possibilidade de uma faixa contínua de ganhos com raízes sobre o eixo.

## Resposta ao degrau

A função de transferência em malha fechada é \(KB(s)/p_K(s)\). Uma realização companheira da parte própria e um termo de passagem direta descrevem a saída. O estado leva uma coordenada constante para representar a entrada unitária. A transição em cada intervalo \(\Delta t\) é a exponencial matricial do gerador aumentado multiplicado pelo passo. O algoritmo escala sua norma, soma a série de Taylor e desfaz a escala com quadraturas sucessivas. Um passo repetido reutiliza sua transição.

O degrau atrasado usa o modelo racional de Padé indicado na chamada. A aproximação pode gerar resposta antes do tempo morto do modelo exponencial. O método temporal informa laços algébricos singulares e valores que excedam a escala numérica.

## Verificação

Os [testes algébricos](../tests/unit/test_algebra.py) verificam a tabela clássica, pivô zero, linhas nulas, multiplicidade no eixo e invariância a escalas binárias. Para o lugar das raízes, verificam custo mínimo de associação, cruzamento fora da malha, tangência e redução de grau. Para o degrau, comparam soluções analíticas, entrada direta, um polo repetido e uma referência SciPy usando coeficientes de Padé gerados pelo Python-control.
