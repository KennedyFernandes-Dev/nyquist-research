# PID com atraso de comunicação

A evidência desta pasta é uma **simulação de bancada em software**, executada com o mesmo cálculo discreto do PID preparado para o microcontrolador. O ESP32 e o adaptador serial não estavam conectados durante os experimentos.

## Modelo e reprodução

A planta simulada é \(P(s)=1/(s+1)\). O controlador contínuo de referência é \(C(s)=K_p+K_i/s+K_ds/(1+T_fs)\), com \(K_p=2\), \(K_i=1\), \(K_d=0{,}1\) e \(T_f=0{,}04\) s. A entrada recebe os comandos após um atraso na trajetória controlador–atuador. O sensor é instantâneo no modelo. O firmware usa os mesmos parâmetros e a mesma lei discreta da simulação.

O estado da planta é atualizado pela solução analítica \(x_{k+1}=e^{-T}x_k+(1-e^{-T})u_k\) sob retenção de ordem zero. O erro também é mantido constante na atualização do estado do PID. A simulação usa referência unitária, passo de 20 ms (50 Hz), confirmação a 10 ms (100 Hz) e duração de 90 s por cenário. O atraso é arredondado para o próximo instante da malha de tempo. Nos casos variáveis, sorteia-se uma perturbação uniforme em ±100 ms com semente 20261124. Quando um comando chega fora da ordem após outro mais recente, ele é descartado. Comandos válidos ficam retidos até a atualização seguinte.

```sh
uv sync --locked --extra analysis
uv run --locked --extra analysis python -m firmware.hil.simulate
```

A execução escreve [amostras temporais](../../results/raw/hil-timeseries.csv), [indicadores por cenário](../../results/processed/hil-summary.json) e [envoltórias](../../results/figures/hil-pid.pdf). O resumo compara o RMS do erro nos dois últimos intervalos de 10 s. Registra acomodação se o RMS final for menor que \(10^{-7}\), crescimento se o RMS final superar 1,2 vez o anterior, decaimento se for menor que 0,8 vez, e persistência no restante. A figura usa máximos de erro absoluto por janelas não sobrepostas de três segundos.

Nyquist e o cruzamento de ganho do modelo contínuo dão 1,05906 s como a primeira fronteira de atraso constante, com \(\omega=1{,}78789\) rad/s. Abaixo e acima desse atraso, as trajetórias simuladas exibiram, respectivamente, acomodação e crescimento. A repetição constante a 100 Hz preservou esse sentido. Com jitter a comparação por Nyquist se refere ao atraso constante nominal. A estabilidade do sistema variante no tempo não é inferida desse critério. A simulação sem saturação mede crescimento em um modelo linear e pode produzir comandos muito acima da faixa de um atuador real.

## Firmware preparado para bancada

1. Use o firmware MicroPython para um ESP32 com UART disponível. Copie `pid.py` e `main.py` para a raiz do dispositivo e reinicie-o.
2. Conecte um adaptador USB–TTL de 3,3 V com terra comum e TX/RX cruzados. O código seleciona UART 2, TX GPIO 17 e RX GPIO 16. Ajuste os pinos conforme o modelo da placa. A configuração padrão é 115200 baud, 8N1.
3. O hospedeiro envia uma linha ASCII `seq,referência,medição\n` a cada 20 ms, com `seq` inteiro não negativo e estritamente crescente. O dispositivo responde `seq,comando\n` ou `ERR,FRAME\n`.
4. Use o hospedeiro serial para ler o comando, medir o tempo de ida e volta e aplicar o atraso solicitado à planta virtual.

```sh
uv run --locked --with 'pyserial==3.5' python -m firmware.hil.serial_host \
  --port /dev/cu.usbserial-EXEMPLO --delay 0.80 --jitter 0.10 \
  --output results/hil-bancada.csv
```

O ensaio físico exige conferir a disponibilidade e pinagem da placa, a cadência real de troca de quadros, o tempo de ida e volta e a diferença entre a latência solicitada e a efetiva. A execução de CPython cobre a lei discreta e o protocolo ASCII. Compilação e temporização no dispositivo seguem pendentes de bancada.

[Documentação UART MicroPython](https://docs.micropython.org/en/latest/library/machine.UART.html) · [API pySerial](https://pyserial.readthedocs.io/en/latest/pyserial_api.html)
