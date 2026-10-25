| Run | Arquitectura | Hidden | lr | Parámetros | Mejor época / épocas | Val loss | Exactitud nota | Exactitud paso | Tiempo (s) |
|-----|--------------|-------:|----:|-----------:|:----:|:----:|:----:|:----:|----:|
| E1 | encoder-decoder LSTM | 128 | 0.001 | 266,752 | 20 / 20 | 0.7494 | 0.7896 | 0.5985 | 32 |
| E2 | encoder-decoder GRU | 128 | 0.001 | 217,088 | 19 / 20 | 0.6339 | 0.8158 | 0.6382 | 54 |
| E3 | encoder-decoder GRU + atención aditiva | 128 | 0.001 | 299,136 | 17 / 20 | 0.7181 | 0.7908 | 0.6058 | 123 |
| E4 | encoder-decoder GRU | 256 | 0.001 | 628,224 | 12 / 16 | 0.6057 | 0.8217 | 0.6451 | 74 |
