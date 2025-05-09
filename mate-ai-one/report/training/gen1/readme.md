# Gen 1
si nota come i risultati miglio siano stati ottenuti da le seguenti combinazioni:
- 0_0_0 -> R^2 0.7947
- 0_4_1 -> R^2 0.7914
- 0_4_2 -> R^2 0.7927

il primo 0_ indica l'input size 768, i successivi indicano l'hidden_sizes:
- 0_ [128,61,32]
- 4_ [512,124]

mentre l'ultimo numero sono i dropouts:
- 0_ 0.1
- 1_ 0.2
- 2_ 0.3

Quindi provvederemo a rimuovere le altre input size, il dropout=0.4 e si creeranno altri valori per l'hidden_size
tutte le liste
hidden_sizes=[[128,61,32],[512, 32, 32],[256,128],[512,124],[1024,512]]
dropouts=[0.1,0.2,0.3,0.4]
input_sizes=[768,1536,45056,82048]