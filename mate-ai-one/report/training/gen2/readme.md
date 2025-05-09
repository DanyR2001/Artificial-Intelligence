# Gen 2
si nota come i risultati miglio siano stati ottenuti da le seguenti combinazioni:
- 0_1_0 -> R^2 0.7904
- 0_3_2 -> R^2 0.7930

il primo 0_ indica l'input size 768, i successivi indicano l'hidden_sizes:
- 1_ [512, 256, 124],
- 3_ [512,124]

mentre l'ultimo numero sono i dropouts:
- 0_ 0.1
- 2_ 0.3

Quindi provvederemo a rimuovere le altre input size, il dropout=0.4 e si creeranno altri valori per l'hidden_size
tutti le list:
hidden_sizes=[[128,64,32],[512, 256, 124],[512,256],[512,124]]
dropouts=[0.1,0.2,0.3]
input_sizes=[768]