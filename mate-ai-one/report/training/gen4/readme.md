# Gen 4
si nota che si è prese quelle che hanno superato r^2>0.8:
- 0_2_0 -> R^2 0.8079

il primo 0_ indica l'input size 768, i successivi indicano l'hidden_sizes:
- 2_ [1024,512,256,128,64,32]

mentre l'ultimo numero sono i dropouts:
- 0 0.1

Quindi provvederemo a rimuovere le altre input size, il dropout=0.4 e si creeranno altri valori per l'hidden_size
tutti le list:
hidden_sizes=[[2048,1024,512,256,128],[1024,512,256,128],[1024,512,256,128,64,32],[1024,512,256,128,64,32,16],[1024,512,256,128,64],[512, 256, 124],[512,128,64]]
dropouts=[0.1,0.2,0.3]
input_sizes=[768]