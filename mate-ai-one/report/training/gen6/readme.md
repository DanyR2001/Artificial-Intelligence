# Gen 5
si nota che si è prese quelle che hanno superato r^2>0.8:
- 0_0_1 -> R^2 0.8043
- 0_0_2 -> R^2 8045
- 0_1_1 -> R^2 0.8009

il primo 0_ indica l'input size 768, i successivi indicano l'hidden_sizes:
- 0_ [1024,512,256,128]
- 1_ [1024,512,256,128,64]

mentre l'ultimo numero sono i dropouts:
- 0 0.0
- 1 0.1
- 2 0.2

tutti le list:
hidden_sizes = [[1024,512,256,128],[1024,512,256,128,64],[512,128,64],[1024,512,256,128,64,32]]
dropouts = [0.0,0.1, 0.2, 0.3]
input_sizes = [768, 1536, 45056, 82048]