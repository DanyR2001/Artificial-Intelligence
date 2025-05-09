# Gen 8
si nota che si è prese quelle che hanno superato r^2>0.8:
- 0_1_0_4000_100 -> R^2 0.8769
- 0_1_1_4000_100 -> R^2 0.8653
- 0_2_1_4000_100 -> R^2 0.8771

il primo 0_ indica l'input size 768, i successivi indicano l'hidden_sizes:
- 1_ [1024,1024,1024,1024,1024]
- 2_ [768,768,768,768,128]

mentre il successivo numero sono i dropouts:
- 0 0.0
- 1 0.1

il penultimo indica la size dal dataset in input ovvero 4000 esempi e il successivo sono in numero di epoche ovvero 100

tutti le list:
sizeCSV=[4000]
hidden_sizes = [[1024,512,256,128],[1024,1024,1024,1024,1024],[768,768,768,768,128],[768,512,512,128,32],[768,512,128,32,32]]
dropouts = [0.1, 0.2]
input_sizes = [768]
epochs = [40,100]