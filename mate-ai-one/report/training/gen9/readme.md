# Gen 9
si nota che si è prese quelle che hanno superato r^2>0.8:
- 0_0_0_5000_50 -> R^2 0.8217

il primo 0_ indica l'input size 768, i successivi indicano l'hidden_sizes:
- 1_ [1024,512,256,128]

mentre il successivo numero sono i dropouts:
- 0 0.1

il penultimo indica la size dal dataset in input ovvero 5000 esempi e il successivo sono in numero di epoche ovvero 50
tutti le list:
sizeCSV=[5000]
hidden_sizes = [[1024,512,256,128],[1024,1024,1024,1024,1024],[768,768,768,768,128]]
dropouts = [0.1, 0.2]
input_sizes = [769]
epochs = [50]